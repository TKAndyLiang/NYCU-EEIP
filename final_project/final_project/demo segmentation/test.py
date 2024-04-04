import torch
from natsort import natsorted
from glob import glob
from PIL import Image
import cv2
import numpy as np
import os, sys
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from dataset import WaterDataset_Test
from torch.utils.data import DataLoader
from model import UNET, NAFNet
from Uformer import Uformer
from utils import (
    load_checkpoint,
    load_checkpoint_multigpu
)

multi_gpu = True
test_model = 'Uformer'
test_dir = 'test/image'
result_dir = 'test/result'
# check_point = 'checkpoint_maxpool_384_v2.pth.tar'
check_point = 'checkpoint_best_Uformer_v2.pth.tar'
cuda_device = 'cuda:0'


def demo_fast(img, model):
    # test the image tile by tile
    b, c, h, w = img.size()
    tile = min(384, h, w)
    tile_overlap = 0
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(img)
    W = torch.zeros_like(E)
    
    in_patch = []
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch.append(img[..., h_idx:h_idx+tile, w_idx:w_idx+tile].squeeze(0))

    in_patch = torch.stack(in_patch, 0)
    # print(in_patch.shape)
    
    out_patch = model(in_patch)
    
    for ii, h_idx in enumerate(h_idx_list):
        for jj, w_idx in enumerate(w_idx_list):
            idx = ii * len(w_idx_list) + jj
            # print(idx)
            out_patch_mask = torch.ones_like(out_patch[idx])

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch[idx])
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            
            
    output = E.div_(W)

    return output


def patch_forward(img, model, tile=384, padding=8, tile_overlap=16, device="cuda"):
    ''' 
    Args:
        img: b c h w
        model: denoising model
        tile: patch size
        padding: overlapping / 2
    '''
    assert img.is_cuda
    pd = padding
    stride = tile - tile_overlap # 368

    img = F.pad(img, (pd, pd, pd, pd), mode='reflect')
    b, c, h, w = img.size()
    h_idx_list = [i * stride for i in range(math.ceil(h / stride))]
    w_idx_list = [i * stride for i in range(math.ceil(w / stride))]
    
    new_h = h_idx_list[len(h_idx_list)-1] + tile
    new_w = w_idx_list[len(w_idx_list)-1] + tile

    E = torch.zeros(b, c, new_h, new_w).type_as(img)
    W = torch.zeros_like(E)

    in_patch = []
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile].squeeze(0)
            store_patch = torch.zeros(c, tile, tile).to(device)
            store_patch[:, 0:patch.shape[1], 0:patch.shape[2]] = patch
            in_patch.append(store_patch)

    in_patch = torch.stack(in_patch, 0)
    out_patch = model(in_patch)
    
    for ii, h_idx in enumerate(h_idx_list):
        for jj, w_idx in enumerate(w_idx_list):
            idx = ii * len(w_idx_list) + jj
            # print(idx)
            out_patch_mask = torch.ones_like(out_patch[idx])
            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch[idx])
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)

    output = E.div_(W)

    return output[..., pd:h-pd, pd:w-pd]
    # return output[..., 0:h, 0:w]


def save_predictions_as_imgs(loader, model, folder='', device="cuda"):
    model.eval()
    for idx, x in enumerate(tqdm(loader)):
        x = x.to(device=device)
        # torchvision.utils.save_image(x, f"{folder}/output_ori{idx+1}.jpg")
        with torch.no_grad():
            # preds = torch.sigmoid(demo_fast(x, model))
            preds = torch.sigmoid(patch_forward(x, model, 384, 0, 8, device=DEVICE))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/output{idx+1}.jpg")


if __name__ == '__main__':
    # prepare data
    test_transforms = A.Compose([
            # A.GaussianBlur((3, 7), 0, p=1.0),
            # A.HorizontalFlip(p=1),
            # A.Rotate(15, p=1.0),
            # A.ColorJitter(p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    test_ds = WaterDataset_Test(image_dir=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=True)

    # prepare model
    
    DEVICE = cuda_device if torch.cuda.is_available() else "cpu"
    if test_model == 'UNET':
        # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        model = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
    elif test_model == 'NAFNet':
        enc_blks = [2, 2, 2, 2]
        middle_blk_num = 2
        dec_blks = [2, 2, 2, 2]
        model = NAFNet(img_channel=3, width=32, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(DEVICE)
    elif test_model == 'Uformer':
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        num_heads = [1, 2, 4, 8, 8, 8, 4, 2, 1]
        # model = Uformer(img_size=384, in_chan=3, out_chan=1,
        #                             embed_dim=16, depths=depths, num_heads=num_heads,
        #                             win_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
        #                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        #                             norm_layer=nn.LayerNorm, patch_norm=True,
        #                             token_projection='linear', token_mlp='leff',
        #                             shift_flag=True).to(DEVICE)
        model = Uformer(img_size=384, in_chan=3, out_chan=1,
                                    embed_dim=32, depths=depths, num_heads=num_heads,
                                    win_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                    drop_rate=0.05, attn_drop_rate=0.05, drop_path_rate=0.05,
                                    norm_layer=nn.LayerNorm, patch_norm=True,
                                    token_projection='linear', token_mlp='mlp',
                                    shift_flag=True).to(DEVICE)
        
    if multi_gpu ==True:
        load_checkpoint_multigpu(torch.load(check_point), model)
    else:
        load_checkpoint(torch.load(check_point), model)

    # start inference
    if os.path.exists(result_dir):
        os.system("rm -rf {}".format(result_dir))
        os.system("mkdir {}".format(result_dir))

    save_predictions_as_imgs(test_loader, model, result_dir, device=DEVICE)