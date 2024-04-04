import torch
import os, sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, NAFNet, Baseline
from Uformer import Uformer
import argparse
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    load_optim,
    load_checkpoint_multigpu,
)

parser = argparse.ArgumentParser(description = 'Generate patches from SIDD full resolution images')
parser.add_argument('--learning_rate', '-lr', default = 1e-4, type = float)
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--epoch', default = 100, type = int)
parser.add_argument('--num_workers', default = 2, type = int)
parser.add_argument('--pretrained', default = 'False', type = str)
parser.add_argument('--resume', default = 'False', type = str)
parser.add_argument('--pretrained_weight', default = 'checkpoint_maxpool_384_v1.pth.tar', type = str)
parser.add_argument('--optimizer', default = 'Adam', type=str)
parser.add_argument('--model', default = 'UNET', type=str)
parser.add_argument('--gpu', default = '', type=str)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Hyperparameters
LEARNING_RATE = args.learning_rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epoch
NUM_WORKERS = args.num_workers
PIN_MEMORY = True
LOAD_MODEL = args.pretrained
RESUME = args.resume
TRAIN_IMG_DIR = "/home/u4988103/EEIP/segmentation/data/train_image/"
TRAIN_MASK_DIR = "/home/u4988103/EEIP/segmentation/data/train_mask"
VAL_IMG_DIR = "/home/u4988103/EEIP/segmentation/data/val_image/"
VAL_MASK_DIR = "/home/u4988103/EEIP/segmentation/data/val_mask/"
SAVED_IMAGE_DIR = "saved_images"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loopclear
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
            A.GaussianBlur((3, 7), 0, always_apply=False, p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    val_transforms = A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    if args.model == 'UNET':
        # model = UNET(in_channels=3, out_channels=1)
        model = UNET(in_channels=3, out_channels=1, features=[32, 64, 128, 256])

    elif args.model == 'NAFNet':
        enc_blks = [2, 2, 2, 2]
        middle_blk_num = 2
        dec_blks = [2, 2, 2, 2]
        model = NAFNet(img_channel=3, width=32, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        
    elif args.model == 'Baseline':
        enc_blks = [1, 2, 2, 2]
        middle_blk_num = 2
        dec_blks = [2, 2, 2, 1]
        model = Baseline(img_channel=3, width=32, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    
    elif args.model == 'Uformer':
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        num_heads = [1, 2, 4, 8, 8, 8, 4, 2, 1]
        model = Uformer(img_size=384, in_chan=3, out_chan=1,
                                    embed_dim=32, depths=depths, num_heads=num_heads,
                                    win_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                    drop_rate=0.05, attn_drop_rate=0.05, drop_path_rate=0.05,
                                    norm_layer=nn.LayerNorm, patch_norm=True,
                                    token_projection='linear', token_mlp='mlp',
                                    shift_flag=True)
        
    if args.gpu != '':
        model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=LEARNING_RATE, pct_start=0.03, div_factor=10, epochs=NUM_EPOCHS, steps_per_epoch=1, verbose=True)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL == 'True':
        load_checkpoint(torch.load(args.pretrained_weight), model)
    
    if RESUME == 'True':
        checkpoint = torch.load(args.pretrained_weight)
        load_checkpoint(checkpoint, model)
        load_optim(checkpoint, optimizer)
        start_epoch = checkpoint["epoch"] + 1
        for i in range(start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    acc = 0

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # check accuracy
        acc = check_accuracy(val_loader, model, device=DEVICE)

        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="./checkpoint_latest.pth.tar")
        
        if acc > best_acc:
            best_acc = acc
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="./checkpoint_best.pth.tar")

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder=SAVED_IMAGE_DIR, device=DEVICE)


if __name__ == "__main__":
    if os.path.exists(SAVED_IMAGE_DIR):
        os.system("rm -rf {}".format(SAVED_IMAGE_DIR))
        os.system("mkdir {}".format(SAVED_IMAGE_DIR))
    main()