from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
import argparse
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser(description = 'Generate patches from SIDD full resolution images')
parser.add_argument('--src_dir', default = '/local_data/M112cyliang/EEIP_data', type = str, help = 'Directory for full resolution images')
parser.add_argument('--tar_dir', default = '/local_data/M112cyliang/EEIP_data', type = str, help = 'Directory for image patches')
parser.add_argument('--ps', default = 384, type = int, help = 'Image Patch Size')
parser.add_argument('--num_cores', default = 10, type = int, help = 'Number of CPU Cores')
args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_CORES = args.num_cores

# get sorted folders
image_files = natsorted(glob(os.path.join(src, 'image', '*.jpg')))
mask_files = natsorted(glob(os.path.join(src, 'mask', '*.jpg')))

# training
image_patchDir_train = os.path.join(tar, 'train_image_new')
mask_patchDir_train = os.path.join(tar, 'train_mask_new')

if os.path.exists(image_patchDir_train):
    os.system("rm -rf {}".format(image_patchDir_train))

if os.path.exists(mask_patchDir_train):
    os.system("rm -rf {}".format(mask_patchDir_train))

os.makedirs(image_patchDir_train)
os.makedirs(mask_patchDir_train)

# validation
image_patchDir_val = os.path.join(tar, 'val_image_new')
mask_patchDir_val = os.path.join(tar, 'val_mask_new')

if os.path.exists(image_patchDir_val):
    os.system("rm -rf {}".format(image_patchDir_val))

if os.path.exists(mask_patchDir_val):
    os.system("rm -rf {}".format(mask_patchDir_val))

os.makedirs(image_patchDir_val)
os.makedirs(mask_patchDir_val)

# for val
pick = np.random.choice(len(image_files), 5, replace=False)
print(pick)
image_files_val = [image_files[id] for id in pick]
mask_files_val = [mask_files[id] for id in pick]

# for train
image_files_train = [image_files[id] for id in range(len(image_files)) if id not in pick]
mask_files_train= [mask_files[id] for id in range(len(image_files)) if id not in pick]
# image_files_train = image_files
# mask_files_train= mask_files

def patching(i, imagefiles, maskfiles, store_image_patchDir, store_mask_patchDir, num_patches):
    image_file, mask_file = imagefiles[i], maskfiles[i]
    image = np.array(cv2.imread(image_file))
    mask = np.array(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)).reshape(image.shape[0], image.shape[1], 1)

    H = image.shape[0]
    W = image.shape[1]

    rr = np.random.choice(H - PS, num_patches, replace=False)
    cc = np.random.choice(W - PS, num_patches, replace=False)

    for j in range(num_patches):
        image_patch = image[rr[j]:rr[j] + PS, cc[j]:cc[j] + PS, :]
        mask_patch = mask[rr[j]:rr[j] + PS, cc[j]:cc[j] + PS, :]
        
        cv2.imwrite(os.path.join(store_image_patchDir, '{}_{}.png'.format(i+1, j+1)), image_patch)
        cv2.imwrite(os.path.join(store_mask_patchDir, '{}_{}.png'.format(i+1, j+1)), mask_patch)


def cutmix_H(num_mix, imagefiles, maskfiles, store_image_patchDir, store_mask_patchDir):
    id = np.random.choice(len(imagefiles), 2, replace=False)
    half_ps = PS // 2

    img1_file = imagefiles[id[0]]
    mask1_file = maskfiles[id[0]]
    img2_file = imagefiles[id[1]]
    mask2_file = maskfiles[id[1]]

    img1 = np.array(cv2.imread(img1_file))
    mask1 = np.array(cv2.imread(mask1_file, cv2.IMREAD_GRAYSCALE)).reshape(img1.shape[0], img1.shape[1], 1)
    H1, W1 = img1.shape[0], img1.shape[1]
    rr1 = np.random.choice(H1 - PS, num_mix, replace=False)
    cc1 = np.random.choice(W1 - half_ps, num_mix, replace=False)

    img2 = np.array(cv2.imread(img2_file))
    mask2 = np.array(cv2.imread(mask2_file, cv2.IMREAD_GRAYSCALE)).reshape(img2.shape[0], img2.shape[1], 1)
    H2, W2 = img2.shape[0], img2.shape[1]
    rr2 = np.random.choice(H2 - PS, num_mix, replace=False)
    cc2 = np.random.choice(W2 - half_ps, num_mix, replace=False)

    for i in range(num_mix):
        image_patch = np.zeros((PS, PS, 3))
        mask_patch = np.zeros((PS, PS, 1))

        image_patch[0:PS, 0:half_ps, :] = img1[rr1[i]:rr1[i] + PS, cc1[i]:cc1[i] + half_ps, :]
        image_patch[0:PS, half_ps:PS, :] = img2[rr2[i]:rr2[i] + PS, cc2[i]:cc2[i] + half_ps, :]

        mask_patch[0:PS, 0:half_ps, :] = mask1[rr1[i]:rr1[i] + PS, cc1[i]:cc1[i] + half_ps, :]
        mask_patch[0:PS, half_ps:PS, :] = mask2[rr2[i]:rr2[i] + PS, cc2[i]:cc2[i] + half_ps, :]

        cv2.imwrite(os.path.join(store_image_patchDir, '{}_{}_{}_mix_H.png'.format(id[0], id[1], i)), image_patch)
        cv2.imwrite(os.path.join(store_mask_patchDir, '{}_{}_{}_mix_H.png'.format(id[0], id[1], i)), mask_patch)

def cutmix_V(num_mix, imagefiles, maskfiles, store_image_patchDir, store_mask_patchDir):
    id = np.random.choice(len(imagefiles), 2, replace=False)
    half_ps = PS // 2

    img1_file = imagefiles[id[0]]
    mask1_file = maskfiles[id[0]]
    img2_file = imagefiles[id[1]]
    mask2_file = maskfiles[id[1]]

    img1 = np.array(cv2.imread(img1_file))
    mask1 = np.array(cv2.imread(mask1_file, cv2.IMREAD_GRAYSCALE)).reshape(img1.shape[0], img1.shape[1], 1)
    H1, W1 = img1.shape[0], img1.shape[1]
    rr1 = np.random.choice(H1 - half_ps, num_mix, replace=False)
    cc1 = np.random.choice(W1 - PS, num_mix, replace=False)

    img2 = np.array(cv2.imread(img2_file))
    mask2 = np.array(cv2.imread(mask2_file, cv2.IMREAD_GRAYSCALE)).reshape(img2.shape[0], img2.shape[1], 1)
    H2, W2 = img2.shape[0], img2.shape[1]
    rr2 = np.random.choice(H2 - half_ps, num_mix, replace=False)
    cc2 = np.random.choice(W2 - PS, num_mix, replace=False)

    for i in range(num_mix):
        image_patch = np.zeros((PS, PS, 3))
        mask_patch = np.zeros((PS, PS, 1))

        image_patch[0:half_ps, 0:PS, :] = img1[rr1[i]:rr1[i] + half_ps, cc1[i]:cc1[i] + PS, :]
        image_patch[half_ps:PS, 0:PS, :] = img2[rr2[i]:rr2[i] + half_ps, cc2[i]:cc2[i] + PS, :]

        mask_patch[0:half_ps, 0:PS, :] = mask1[rr1[i]:rr1[i] + half_ps, cc1[i]:cc1[i] + PS, :]
        mask_patch[half_ps:PS, 0:PS, :] = mask2[rr2[i]:rr2[i] + half_ps, cc2[i]:cc2[i] + PS, :]

        cv2.imwrite(os.path.join(store_image_patchDir, '{}_{}_{}_mix_V.png'.format(id[0], id[1], i)), image_patch)
        cv2.imwrite(os.path.join(store_mask_patchDir, '{}_{}_{}_mix_V.png'.format(id[0], id[1], i)), mask_patch)


# for val
Parallel(n_jobs=NUM_CORES)(delayed(patching)(i, image_files_val, mask_files_val, image_patchDir_val, mask_patchDir_val, 20) for i in tqdm(range(len(image_files_val))))
Parallel(n_jobs=NUM_CORES)(delayed(cutmix_H)(10, image_files_val, mask_files_val, image_patchDir_val, mask_patchDir_val) for i in tqdm(range(20)))
Parallel(n_jobs=NUM_CORES)(delayed(cutmix_V)(10, image_files_val, mask_files_val, image_patchDir_val, mask_patchDir_val) for i in tqdm(range(20)))

# for train
Parallel(n_jobs=NUM_CORES)(delayed(patching)(i, image_files_train, mask_files_train, image_patchDir_train, mask_patchDir_train, 80) for i in tqdm(range(len(image_files_train))))
Parallel(n_jobs=NUM_CORES)(delayed(cutmix_H)(10, image_files_train, mask_files_train, image_patchDir_train, mask_patchDir_train) for i in tqdm(range(1000)))
Parallel(n_jobs=NUM_CORES)(delayed(cutmix_V)(10, image_files_train, mask_files_train, image_patchDir_train, mask_patchDir_train) for i in tqdm(range(1000)))

