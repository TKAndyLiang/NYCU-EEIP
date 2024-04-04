import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted
from glob import glob


class WaterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)
        self.images = natsorted(os.listdir(image_dir))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        # image = np.array(cv2.imread(img_path, cv2.COLOR_BGR2RGB))
        # mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).reshape(image.shape[0], image.shape[1], 1)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    

class WaterDataset_Test(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)
        self.images = natsorted(os.listdir(image_dir))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image
    

