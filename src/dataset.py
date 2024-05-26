import numpy as np
import os
import utils
import torch.utils.data as data
from torchvision import transforms
import rasterio
import albumentations as A

class SN6Dataset(data.Dataset):

    def __init__(self,
                root_dir = None, 
                split = "train",
                transform = None,
                dtype = "PS-RGB"):
        assert(split in ["train", "val", "test"] and "error with split")
        self.root_dir = root_dir
        self.split = split
        self.dtype = dtype
        self.transform = transform
        self.img_dir = []
        self.lbl_dir = []
        self.msk_dir = []

        split_file = open(os.path.join(root_dir,f"splits/{split}.txt"), "r")
        for line in split_file:
            self.img_dir.append(line.strip())
            self.msk_dir.append(line.strip().replace(f"/{dtype}/", "/masks/").replace(".tif", ".png"))
            self.lbl_dir.append(line.strip().replace(f"/{dtype}/", "/geojson_buildings/").replace(f"{dtype}","Buildings").replace(".tif", ".geojson"))
        split_file.close()


    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = rasterio.open(self.img_dir[idx]).read().transpose(1,2,0).astype(np.float32)
        mask = rasterio.open(self.msk_dir[idx]).read().transpose(1,2,0).astype(np.float32)
        transforms = self.transform(image=image, mask=mask)
        image = transforms['image']
        mask = transforms['mask']
        # check if the image needs to be rotatedS
        if self.split == "train":
            image, mask = apply_augmentations(image, mask)
        image = image.transpose(2,0,1)
        mask = mask.transpose(2,0,1)
        
        return image, mask

def apply_augmentations(image, mask):
    # Define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])

    # Apply the augmentations to the image and mask
    augmented = transform(image=image, mask=mask)
    img_trsf = augmented['image']
    msk_trsf = augmented['mask']

    return img_trsf, msk_trsf