import numpy as np
import os
import torch.utils.data as data
from torchvision import  transforms
import rasterio

class SN6Dataset(data.Dataset):

    def __init__(self, root_dir = None, 
                split = "train",
                dtype = "PS-RGBNIR",
                transform = None):
        self.root_dir = root_dir
        self.split = split
        self.dtype = dtype
        self.transform = transform
        self.img_dir = []
        self.lbl_dir = []

        split_file = open(os.path.join(root_dir,f"splits/{split}.txt"), "r")
        for line in split_file:
            self.img_dir.append(line.strip())
            # TO BE CHANGED WITH DATA FROM TUTOR
            self.lbl_dir.append(line.strip())
            # self.lbl_dir.append(line.strip().replace(f"/{dtype}/", "/geojson_buildings/").replace(f"{dtype}", "Buildings").replace(".tif", ".geojson"))
        split_file.close()


    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = rasterio.open(self.img_dir[idx]).read().astype(np.float32)
        label = rasterio.open(self.lbl_dir[idx]).read()

        mask = print("something something")

        if(self.split == "train"):
            # Augment the image and mask computing transformations
            None
        return image, label
    

    def compute_augmentations(self):
        self.transform.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        return self.transform