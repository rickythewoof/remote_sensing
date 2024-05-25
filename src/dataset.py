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
                mean = None,
                std = None,
                dtype = "PS-RGB"):
        assert(split in ["train", "val", "test"] and "error with split")
        self.root_dir = root_dir
        self.split = split
        self.dtype = dtype
        mean, std = 0 ,5
        self.transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std),
            transforms
        ])
        self.img_dir = []
        self.lbl_dir = []

        split_file = open(os.path.join(root_dir,f"splits/{split}.txt"), "r")
        for line in split_file:
            self.img_dir.append(line.strip())
            self.lbl_dir.append(line.strip().replace(f"/{dtype}/", "/geojson_buildings/").replace(f"{dtype}","Buildings").replace(".tif", ".geojson"))
        split_file.close()


    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = rasterio.open(self.img_dir[idx]).read().astype(np.float32).transpose(1,2,0)
        
        mask = utils.create_segmentation_mask(image, self.lbl_dir[idx]) # If failing wait for the professor's mask creation function

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        
        return image, mask
