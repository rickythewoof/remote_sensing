import numpy as np
import os
import utils
import torch.utils.data as data
from torchvision import transforms
import rasterio

class SN6Dataset(data.Dataset):

    def __init__(self,
                root_dir = None, 
                split = "train",
                dtype = "PS-RGBNIR"):
        self.root_dir = root_dir
        self.split = split
        self.dtype = dtype
        mean, std = 0 ,5
        self.transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])
        self.img_dir = []
        self.lbl_dir = []

        split_file = open(os.path.join(root_dir,f"splits/{split}.txt"), "r")
        for line in split_file:
            self.img_dir.append(line.strip())
            self.lbl_dir.append(line.strip())   # TO BE CHANGED WITH DATA FROM TUTOR
        split_file.close()


    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = rasterio.open(self.img_dir[idx]).read().astype(np.float32)
        
        label = rasterio.open(self.lbl_dir[idx]).read(1) # TO BE CHANGED WITH DATA FROM TUTOR
        
        # TODO: Do something with the label
        mask = print("something something")

        if(self.split == "train"):      # Compute data augmentation only for the training set
            self.transform.transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            self.transform.transforms.append(transforms.RandomVerticalFlip(p=0.5))

        image, mask = self.transform(image, mask)
        return image, label
