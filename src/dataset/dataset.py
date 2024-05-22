import torch 
import numpy as np
import os
import torch.utils.data as data
import rasterio

class SN6Dataset(data.Dataset):

    def __init__(self, root_dir = None, 
                split = "train",
                transform = None):
        self.__init__() 
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = []
        self.lbl_dir = []
        
        split_file = os.open(os.join(root_dir,f"data/splits/{split}.txt"), "r")
        for line in split_file:
            self.img_dir.append(line.strip())
            # TO BE CHANGED WITH DATA FROM TUTOR
            self.lbl_dir.append(line.strip().replace("/PS-RGBNIR/", "/geojson_buildings/").replace("PS-RGBNIR", "Buildings").replace(".tif", ".geojson"))
        split_file.close()


    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = rasterio.open(self.img_dir[idx]).read()
        label = rasterio.open(self.lbl_dir[idx]).read()
        return image, label
