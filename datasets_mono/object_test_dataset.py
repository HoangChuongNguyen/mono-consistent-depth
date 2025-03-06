import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class CityscapeObjectTestDataset(torch.utils.data.Dataset):
    def __init__(self, file_names, image_dir, mask_dir, mask_size_threshold=0.005, height=192, width=512):
        self.file_names = file_names
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_size_threshold = mask_size_threshold
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        city_id, frame_id = self.file_names[index].split(" ")
        # Read image
        image = plt.imread(f'{self.image_dir}/{city_id}/{frame_id}_leftImg8bit.png')
        image = cv2.resize(image[:768], (self.width, self.height))
        # Read dynamic mask
        if not os.path.isdir(f'{self.mask_dir}/{city_id}/{frame_id}'):
            mask_list = np.zeros((1, self.height, self.width))
        else:
            mask_file_list = os.listdir(f'{self.mask_dir}/{city_id}/{frame_id}')
            mask_file_list = [file for file in mask_file_list if file.endswith('.png')]
            mask_list = [plt.imread(f'{self.mask_dir}/{city_id}/{frame_id}/{i}.png')[:768] for i in range(len(mask_file_list))]
            mask_list = [cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST) for mask in mask_list ]
            mask_list = [mask for mask in mask_list if np.sum(mask)>=self.mask_size_threshold*mask.shape[0]*mask.shape[1]]
            
            if len(mask_list) == 0: mask_list = np.zeros((1, self.height, self.width))
            else: mask_list = np.stack(mask_list)
        image = torch.from_numpy(image).permute(2,0,1) # 3 192 512
        mask_list = torch.from_numpy(mask_list).unsqueeze(1) # N 1 192 512
        return image, mask_list, frame_id