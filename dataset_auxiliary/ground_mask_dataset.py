

import torch
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter

class GroundMaskDataset(Dataset):
    def __init__(self, image_dir, depth_dir, file_list, height, width, is_train=False):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.file_list = file_list
        self.height = height
        self.width = width
        self.is_train = is_train
        if is_train:
            self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        city_id, frame_id = self.file_list[idx].split(' ')

        # Load image
        _, image, _ = np.split(plt.imread(f'{self.image_dir}/{city_id}/{frame_id}.png')[:,:,:3], axis=1, indices_or_sections=3)
        depth = np.load(f'{self.depth_dir}/{city_id}/{frame_id}.npy')
        K = np.loadtxt(f'{self.image_dir}/{city_id}/{frame_id}_cam.txt', dtype=float, delimiter=',').reshape(3,3)
        original_height, original_width, _ = image.shape

        # Load depth

        image = cv2.resize(image, (self.width, self.height))
        K[0] = K[0]*(self.width/original_width)
        K[1] = K[1]*(self.height/original_height)

        # Apply color jitter
        if self.is_train and np.random.rand() < 0.5:
            image = np.array(self.color_jitter(Image.fromarray((image * 255).astype(np.uint8))))
            image = image / 255.0

        # Random horizontal flip
        if self.is_train and np.random.rand() < 0.5:
            image = np.fliplr(image)
            depth = np.ascontiguousarray(np.expand_dims(np.fliplr(depth[0]),0))

        # Convert to pytorch tensor
        image = torch.from_numpy(np.copy(image)).permute(2,0,1).float() # 3 192 512
        K = torch.from_numpy(K)

        return image, depth, K, frame_id
