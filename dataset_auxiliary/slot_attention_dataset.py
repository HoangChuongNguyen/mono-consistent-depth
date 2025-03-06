

import torch
from torchvision import transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision.utils import flow_to_image


def rescale(x):
    return x * 2 - 1

class SlotAttentionDataset(torch.utils.data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self, image_dir, depth_dir, flow_neg_dir, flow_pos_dir, file_list, height, width, flow_threshold, is_train=True):
        super(SlotAttentionDataset, self).__init__()
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.flow_neg_dir = flow_neg_dir
        self.flow_pos_dir = flow_pos_dir
        self.file_list = file_list
        self.height = height
        self.width = width
        self.flow_threshold = flow_threshold
        self.is_train = is_train
        resolution = (height, width)
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(rescale),  # rescale between -1 and 1
                transforms.Resize(resolution)
                ])

        self.transforms_no_scale = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(resolution)
                    ])
            

    def __len__(self):
        return len(self.file_list)

    def transform_depth_exp(self, depth, flow_neg_mask, flow_pos_mask):
        # Normalize depth and apply exponential transformation
        depth = depth / np.max(depth)
        depth = np.exp(depth) - 1
        depth_neg, depth_pos = np.copy(depth), np.copy(depth)
        depth_neg[0][flow_neg_mask[0,:,:] == 0] = np.max(depth_neg)
        depth_pos[0][flow_pos_mask[0,:,:] == 0] = np.max(depth_pos)
        depth_neg = depth_neg / (np.max(depth_neg))
        depth_pos = depth_pos / (np.max(depth_pos))
        return depth_neg.reshape(depth.shape[1], depth.shape[2], 1), depth_pos.reshape(depth.shape[1], depth.shape[2], 1)

    def convert_flow_to_image(self, flow):
        flow_image = flow_to_image(torch.from_numpy(flow).float().unsqueeze(0))[0].permute(1,2,0).numpy() # 192 512 3
        return flow_image


    def __getitem__(self, index):
        city_id, frame_id = self.file_list[index].split(' ')
        # Get image, flow image, depth (both neg and pos)
        # Get image
        _, image, _ = np.split(plt.imread(f'{self.image_dir}/{city_id}/{frame_id}.png')[:,:,:3], axis=1, indices_or_sections=3)
        image = cv2.resize(image, (self.width, self.height)) # 192 512 3

        # Get flow_image
        flow_neg = np.load(f'{self.flow_neg_dir}/{city_id}/{frame_id}.npy') # 2 192 512
        flow_pos = np.load(f'{self.flow_pos_dir}/{city_id}/{frame_id}.npy') # 2 192 512
        flow_neg_dynamic_mask = (np.sum(np.abs(flow_neg), axis=0, keepdims=True) >= self.flow_threshold).astype(float) # 1 192 512
        flow_pos_dynamic_mask = (np.sum(np.abs(flow_pos), axis=0, keepdims=True) >= self.flow_threshold).astype(float) # 1 192 512
        flow_neg = flow_neg * flow_neg_dynamic_mask
        flow_pos = flow_pos * flow_pos_dynamic_mask
        flow_image_neg = self.convert_flow_to_image(flow_neg) # 192 512 3
        flow_image_pos = self.convert_flow_to_image(flow_pos) # 192 512 3

        # Get depth
        depth = np.load(f'{self.depth_dir}/{city_id}/{frame_id}.npy') # 1 192 512
        depth_neg_exp, depth_pos_exp = self.transform_depth_exp(depth, flow_neg_dynamic_mask, flow_pos_dynamic_mask) # 192 512 1 
        flow_neg_dynamic_mask = flow_neg_dynamic_mask.reshape(depth_neg_exp.shape)
        flow_pos_dynamic_mask = flow_pos_dynamic_mask.reshape(depth_neg_exp.shape)
        # Random horizontal flip
        if self.is_train and np.random.rand() < 0.5:
            image = np.fliplr(image)
            flow_image_neg = np.fliplr(flow_image_neg)
            flow_image_pos = np.fliplr(flow_image_pos)
            flow_neg_dynamic_mask = np.fliplr(flow_neg_dynamic_mask)
            flow_pos_dynamic_mask = np.fliplr(flow_pos_dynamic_mask)
            depth_neg_exp = np.fliplr(depth_neg_exp)
            depth_pos_exp = np.fliplr(depth_pos_exp)

        return (frame_id, self.transforms(np.copy(image)), 
                self.transforms(np.copy(flow_image_neg)), self.transforms(np.copy(flow_image_pos)), 
                self.transforms_no_scale(np.copy(flow_neg_dynamic_mask)), self.transforms_no_scale(np.copy(flow_pos_dynamic_mask)), 
                self.transforms_no_scale(np.copy(depth_neg_exp)), self.transforms_no_scale(np.copy(depth_pos_exp)))

