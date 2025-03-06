
import os
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
import torchvision
 
 
class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, file_lists, image_dir, mask_dir, ground_mask_dir, depth_dir, height, width, is_train=False):
        self.file_lists = file_lists
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ground_mask_dir = ground_mask_dir
        self.depth_dir = depth_dir
        self.height = height
        self.width = width
        self.is_train = is_train

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):
        city_id, frame_id = self.file_lists[index].split(" ")

        # Load RGB image
        img = plt.imread(f"{self.image_dir}/{city_id}/{frame_id}.png")[:,:,:3]
        _, img, _ = np.split(img, indices_or_sections=3, axis=1)
        img = cv2.resize(img, (self.width, self.height))
        
        # Load depth data
        depth = np.load(f"{self.depth_dir}/{city_id}/{frame_id}.npy")[0]
        depth = np.expand_dims(depth, axis=-1)
        disp = 1/depth
        disp = disp / np.max(disp)

        road_mask = plt.imread(f'{self.ground_mask_dir}/{city_id}/{frame_id}.png') # H W

        # Combine RGB and depth
        img = np.dstack((img, disp, np.expand_dims(road_mask,-1)))
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Load road mask
        # road_mask = plt.imread(f'{self.ground_mask_dir}/{city_id}/{frame_id}.png') # H W
        road_mask = (torch.from_numpy(road_mask).unsqueeze(0) >= 0.5).float()

        if self.mask_dir is not None:
            mask_files = os.listdir(f"{self.mask_dir}/{city_id}/{frame_id}")
            masks = [plt.imread(f"{self.mask_dir}/{city_id}/{frame_id}/{file}") for file in mask_files]
            masks = [mask for mask in masks if np.sum(mask) > 100]
            if len(masks) == 0: masks = [np.zeros((self.height,self.width))]
            masks = torch.stack([torch.tensor(np.array(mask), dtype=torch.uint8) for mask in masks])
            if self.is_train and np.random.rand() < 0.5:
                img = torchvision.transforms.functional.hflip(img)
                masks =  torchvision.transforms.functional.hflip(masks)
            boxes = []
            for mask in masks:
                try:
                    pos = np.where(np.array(mask))
                    xmin, ymin = np.min(pos[1]), np.min(pos[0])
                    xmax, ymax = np.max(pos[1]), np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                except: boxes.append([1, 1, 3, 3]) # Fix later here
            if len(boxes) != 0: boxes = np.stack(boxes)
        else:
            masks = np.zeros((1,self.height,self.width))
            boxes = np.zeros((1,self.height,self.width))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])

        # Create dummy labels with all 1s
        num_objs = len(masks)
        labels = torch.ones((num_objs), dtype=torch.int64)
        target = {"boxes": boxes, "masks": masks, "image_id": image_id, "labels": labels}
        return img, target, road_mask, frame_id
