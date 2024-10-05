
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

class ObjectScaleDataset(Dataset):
    def __init__(self, image_path, depth_path, object_mask_path, road_mask_path, 
                object_predicted_flow_neg_path, object_predicted_flow_pos_path, flow_threshold, 
                height, width, file_lists, is_train=False):
        self.image_path = image_path
        self.depth_path = depth_path
        self.object_mask_path = object_mask_path
        self.road_mask_path = road_mask_path
        self.object_predicted_flow_neg_path = object_predicted_flow_neg_path
        self.object_predicted_flow_pos_path = object_predicted_flow_pos_path
        self.flow_threshold = flow_threshold
        self.file_lists = file_lists
        self.is_train = is_train
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.file_lists)


    def generate_bounding_box_mask(self, binary_mask):
        rows, cols = np.nonzero(binary_mask)
        min_row, min_col = np.min(rows), np.min(cols)
        max_row, max_col = np.max(rows), np.max(cols)
        bounding_box_mask = np.zeros_like(binary_mask)
        bounding_box_mask[min_row:max_row + 1, min_col:max_col + 1] = 1
        return bounding_box_mask


    def __getitem__(self, index):
        city_id, full_frame_id, mask_id, dynamic_label = self.file_lists[index].split(' ')
        # Get image
        image = plt.imread(f"{self.image_path}/{city_id}/{full_frame_id}.png")[...,:3]
        _, image, _ = np.split(image, indices_or_sections=3, axis=1)
        image = cv2.resize(image, (self.width, self.height))
        # Get depth
        depth = np.load(f"{self.depth_path}/{city_id}/{full_frame_id}.npy") # 1 192 512
        # Get all object masks
        # mask_file_list = os.listdir(f'{self.object_mask_path}/{city_id}/{full_frame_id}')
        # mask_file_list = [file for file in mask_file_list if file not in ['full_mask.png', 'labels.txt' ]]
        # mask_file_list = np.array(sorted(mask_file_list, key=lambda x: int(x[:-4])))
        # mask_list = np.stack([plt.imread(f'{self.object_mask_path}/{city_id}/{full_frame_id}/{mask_file}') for mask_file in mask_file_list]) # N 192 512
        # # Get road masks
        road_mask = (plt.imread(f'{self.road_mask_path}/{city_id}/{full_frame_id}.png') >= 0.5).astype(float)
        # Get target object mask
        object_mask = plt.imread(f'{self.object_mask_path}/{city_id}/{full_frame_id}/{mask_id}.png') # 192 512
        object_bbox_mask = self.generate_bounding_box_mask(object_mask) 
        # Get mask of dynamic region
        predicted_flow_neg = np.load(f'{self.object_predicted_flow_neg_path}/{city_id}/{full_frame_id}.npy')
        predicted_flow_pos = np.load(f'{self.object_predicted_flow_pos_path}/{city_id}/{full_frame_id}.npy')
        dynamic_region_mask = (np.sum(np.abs(predicted_flow_neg) + np.abs(predicted_flow_pos), axis=0)/2 >= self.flow_threshold).astype(float)
        # Transform to torch
        image = torch.from_numpy(image).permute(2,0,1).float() # 3 192 512
        depth = torch.from_numpy(depth).float() # 1 192 512
        object_mask = torch.from_numpy(object_mask).unsqueeze(0).float() # 1 192 512
        dynamic_region_mask = torch.from_numpy(dynamic_region_mask).unsqueeze(0).float() # 1 192 512
        road_mask = torch.from_numpy(road_mask).unsqueeze(0).float() # 1 192 512
        object_bbox_mask = torch.from_numpy(object_bbox_mask).unsqueeze(0).float() # 1 192 512

        if self.is_train and np.random.random() < 0.5:
            image = torch.flip(image, [2])
            depth = torch.flip(depth, [2])
            object_mask = torch.flip(object_mask, [2])
            dynamic_region_mask = torch.flip(dynamic_region_mask, [2])
            road_mask = torch.flip(road_mask, [2])
            object_bbox_mask = torch.flip(object_bbox_mask, [2])

        return image, depth, object_mask, object_bbox_mask, road_mask, dynamic_region_mask, full_frame_id


class CityScapesObjectScaleEvalDataset(Dataset):
    def __init__(self, image_path, depth_path, object_mask_path, road_mask_path, 
                object_predicted_flow_neg_path, object_predicted_flow_pos_path, flow_threshold, height, width, filenames, middle_portion=20, is_train=False):
        self.image_path = image_path
        self.depth_path = depth_path
        self.object_mask_path = object_mask_path
        self.road_mask_path = road_mask_path
        self.object_predicted_flow_neg_path = object_predicted_flow_neg_path
        self.object_predicted_flow_pos_path = object_predicted_flow_pos_path
        self.flow_threshold = flow_threshold
        self.filenames = filenames
        self.is_train = is_train
        self.height = height
        self.width = width

        self.middle_mask = np.zeros((self.height,self.width))
        self.middle_mask[:,int(self.width*(middle_portion-1)/(2*middle_portion)):int(512*(middle_portion-1)/(2*middle_portion))] = 1
        self.color_augmentation = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __len__(self):
        return len(self.filenames)

    def generate_bounding_box_mask(self, binary_mask):
        rows, cols = np.nonzero(binary_mask)
        min_row, min_col = np.min(rows), np.min(cols)
        max_row, max_col = np.max(rows), np.max(cols)
        bounding_box_mask = np.zeros_like(binary_mask)
        bounding_box_mask[min_row:max_row + 1, min_col:max_col + 1] = 1
        return bounding_box_mask

    def __getitem__(self, index):
        city_id, full_frame_id, mask_id = self.filenames[index].split(' ')
        mask_id_list = mask_id.split(',')
        # Get image
        image = plt.imread(f"{self.image_path}/{city_id}/{full_frame_id}.png")[...,:3]
        _, image, _ = np.split(image, indices_or_sections=3, axis=1)
        image = cv2.resize(image, (self.width,self.height))
        # Get static depth
        depth = np.load(f'{self.depth_path}/{city_id}/{full_frame_id}.npy')
        # Get dynamic object masks
        # N 192 512
        dynamic_object_mask = []
        dynamic_object_bbox = []
        for mask_id in mask_id_list:
            if mask_id == '-1':
                dynamic_object_mask.append(np.zeros((self.height,self.width)))
                dynamic_object_bbox.append(np.zeros((self.height,self.width)))
                continue
            mask = plt.imread(f'{self.object_mask_path}/{city_id}/{full_frame_id}/{mask_id}.png')
            dilate_iterations = 5 if np.any(self.middle_mask + mask > 1) else 2
            object_bbox = cv2.dilate(self.generate_bounding_box_mask(mask), np.ones((5,5)), iterations=dilate_iterations)
            dynamic_object_mask.append(mask)
            dynamic_object_bbox.append(object_bbox)
        dynamic_object_mask = np.stack(dynamic_object_mask)
        dynamic_object_bbox = np.stack(dynamic_object_bbox)
        # Get road masks
        if not os.path.isfile(f'{self.road_mask_path}/{city_id}/{full_frame_id}.png'):
            road_mask = np.zeros((self.height,self.width))
        else:
            road_mask = (plt.imread(f'{self.road_mask_path}/{city_id}/{full_frame_id}.png') >= 0.5).astype(float)
        # Get mask of dynamic region
        predicted_flow_neg = np.load(f'{self.object_predicted_flow_neg_path}/{city_id}/{full_frame_id}.npy')
        predicted_flow_pos = np.load(f'{self.object_predicted_flow_pos_path}/{city_id}/{full_frame_id}.npy')
        dynamic_region_mask = (np.sum(np.abs(predicted_flow_neg) + np.abs(predicted_flow_pos), axis=0)/2 >= self.flow_threshold).astype(float)
        # Transform to torch
        image = torch.from_numpy(image).permute(2,0,1).float() # 3 192 512
        depth = torch.from_numpy(depth).float() # N 192 512
        dynamic_object_mask = torch.from_numpy(dynamic_object_mask).float().unsqueeze(1) # N 192 512
        dynamic_object_bbox = torch.from_numpy(dynamic_object_bbox).float().unsqueeze(1) # N 192 512
        dynamic_region_mask = torch.from_numpy(dynamic_region_mask).unsqueeze(0).float() # 1 192 512
        road_mask = torch.from_numpy(road_mask).unsqueeze(0).float() # 1 192 512
        


        if self.is_train and np.random.random() < 0.5:
            image = torch.flip(image, [-1])
            depth = torch.flip(depth, [-1])
            object_mask = torch.flip(object_mask, [-1])
            dynamic_region_mask = torch.flip(dynamic_region_mask, [-1])
            road_mask = torch.flip(road_mask, [-1])
            object_bbox_mask = torch.flip(object_bbox_mask, [-1])

        if self.is_train and np.random.random() < 0.5:
            image = self.color_augmentation(image)

        return image, depth, dynamic_object_mask, dynamic_object_bbox, road_mask, dynamic_region_mask, full_frame_id
    
