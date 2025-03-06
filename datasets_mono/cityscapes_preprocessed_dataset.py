# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import numpy as np
import PIL.Image as pil

from matplotlib import pyplot as plt
import cv2
from .mono_dataset import MonoDataset


class CityscapesPreprocessedDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    RAW_WIDTH = 1024
    RAW_HEIGHT = 384

    def __init__(self, *args, **kwargs):
        super(CityscapesPreprocessedDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        file_split = self.filenames[index].split(' ')
        if self.object_mask_path is not None:
            city, frame_name, object_id_list = file_split[0], file_split[1], file_split[2]
            object_id_list = object_id_list.split(',')
            object_id_list = np.array(object_id_list).astype(int)
        else:
            city, frame_name = file_split[0], file_split[1]
            object_id_list = None 
        side = None
        return city, frame_name, side, object_id_list

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner

        camera_file = os.path.join(self.data_path, city, "{}_cam.txt".format(frame_name))
        camera = np.loadtxt(camera_file, delimiter=",")
        fx = camera[0]
        fy = camera[4]
        u0 = camera[2]
        v0 = camera[5]
        intrinsics = np.array([[fx, 0, u0, 0],
                               [0, fy, v0, 0],
                               [0,  0,  1, 0],
                               [0,  0,  0, 1]]).astype(np.float32)

        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics

    def get_colors(self, city, frame_name, side, object_id_list, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        color = self.loader(self.get_image_path(city, frame_name))
        color = np.array(color)

        w = color.shape[1] // 3
        inputs = {}
        inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
        inputs[("color", 0, -1)] = pil.fromarray(color[:, w:2*w])
        inputs[("color", 1, -1)] = pil.fromarray(color[:, 2*w:])

        # Load pseudo depth label 
        if self.pseudo_label_path is not None:
            if os.path.isfile(f'{self.pseudo_label_path}/{city}/{frame_name}.npy')  :
                pseudo_depth_label = np.load(f'{self.pseudo_label_path}/{city}/{frame_name}.npy')
            else:
                pseudo_depth_label = np.zeros((1,self.height,self.width))
            inputs[("pseudo_depth_label", 0, 0)] = np.copy(pseudo_depth_label)

        # Get dynamic object masks
        # N 192 512
        object_mask = None
        if object_id_list is not None:
            for mask_id in object_id_list:
                if mask_id == -1:
                    # dynamic_object_mask.append(np.zeros((self.height,self.width)))
                    continue
                mask = plt.imread(f'{self.object_mask_path}/{city}/{frame_name}/{mask_id}.png')
                object_mask = mask if object_mask is None else object_mask + mask
            if self.load_mask_triplet:
                inputs[("object_mask", -1, 0)] = np.expand_dims(object_mask[:, :self.width],0)
                inputs[("object_mask", 0, 0)] = np.expand_dims(object_mask[:, self.width:2*self.width],0)
                inputs[("object_mask", 1, 0)] = np.expand_dims(object_mask[:, 2*self.width:],0)
            else:
                if object_mask is not None:
                    object_mask = np.expand_dims(np.clip(object_mask, 0, 1),0)
                    inputs[("object_mask", 0, 0)] = np.copy(object_mask)
                else:
                    inputs[("object_mask", 0, 0)] = np.zeros((1,self.height,self.width))

        if do_flip:
            for key in inputs:
                if 'color' in key: inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)
                if 'pseudo_depth_label' in key: inputs[key] = np.copy(np.expand_dims(np.fliplr(pseudo_depth_label[0]), axis=0))
                if 'object_mask' in key[0] and type(inputs[key]) is np.ndarray:
                    inputs[key] = np.copy(np.fliplr(inputs[key]))
                # if 'object_mask' in key: inputs[key] = np.copy(np.expand_dims(np.fliplr(object_mask[0]), axis=0))
        return inputs


    def get_image_path(self, city, frame_name):
        return os.path.join(self.data_path, city, "{}.{}".format(frame_name, self.img_ext))
