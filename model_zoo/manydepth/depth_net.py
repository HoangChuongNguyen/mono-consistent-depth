
import torch
import torch.nn as nn


class ManyDepthNet(nn.Module):
    def __init__(self, depth_encoder, depth_decoder):
        super(ManyDepthNet, self).__init__()
        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder

    def disp_to_depth(self, disp, min_depth=0.1, max_depth=100.0):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=None, max_depth_bin=None):
        features, lowest_cost, confidence_mask = self.depth_encoder(current_image,
                                                                    lookup_images,
                                                                    poses,
                                                                    K,
                                                                    invK,
                                                                    min_depth_bin=min_depth_bin,
                                                                    max_depth_bin=max_depth_bin)
        disp_dict = self.depth_decoder(features)
        disp_list = [disp_dict[key] for key in disp_dict]
        depth_list = [self.disp_to_depth(disp) for disp in disp_list][::-1]
        return depth_list, lowest_cost, confidence_mask

class MonoDepthNet(nn.Module):
    def __init__(self, depth_encoder, depth_decoder):
        super(MonoDepthNet, self).__init__()
        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder

    def disp_to_depth(self, disp, min_depth=0.1, max_depth=100.0):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def forward(self, images):
        features = self.depth_encoder(images)
        disp_dict = self.depth_decoder(features)
        disp_list = [disp_dict[key] for key in disp_dict]
        depth_list = [self.disp_to_depth(disp) for disp in disp_list][::-1]
        return depth_list