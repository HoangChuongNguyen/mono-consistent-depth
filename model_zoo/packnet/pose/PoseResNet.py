# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from ..layers.resnet.resnet_encoder import ResnetEncoder
from ..layers.resnet.pose_decoder import PoseDecoder

########################################################################################################################

class PoseResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "PoseResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

    def forward(self, inputs):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        rot_angle, translation = self.decoder([self.encoder(inputs)])
        return rot_angle[:,0].reshape(-1,3), translation[:,0].reshape(-1,3)

########################################################################################################################

