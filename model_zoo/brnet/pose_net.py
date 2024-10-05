
import torch
import torch.nn as nn


class BrPoseNet(nn.Module):
    def __init__(self, pose_encoder, pose_decoder):
        super(BrPoseNet, self).__init__()
        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder


    def forward(self, x):
        pose_inputs = [self.pose_encoder(x)]
        rot_angle, translation = self.pose_decoder(pose_inputs)
        return rot_angle, translation

