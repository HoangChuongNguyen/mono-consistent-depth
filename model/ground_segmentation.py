
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
                       requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1).reshape(
            self.batch_size, 4, self.height, self.width)

        return cam_points

class GroundMaskExtractor(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self, batch_size, height, width):
        super(GroundMaskExtractor, self).__init__()
        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=11):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, threshold=10):
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        vertical = torch.cat((zeros, ones, zeros), dim=1).cuda()

        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill((torch.abs(y)<=1e-5), False)

        return cosine_sim, ground_mask

    def forward(self, depth, K):
        inv_K = torch.inverse(K)
        cam_points = self.backproject_depth(depth, inv_K)
        surface_normal = self.get_surface_normal(cam_points)
        cosine_sim, ground_mask = self.get_ground_mask(cam_points, surface_normal)
        return ground_mask.float()
        

class GroundSegmentationNet(nn.Module):
    def __init__(self, use_depth):
        super(GroundSegmentationNet, self).__init__()

        # Input layer (assuming 3-channel image input and 1-channel depth map)
        if use_depth:
            self.input_layer = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.input_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Additional convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((64, 192, 512))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((128, 192, 512))
        )

        # Final layer (assuming binary output for road/non-road)
        self.final_layer = nn.Conv2d(128, 1, kernel_size=1, stride=1)

    def forward(self, x):
        # Pass through the layers
        x = nn.ReLU()(self.input_layer(x))  # Output: [N, 64, 192, 512]
        x = nn.ReLU()(self.conv1(x))  # Output: [N, 64, 192, 512]
        x = nn.ReLU()(self.conv2(x))  # Output: [N, 128, 192, 512]

        # Final layer
        x = torch.sigmoid(self.final_layer(x))  # Output: [N, 1, 192, 512]
        return x
