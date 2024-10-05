


import torch
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.transforms import matrix_to_euler_angles
import numpy as np
import torch.nn.functional as F
from torch_sparse import coalesce

def matrix_from_angles(angle):
    return euler_angles_to_matrix(angle, convention="XYZ")

    

def angles_from_matrix(matrix):
    return matrix_to_euler_angles(matrix, convention="XYZ")

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def transformation_from_parameters(angle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    if invert:
        rot_matrix_inv = torch.inverse(matrix_from_angles(angle)) # b 3 3 
        angle = angles_from_matrix(rot_matrix_inv)
        translation = -torch.bmm(rot_matrix_inv, translation.view(len(translation), 3 ,1)).view(len(translation),3)
    T = torch.eye(4).unsqueeze(0).repeat(len(angle),1,1).to(angle.device) # B 4 4
    T[:,:3,:3] = matrix_from_angles(angle)
    T[:,:3,-1] = translation
    return T, angle, translation

    # R = matrix_from_angles(angle)
    # t = translation.clone()
    # if invert:
    #     R = R.transpose(1, 2)
    #     t *= -1

    # T = get_translation_matrix(t)

    # if invert:
    #     M = torch.bmm(R, T)
    # else:
    #     M = torch.matmul(T, R)

    # return M


def back_project_depth(depth, K_inv, pixel_coords):
    """
    depth: b h w 
    K_inv: b 3 3
    pixel_coords: 3 h w
    """
    xyz = torch.einsum('bij,jhw,bhw->bihw', K_inv, pixel_coords, depth) # 16, 3, 192, 512
    return xyz

# def project_3d(K, ref_T_src, xyz, object_pixel_wise_motion_2d=None):
#     """
#     xyz: 16 3 192 512
#     K: b 3 3
#     ref_T_src: b 4 4 
#     """
#     b, _, height, width = xyz.shape
#     xyz_src = torch.bmm(ref_T_src[:,:3,:3], xyz.view(xyz.shape[0],3,-1)) + ref_T_src[:,:3,[3]] # 16 3 192*512
#     uv = torch.bmm(K, xyz_src).view(len(xyz),3,height,width) # 16 3 192 512
#     if object_pixel_wise_motion_2d is not None: 
#         # print(cam_points.shape)
#         # print(cam_points.shape)
#         uv[:,0] = uv[:,0] + object_pixel_wise_motion_2d[:,0]
#         uv[:,1] = uv[:,1] + object_pixel_wise_motion_2d[:,1]

#     uv = uv[:,:2] / (uv[:,[-1]]+1e-5) # b 2 h w
#     uv = clamp_uv(uv[:,0], uv[:,1])
#     return uv

def project_3d(K, ref_T_src, xyz, object_pixel_wise_motion_2d=None, object_pixel_wise_motion_3d=None):
    """
    xyz: 16 3 192 512
    K: b 3 3
    ref_T_src: b 4 4 
    """
    if object_pixel_wise_motion_2d is not None and object_pixel_wise_motion_3d is not None: 
        assert False, "Either object_pixel_wise_motion_2d or object_pixel_wise_motion_3d must be None"
    b, _, height, width = xyz.shape
    xyz_src = torch.bmm(ref_T_src[:,:3,:3], xyz.view(xyz.shape[0],3,-1)) + ref_T_src[:,:3,[3]] # 16 3 192*512
    if object_pixel_wise_motion_3d is not None: 
        xyz_src = xyz_src + object_pixel_wise_motion_3d.view(xyz_src.shape)
    uv = torch.bmm(K, xyz_src).view(len(xyz),3,height,width) # 16 3 192 512
    if object_pixel_wise_motion_2d is not None: 
        uv[:,0] = uv[:,0] + object_pixel_wise_motion_2d[:,0]
        uv[:,1] = uv[:,1] + object_pixel_wise_motion_2d[:,1]
    uv = uv[:,:2] / (uv[:,[-1]]+1e-5) # b 2 h w
    uv = clamp_uv(uv[:,0], uv[:,1])
    return uv

def clamp_uv(pixel_x, pixel_y):
    _, height, width  = pixel_x.shape
    pixel_x = torch.clamp(pixel_x, 0.0, width - 1)
    pixel_y = torch.clamp(pixel_y, 0.0, height - 1)
    return torch.stack([pixel_x, pixel_y],dim=1)

def resample(src_frame, uv):
    """
    src_frame: b c h w
    uv: b 2 h w
    """
    _, _, height, width = src_frame.shape
    warp_x, warp_y = uv[:,0], uv[:,1]
    warp_x = warp_x / ((width - 1) / 2) - 1
    warp_y = warp_y / ((height - 1) / 2) - 1
    coord = torch.stack([warp_x, warp_y], dim=-1)
    warped_image = torch.nn.functional.grid_sample(src_frame, coord, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped_image


def get_grid(batch,height, width):
    grid = torch.squeeze(torch.stack(torch.meshgrid(torch.arange(0, end=height, dtype=torch.float),
                                   torch.arange( 0, end=width, dtype=torch.float),
                                   torch.tensor([1.0, ]))), dim=3)
    temp = torch.clone(grid[0,:,:])
    grid[0,:,:] = grid[1,:,:]
    grid[1,:,:] = temp
    grid = grid.unsqueeze(0).repeat(batch,1,1,1)
    return grid


def forward_warp(ref_depth, ref_img, ref_T_src, src_T_ref, intrinsics, upscale=None, deltaT=None, deltaT_inv=None):
    bs, _, hh, ww = ref_depth.size()
    # with torch.no_grad(): depth_u_mask = F.interpolate((ref_depth!=0).float(), scale_factor=upscale, mode='nearest').detach()
    depth_u = F.interpolate(ref_depth, scale_factor=upscale) # b 1 h w
    depth_u = depth_u # * depth_u_mask
    intrinsic_u = torch.cat((intrinsics[:, 0:2]*upscale, intrinsics[:, 2:]), dim=1)
    if deltaT is not None: 
        deltaT_inv_u = F.interpolate(deltaT_inv, scale_factor=upscale).squeeze(1)
    grid = get_grid(depth_u.shape[0], depth_u.shape[2], depth_u.shape[3]).to(ref_depth.device) # [B,3,uH,uW] 

    # trans_inv, rot_angle_inv = torch.split(pose_inv, dim=1, split_size_or_sections=3)
    cam_coord = grid.view(bs,3,-1)*depth_u.view(bs,1,-1)
    cam_coord = torch.bmm(torch.inverse(intrinsic_u[:,:3,:3]), cam_coord) # [B,3,uH*uW] 
    pcoords = torch.bmm(ref_T_src[:,:3], torch.cat([cam_coord, torch.ones_like(cam_coord[:,[0]])], dim=1))
    if deltaT is not None:
        pcoords = pcoords + deltaT_inv_u.view(pcoords.shape)
    pcoords = torch.bmm(intrinsics[:,:3,:3], pcoords) # [B,3,uH*uW] 

    # Note: Z is the depth in the reference frame (transform from source to reference)
    Z = pcoords[:,[2]].view(bs,1,hh*upscale,ww*upscale) # *depth_u_mask
    # print(torch.min(Z))
    # print(Z.shape)
    # plt.imshow(1/Z[0,0].detach().cpu())
    # plt.show()
    pcoords = (pcoords[:,:2].view(bs,2,hh*upscale,ww*upscale)/(Z+1e-5)).permute(0,2,3,1)
    depth_w, fw_val = [], []
    for coo, z in zip(pcoords, Z):
        idx = coo.reshape(-1,2).permute(1,0).long()[[1,0]]
        val = z.reshape(-1)
        idx[0][idx[0]<0] = hh
        idx[0][idx[0]>hh-1] = hh
        idx[1][idx[1]<0] = ww
        idx[1][idx[1]>ww-1] = ww
        _idx, _val = coalesce(idx, 1/val, m=hh+1, n=ww+1, op='max')     # Cast an index with maximum-inverse-depth: we do NOT interpolate points! >> errors near boundary
        depth_w.append( 1/torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1])).to_dense()[:-1,:-1] )
        fw_val.append( 1- (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1])).to_dense()[:-1,:-1]==0).float() )
        # pdb.set_trace()
    depth_w = torch.stack(depth_w, dim=0).unsqueeze(1) # b 1 192 512
    fw_val = torch.stack(fw_val, dim=0).unsqueeze(1) # b 1 192 512
    depth_w[fw_val==0] = 0 

    # cam_points = backproject_depth(depth_w, torch.linalg.inv(K))
    # pix_coords = project_3d(cam_points, K, T)
    # img_w = F.grid_sample(ref_img, pcoords, mode='bilinear', padding_mode='zeros', align_corners=True)

    xyz = back_project_depth(depth_w[:,0], torch.linalg.inv(intrinsics)[:,:3,:3], get_grid(1,depth_w.shape[2],depth_w.shape[3])[0].to(depth_w.device)) # 16, 3, 192, 512
    # pcoords = project_3d(intrinsics[:,:3,:3], src_T_ref, xyz) # b 2 h w 
    pcoords = project_3d(intrinsics[:,:3,:3], src_T_ref, xyz, object_pixel_wise_motion_3d=deltaT) # b 2 h w 
    img_w = resample(ref_img, pcoords)

    return img_w*(depth_w!=0), depth_w