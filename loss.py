

import torch
import numpy as np

def depth_scale_invariant_loss(pred, target, lambda_= 0.5):
    # Scale the psudo label to has the same scale as the prediction
    with torch.no_grad():
        # First scaling ==> Find valid_depth_mask_list ==> Aim to exclude pixels that are very wrong
        median_scale_ = torch.stack([torch.median(pred[i]/target[i]) for i in range(len(pred))]).reshape(len(pred),1,1,1)
        target_ = target*median_scale_
        
        valid_depth_mask_list = torch.stack([((target_[i] >= torch.min(pred[i])) & (target_[i] <= np.percentile(pred[i].detach().cpu().numpy().flatten(), 90))).float() for i in range(len(pred))])
        # Second scale. Find the actual scale between the two depth map
        median_scale = torch.stack([torch.median((pred[i]/target[i])[valid_depth_mask_list[i]==1]) for i in range(len(pred))]).reshape(len(pred),1,1,1)
        target = target*median_scale

    loss = torch.abs(pred-target)
    return loss, valid_depth_mask_list

def multiply_no_nan(a, b):
    res = torch.mul(a, b)
    res[res != res] = 0
    return res 

def _gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def sqrt_motion_sparsity(motion_map):
    """motion_map: b 3 h w """
    motion_map = motion_map.permute(0,2,3,1) # from b 3 h w to b h w 3 to match the old code => will fix later
    tensor_abs = torch.abs(motion_map)
    mean = torch.mean(tensor_abs, dim=(1, 2), keepdim=True).detach()
    # We used L0.5 norm here because it's more sparsity encouraging than L1.
    # The coefficients are designed in a way that the norm asymptotes to L1 in
    # the small value limit.
    return torch.mean(2 * mean * torch.sqrt(tensor_abs / (mean + 1e-24) + 1))

def joint_bilateral_smoothing(disp, ref_frame):
    # From b c h w to b h w c to match with the old implemetntaiton => will fix later
    disp, ref_frame = [tensor.permute(0,2,3,1) for tensor in [disp, ref_frame]] 
    smoothed_dx = _gradient_x(disp)
    smoothed_dy = _gradient_y(disp)
    ref_dx = _gradient_x(ref_frame)
    ref_dy = _gradient_y(ref_frame)
    weights_x = torch.exp(-torch.mean(torch.abs(ref_dx), dim=3, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(ref_dy), dim=3, keepdim=True))
    smoothness_x = smoothed_dx * weights_x
    smoothness_y = smoothed_dy * weights_y
    return torch.mean(abs(smoothness_x)) + torch.mean(abs(smoothness_y))
def object_disp_smoothing(disp, object_mask):
    """Computes edge-aware smoothness loss.
    Args:
    smoothed: A torch.Tensor of shape [B, H, W, C1] to be smoothed.
    reference: A torch.Tensor of the shape [B, H, W, C2]. Wherever `reference` has
        more spatial variation, the strength of the smoothing of `smoothed` will
        be weaker.
    Returns:
    A scalar torch.Tensor containing the regularization, to be added to the
    training loss.
    """
    # From b c h w to b h w c to match with the old implemetntaiton => will fix later
    disp, object_mask = [tensor.permute(0,2,3,1) for tensor in [disp, object_mask]] 
    smoothed_dx = _gradient_x(disp)
    smoothed_dy = _gradient_y(disp)
    ref_dx = _gradient_x(object_mask)
    ref_dy = _gradient_y(object_mask)
    # weights_x = torch.exp(-torch.mean(torch.abs(ref_dx), dim=3, keepdim=True))
    # weights_y = torch.exp(-torch.mean(torch.abs(ref_dy), dim=3, keepdim=True))
    # print(torch.unique(weights_x))
    weight_x_mask = ((ref_dx==0) & (object_mask[:,:,:-1]==1)).float() # This is similar to doing mask erosion
    weight_y_mask = ((ref_dy==0) & (object_mask[:,:-1,:]==1)).float() # This is similar to doing mask erosion

    smoothness_x = torch.sum(torch.abs(smoothed_dx * weight_x_mask)) / torch.sum(weight_x_mask)
    smoothness_y = torch.sum(torch.abs(smoothed_dy * weight_y_mask)) / torch.sum(weight_y_mask)
    return smoothness_x + smoothness_y

def rgbd_consistency_loss(ref_frame, src_frame, warped_frame, ssim_weight, rgb_consistency_weight, c1, c2):
    # From b c h w to b h w c to match with the old implemetntaiton => will fix later
    ref_frame, src_frame, warped_frame = [tensor.permute(0,2,3,1) for tensor in [ref_frame, src_frame, warped_frame]] 
    frames_rgb_l1_diff = torch.abs(ref_frame - warped_frame)
    rgb_error = frames_rgb_l1_diff # TODO
    rgb_error = torch.mean(rgb_error, dim=-1)
    depth_proximity_weight = torch.ones_like(rgb_error).cuda()
    ssim_error, avg_weight = weighted_ssim(warped_frame,
                                           ref_frame,
                                           depth_proximity_weight,
                                           c1=c1,  # These values of c1 and c2 seemed to work better than
                                           c2=c2)
    ssim_error_mean = torch.mean(multiply_no_nan(ssim_error, avg_weight), -1) # TODO
    photometric_error = ssim_weight*ssim_error_mean + rgb_consistency_weight*rgb_error
    photometric_error = torch.split(photometric_error, split_size_or_sections=len(photometric_error)//2)
    # photometric_error.view(len(photometric_error)//2, 2, photometric_error.shape[1], photometric_error.shape[2])
    photometric_error = torch.minimum(photometric_error[0], photometric_error[1])
    # Calculate the mask to remove stationary pixel and car moving at constant speed
    identity_rgb_error = torch.abs(ref_frame - src_frame)
    identity_rgb_error = torch.mean(identity_rgb_error, dim=-1)
    identity_ssim_error, avg_weight = weighted_ssim(src_frame,
                                                    ref_frame,
                                                    depth_proximity_weight,
                                                    c1=c1,  # These values of c1 and c2 seemed to work better than
                                                    c2=c2)
    identity_ssim_error_mean = torch.mean(multiply_no_nan(identity_ssim_error, avg_weight), -1) # TODO
    identity_photometric_error = ssim_weight*identity_ssim_error_mean + rgb_consistency_weight*identity_rgb_error
    identity_photometric_error = torch.split(identity_photometric_error, split_size_or_sections=len(identity_photometric_error)//2)
    identity_photometric_error = torch.minimum(identity_photometric_error[0], identity_photometric_error[1])
    stationary_mask = (photometric_error < identity_photometric_error).float().detach()

    # photometric_error = torch.sum(stationary_mask*photometric_error) / torch.sum(stationary_mask+1e-10)
    return photometric_error, stationary_mask


def _avg_pool3x3(x):
    # Shape of input is x: B,H,W,C
    x = x.permute(0,3,1,2) # Change shape of x to B,C,H,W
    avg = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0)(x)
    avg = avg.permute(0,2,3,1) # Change shape of output to B,H,W,C
    return avg


def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                            'likely unintended.')
    weight = torch.unsqueeze(weight, -1)
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)
    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight
    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    result = torch.nn.ReflectionPad2d(1)(result.permute(0,3,1,2)).permute(0,2,3,1)
    average_pooled_weight = torch.nn.ReflectionPad2d(1)(average_pooled_weight.permute(0,3,1,2)).permute(0,2,3,1)
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


# def binary_cross_entropy_loss(pred, target):
#     bce_loss = torch.nn.functional.binary_cross_entropy(pred, target)
#     return bce_loss
