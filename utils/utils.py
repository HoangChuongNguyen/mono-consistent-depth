# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
from matplotlib import pyplot as plt
import pickle as pkl
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def load_tf_state_dict(torch_model, tf_state_dict, matches_key_pair):
    """
    torch_state_dict: Pytorch state dict
    tf_state_dict: dictionary of params_name and numpy array of params_value (converted from Tensorflow checkpoint)
    matches_key_pair: list of tuples (torch_key, tf_key) of matching key between Pytorch and TensorFlow version
    """
    torch_state_dict = torch_model.state_dict()
    loaded_dict = {}
    for i in range(len(matches_key_pair)):
        torch_key = matches_key_pair[i][0]
        tf_key = matches_key_pair[i][1]
        assert torch_state_dict[torch_key].shape == tf_state_dict[tf_key].shape
        loaded_dict[torch_key] = tf_state_dict[tf_key]
    torch_state_dict.update(loaded_dict) 
    torch_model.load_state_dict(torch_state_dict)


def load_weights_from_tfResNet_to_depthEncoder(tf_imageNet_checkpoint_path, depth_encoder):
    """
    tfResnet_checkpoint_path: dictionary of key and weights(np array)
    """
    with open(tf_imageNet_checkpoint_path, "rb") as f:
        tf_state_dict = pkl.load(f)

    tf_resnet_state_dict = {}
    for key in tf_state_dict:
        tf_resnet_state_dict[key] = tf_state_dict[key]
    
    torch_depth_encoder_state_dict = depth_encoder.state_dict()
    tf_resnet_state_dict_converted = {}
    for key in tf_resnet_state_dict:
        tf_parameters = torch.tensor(tf_resnet_state_dict[key])
        if len(tf_parameters.shape) == 4:
            tf_parameters = tf_parameters.permute(3,2,0,1)
        tf_resnet_state_dict_converted[key] = tf_parameters

    depth_encoder_matches_key_pair = []
    for i in range(len(torch_depth_encoder_state_dict)):
        torch_key = list(torch_depth_encoder_state_dict.keys())[i]
        tf_key = list(tf_resnet_state_dict_converted.keys())[i]
        if (torch_depth_encoder_state_dict[torch_key].shape == tf_resnet_state_dict_converted[tf_key].shape):
            depth_encoder_matches_key_pair.append( (torch_key, tf_key) )
        else:
            assert ("Shape does not match")
            break
    load_tf_state_dict(depth_encoder, tf_resnet_state_dict_converted, depth_encoder_matches_key_pair)
    print("Load weights successfully")

def manydepth_update_adaptive_depth_bins(outputs, min_depth_tracker, max_depth_tracker):
    """Update the current estimates of min/max depth using exponental weighted average"""

    min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
    max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

    min_depth = min_depth.mean().cpu().item()
    max_depth = max_depth.mean().cpu().item()

    # increase range slightly
    min_depth = max(0.1, min_depth * 0.9)
    max_depth = max_depth * 1.1

    max_depth_tracker = max_depth_tracker * 0.99 + max_depth * 0.01
    min_depth_tracker = min_depth_tracker * 0.99 + min_depth * 0.01
    return min_depth_tracker, max_depth_tracker

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)