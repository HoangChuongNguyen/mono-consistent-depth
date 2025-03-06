
import os
import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import flow_to_image
from matplotlib import pyplot as plt

from datasets_mono import CityscapesPreprocessedDataset, CityscapesEvalDataset
from utils.utils import readlines, load_weights_from_tfResNet_to_depthEncoder, colormap, manydepth_update_adaptive_depth_bins, Namespace
from layers import transformation_from_parameters, back_project_depth, project_3d, resample, forward_warp
# from forward_warp import forward_warp
from model_zoo.model_zoo import *
from loss import joint_bilateral_smoothing, rgbd_consistency_loss, sqrt_motion_sparsity
from utils.validation_utils import compute_errors_all_dataset
from utils.checkpoints_utils import *
from raft_core.raft import RAFT

class PixelWiseTrainer:
    def __init__(self, cfg):
        # Depth network architecture
        self.cfg = cfg
        self.device = cfg['device']
        self.depth_net_type = cfg['depth_net_type']
        self.pose_net_type = cfg['pose_net_type']
        self.depth_encoder_pretrained = cfg['depth_encoder_pretrained']
        self.encoder_use_randomize_layernorm = cfg['encoder_use_randomize_layernorm']
        self.tf_imageNet_checkpoint_path = cfg['tf_imageNet_checkpoint_path']
        # self.many_depth_teacher_freeze_epoch = cfg['many_depth_teacher_freeze_epoch']
        
        # Loss weight
        self.ssim_c1 = float('inf') if cfg['ssim_c1'] == 'inf' else cfg['ssim_c1']
        self.ssim_c2 = cfg['ssim_c2']
        self.rgb_consistency_weight = cfg['rgb_consistency_weight']
        self.ssim_weight = cfg['ssim_weight']
        self.photometric_error_weight = cfg['photometric_error_weight']
        self.depth_smoothing_weight = cfg['depth_smoothing_weight']
        self.sparsity_loss_weight = cfg['sparsity_loss_weight']
        self.consistency_loss_weight = cfg['consistency_loss_weight']
        
        # Dataloading
        self.batch_size = cfg['batch_size']
        self.height = cfg['height']
        self.width = cfg['width']
        self.loaded_frame_idxs = cfg['loaded_frame_idxs']
        self.num_scales = cfg['num_scales']
        self.num_workers = cfg['num_workers']
        self.img_ext = cfg['img_ext']
        self.train_data_dir = cfg['train_data_dir']
        self.train_file_path = cfg['train_file_path']
        self.val_data_dir = cfg['val_data_dir']
        self.val_file_path = cfg['val_file_path']
        self.val_depth_data_dir = cfg['val_depth_data_dir']
        self.val_depth_file_path = cfg['val_depth_file_path']
        self.val_gt_depth_path = cfg['val_gt_depth_path']
        self.val_gt_mask_path = cfg['val_gt_mask_path']
        self.test_depth_data_dir = cfg['test_depth_data_dir']
        self.test_depth_file_path = cfg['test_depth_file_path']
        self.test_gt_depth_path = cfg['test_gt_depth_path']
        self.test_gt_mask_path = cfg['test_gt_mask_path']

        # Training setting
        self.initial_depth_pose_epochs = cfg['initial_depth_pose_epochs']
        self.initial_raft_epochs = cfg['initial_raft_epochs']
        self.reduce_lr_epochs_list = cfg['reduce_lr_epochs_list']
        self.nb_gpus = cfg['nb_gpus']
        self.epochs = cfg['epochs']
        self.learning_rate = cfg['learning_rate']
        self.optim_beta = cfg['optim_beta']
        self.current_epoch = 0
        self.global_step = 0
        self.smallest_error = 10e10
        
        # Evaluation
        self.first_epoch_eval_steps = cfg['first_epoch_eval_steps']
        self.eval_steps = cfg['eval_steps']
        self.eval_and_save_every_epoch = cfg['eval_and_save_every_epoch']
        
        # Logging
        self.log_path = f"{cfg['log_path']}/pixelwise_depthnet"
        self.models_to_load = cfg['models_to_load']
        self.eval_model_load_path = cfg["eval_model_load_path"]
        self.load_weights_folder = cfg['pixelwise_depthnet_load_weights_folder']

        # Define dataset 
        self.train_file = sorted(readlines(self.train_file_path))
        self.val_file = sorted(readlines(self.val_file_path))
        self.train_set = CityscapesPreprocessedDataset(data_path = self.train_data_dir,
                                                filenames = self.train_file,
                                                height = self.height,
                                                width = self.width,
                                                frame_idxs = self.loaded_frame_idxs,
                                                num_scales = self.num_scales,
                                                is_train=True,
                                                img_ext=self.img_ext)
        self.val_set = CityscapesPreprocessedDataset(data_path = self.val_data_dir,
                                                filenames = self.val_file,
                                                height = self.height,
                                                width = self.width,
                                                frame_idxs = self.loaded_frame_idxs,
                                                num_scales = self.num_scales,
                                                is_train=False,
                                                img_ext=self.img_ext)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, 
                                                batch_size=self.batch_size, 
                                                shuffle=True, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_set, 
                                                batch_size=self.batch_size, 
                                                shuffle=True, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=True)
                                                
        self.val_depth_file = readlines(self.val_depth_file_path)
        self.val_depth_set = CityscapesEvalDataset(
                                        data_path = self.val_depth_data_dir,
                                        filenames = self.val_depth_file,
                                        height = self.height,
                                        width = self.width,
                                        frame_idxs = [0],
                                        num_scales = 4,
                                        is_train=False,
                                        split = 'val',
                                        img_ext="png")
        self.val_depth_loader = torch.utils.data.DataLoader(dataset=self.val_depth_set, 
                                                batch_size=self.batch_size, 
                                                shuffle=False, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=False)

        self.test_depth_file = readlines(self.test_depth_file_path)
        self.test_depth_set = CityscapesEvalDataset(
                                        data_path = self.test_depth_data_dir,
                                        filenames = self.test_depth_file,
                                        height = self.height,
                                        width = self.width,
                                        frame_idxs = [0],
                                        num_scales = 4,
                                        is_train=False,
                                        split = 'test',
                                        img_ext="png")
        self.test_depth_loader = torch.utils.data.DataLoader(dataset=self.test_depth_set, 
                                                batch_size=self.batch_size, 
                                                shuffle=False, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=False)
        self.val_iter = iter(self.val_loader)


        # Define depthnet
        if self.depth_net_type == 'corl2020':
            self.depth_net = CorlDepthNet({}) # We pass an empty parameters here => Use default setting of the CorlDepthnet
            if self.depth_encoder_pretrained and self.tf_imageNet_checkpoint_path is None: assert False, 'Please specify path to imagenet checkpoint for the CORLDepthNet'
            if self.depth_encoder_pretrained: load_weights_from_tfResNet_to_depthEncoder(self.tf_imageNet_checkpoint_path, depth_net.depth_encoder)
        elif self.depth_net_type == 'monodepth2':
            raise NotImplementedError
        elif self.depth_net_type == 'packnet':
            self.depth_net = PackNet01(dropout=None, version='1A')
        elif self.depth_net_type == 'diffnet':
            self.depth_encoder = DiffNetDepthEncoder.hrnet18(True)
            self.depth_encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
            self.depth_decoder = DiffNetHRDepthDecoder(self.depth_encoder.num_ch_enc, scales=list(range(self.num_scales)))
            self.depth_net = DiffDepthNet(self.depth_encoder, self.depth_decoder)
        elif self.depth_net_type == 'brnet':
            self.depth_encoder = BrNetResnetEncoder(num_layers=18, pretrained=True)
            self.depth_decoder = BrNetDepthDecoder(self.depth_encoder.num_ch_enc, scales=list(range(self.num_scales)))
            self.depth_net = BrDepthNet(self.depth_encoder, self.depth_decoder)
        # elif self.depth_net_type == 'manydepth':
        #     raise NotImplementedError
        #     # Define extra parameters used for the ManyDepth models
        #     min_depth_tracker = 0.1
        #     max_depth_tracker = 10.0
        #     matching_ids = [0,-1] 
        #     depth_encoder = ManyDepthEncoder(num_layers=18, pretrained=True,
        #                                     input_height=height, input_width=width,
        #                                     adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
        #                                     depth_binning='linear', num_depth_bins=96)
        #     depth_decoder = ManyDepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=list(range(num_scales)))
        #     depth_encoder.to(device)
        #     depth_decoder.to(device)
        #     depth_net = ManyDepthNet(depth_encoder, depth_decoder)
        #     # Define monodepth2 models
        #     mono_depth_encoder = ManyDepthResnetEncoder(num_layers=18, pretrained=True)
        #     mono_depth_decoder = ManyDepthDecoder(mono_depth_encoder.num_ch_enc, scales=list(range(num_scales)))
        #     mono_depth_encoder.to(device)
        #     mono_depth_decoder.to(device)
        #     mono_depth_net = MonoDepthNet(mono_depth_encoder, mono_depth_decoder)
        else:
            assert False, f'Unknow depth_net_type {self.depth_net_type}. It should be in ["corl2020", "diffnet", "brnet", "packnet"]'


        # Define posenet
        if self.pose_net_type == 'corl2020':
            pose_net = CorlPoseNet(in_channels=6) 
        elif self.pose_net_type == 'packnet':
            self.pose_net = PackNetPoseResNet(version='18pt')
        elif self.pose_net_type in ['manydepth', "monodepth2", 'diffnet', 'brnet']:
            self.pose_encoder = ManyDepthResnetEncoder(num_layers=18, pretrained=True, num_input_images=2)
            self.pose_decoder = ManyPoseDecoder(num_ch_enc=self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
            self.pose_net = ManyPoseNet(self.pose_encoder, self.pose_decoder)
        # Define raft (pixel-wise object motion network)
        self.raft =  RAFT(Namespace(small=False, dropout=False, mixed_precision=False)) # ==> will fix this Namespace class later
        # Bring models to cuda
        self.depth_net.to(self.device)
        self.depth_net = torch.nn.DataParallel(self.depth_net, device_ids=list(range(self.nb_gpus)))
        # if self.depth_net_type == 'manydepth':
        #     mono_depth_net.to(device)
        #     mono_depth_net = torch.nn.DataParallel(mono_depth_net, device_ids=list(range(nb_gpus)))
        self.pose_net.to(self.device)
        self.pose_net = torch.nn.DataParallel(self.pose_net, device_ids=list(range(self.nb_gpus)))
        self.raft.to(device=self.device)
        self.raft = torch.nn.DataParallel(self.raft, device_ids=list(range(self.nb_gpus)))
        # Init dummy data here
        self.parameters_to_train = None 
        self.optimizer = None
        # Define default grid
        self.grid = torch.squeeze(torch.stack(torch.meshgrid(torch.arange(0, end=self.height, dtype=torch.float),
                                        torch.arange( 0, end=self.width, dtype=torch.float),
                                        torch.tensor([1.0, ]))), dim=3)
        self.grid[[0,1]] = self.grid[[1,0]]
        self.grid = self.grid.type(torch.FloatTensor).to(device=self.device)
        # Log set up 
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=10)
        # Load model
        if self.load_weights_folder is not None:
            try: 
                current_epoch = int(self.load_weights_folder.split('_')[-1])
            except:
                current_epoch = 10000 # Set dummy for current_epoch when load the best model
            self.init_optimizer_based_on_epoch(current_epoch)
            load_model([self.depth_net, self.pose_net, self.raft], self.models_to_load, self.optimizer, self.load_weights_folder)
            # if self.depth_net_type == 'manydepth':
            #     min_depth_tracker, max_depth_tracker = np.loadtxt(f'{self.load_weights_folder}/depth_range.txt')
            #     print(f'Loaded depth range: [{min_depth_tracker}, {max_depth_tracker}]')
            self.current_epoch = current_epoch + 1


    def init_optimizer_based_on_epoch(self, current_epoch):
        print(f'Initialized optimizer at {current_epoch}')
        parameters_to_train = self.parameters_to_train
        # The stage when we train everything
        if current_epoch >= self.initial_raft_epochs[-1]:
            print('Models to train: depth_net, pose_net, raft')
            parameters_to_train = list(self.depth_net.parameters()) + list(self.pose_net.parameters()) + list(self.raft.parameters())
            # if self.depth_net_type == 'manydepth':
            #     parameters_to_train += list(self.mono_depth_net.parameters()) 
        # The stage when we initialize raft
        elif current_epoch >= self.initial_depth_pose_epochs[-1]:
            print('Models to train: raft')
            parameters_to_train = list(self.raft.parameters())
        # The stage when we initialize depth_net and pose_net
        elif current_epoch >= 0:
            print('Models to train: depth_net, pose_net')
            parameters_to_train = list(self.depth_net.parameters()) + list(self.pose_net.parameters())
            # if depth_net_type == 'manydepth':
            #     parameters_to_train += list(mono_depth_net.parameters()) 
        self.optimizer = torch.optim.Adam(parameters_to_train, lr=self.learning_rate)

    # def manydepth_freeze_teacher_and_pose_net():
    #     pose_net.eval()
    #     raft.eval()
    #     mono_depth_net.eval()
    #     global parameters_to_train
    #     global optimizer
    #     print("Start freeze teacher monodepth and pose net and raft")
    #     print('Models to train: depth_net')
    #     parameters_to_train = list(depth_net.parameters())
    #     optimizer = torch.optim.Adam(parameters_to_train, lr=learning_rate)


    def log(self, train_inputs, train_outputs, val_inputs, val_outputs, global_step):
        with torch.no_grad():
            losses = {"train": train_outputs['losses'], "val": val_outputs['losses']}
            endpoints = {"train": train_inputs, "val": val_inputs}
            output_endpoints = {"train": train_outputs, "val": val_outputs}

            # if self.depth_net_type == 'manydepth':
            #     self.writers['train'].add_scalar("min_depth_tracker", self.min_depth_tracker, global_step) 
            #     self.writers['train'].add_scalar("max_depth_tracker", self.max_depth_tracker, global_step) 

            for mode in ["train", "val"]:
                writer = self.writers[mode]
                loss = losses[mode]
                endpoint = endpoints[mode]
                output_endpoint = output_endpoints[mode]
                for l, v in loss.items():
                    writer.add_scalar("loss/{}".format(l), v, global_step) 
                
                for j in range(2):
                    
                    ref_frame = endpoint[('color_aug',0,0)][j].detach().cpu() # 3 h w
                    neg_src_frame = endpoint[('color_aug',-1,0)][j].detach().cpu() # 3 h w
                    pos_src_frame = endpoint[('color_aug',1,0)][j].detach().cpu() # 3 h w
                    fw_neg_warped_frame = output_endpoint[('fw_warped_image',-1,0)][j].detach().cpu() # 3 h w
                    fw_pos_warped_frame = output_endpoint[('fw_warped_image',1,0)][j].detach().cpu() # 3 h w
                    neg_warped_frame = output_endpoint[('warped_image',-1,0)][j].detach().cpu() # 3 h w
                    pos_warped_frame = output_endpoint[('warped_image',1,0)][j].detach().cpu() # 3 h w

                    neg_object_motion_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('object_pixel_wise_motion',-1, s)][[j]].detach().cpu(), [self.height, self.width], mode="bilinear", align_corners=False) for s in range(self.num_scales)])
                    pos_object_motion_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('object_pixel_wise_motion',1, s)][[j]].detach().cpu(), [self.height, self.width], mode="bilinear", align_corners=False) for s in range(self.num_scales)])

                    neg_object_motion = neg_object_motion_all[0] # 2 h w
                    pos_object_motion = pos_object_motion_all[0]# 2 h w

                    # ref_disp_0 = output_endpoint[('disp',0)][j].detach().cpu() # 1 h w 
                    # ref_disp_0_colmap = torch.from_numpy(colormap(ref_disp_0.unsqueeze(0))[0]) # 3 h w
                    ref_disp_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('disp',0,s)][[j]].detach().cpu(), [self.height, self.width], mode="bilinear", align_corners=False) for s in range(self.num_scales)])
                    ref_disp_all_colmap = torch.from_numpy(colormap(ref_disp_all)) # 4 3 h w 
                    ref_disp_0_colmap = ref_disp_all_colmap[0]

                    try: neg_optical_flow = flow_to_image(neg_object_motion).float()/255.0
                    except: neg_optical_flow = torch.zeros(3,neg_object_motion.shape[1],neg_object_motion.shape[2])
                    neg_stack_img = torch.concat([ref_disp_0_colmap,
                                                ref_frame,
                                                neg_warped_frame,
                                                fw_neg_warped_frame,
                                                neg_src_frame,
                                                neg_optical_flow
                                                ], dim=1)

                    try: pos_optical_flow = flow_to_image(pos_object_motion).float()/255.0
                    except: pos_optical_flow = torch.zeros(3,pos_object_motion.shape[1],pos_object_motion.shape[2])
                    pos_stack_img = torch.concat([ref_disp_0_colmap,
                                                ref_frame,
                                                pos_warped_frame,
                                                fw_pos_warped_frame,
                                                pos_src_frame,
                                                pos_optical_flow
                                                ], dim=1)

                    disp_all_stack = torch.concat([disp for disp in ref_disp_all_colmap], dim=1)
                    neg_motion_stack = torch.concat([flow_to_image(object_motion).float()/255.0 for object_motion in neg_object_motion_all], dim=1)
                    pos_motion_stack = torch.concat([flow_to_image(object_motion).float()/255.0 for object_motion in pos_object_motion_all], dim=1)

                    writer.add_image(f"a_visualzaition_-1/{j}/{mode}", neg_stack_img, global_step)
                    writer.add_image(f"b_visualzaition_1/{j}/{mode}", pos_stack_img, global_step)
                    writer.add_image(f"c_disp_multiscale/{j}/{mode}", disp_all_stack, global_step)
                    writer.add_image(f"d_neg_motion_multiscale/{j}/{mode}", neg_motion_stack, global_step)
                    writer.add_image(f"e_pos_motion_multiscale/{j}/{mode}", pos_motion_stack, global_step)

                    writer.add_image(f"f_stationary_mask/{j}/{mode}", output_endpoint['stationary_mask'][[j]].detach().cpu(), global_step)

                    # if self.depth_net_type=='manydepth':
                    #     lowest_cost = output_endpoint["lowest_cost"][j]
                    #     consistency_mask = output_endpoint['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()
                    #     min_val = np.percentile(lowest_cost.numpy(), 10)
                    #     max_val = np.percentile(lowest_cost.numpy(), 90)
                    #     lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                    #     lowest_cost = colormap(lowest_cost)
                    #     writer.add_image("lowest_cost/{}".format(j),lowest_cost, global_step)
                    #     writer.add_image("lowest_cost_masked/{}".format(j),lowest_cost * consistency_mask, global_step)
                    #     writer.add_image("consistency_mask/{}".format(j),consistency_mask, global_step)

                    #     consistency_target = colormap(output_endpoint["consistency_target/0"][[j]])[0]
                    #     writer.add_image("consistency_target/{}".format(j), consistency_target, global_step)
                
                    #     mono_ref_disp_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('mono_disp',0,s)][[j]].detach().cpu(), [height, width], mode="bilinear", align_corners=False) for s in range(num_scales)])
                    #     mono_ref_disp_all_colmap = torch.from_numpy(colormap(mono_ref_disp_all)) # 4 3 h w 
                    #     mono_ref_disp_0_colmap = mono_ref_disp_all_colmap[0]
                    #     mono_disp_all_stack = torch.concat([disp for disp in mono_ref_disp_all_colmap], dim=1)
                    #     writer.add_image(f"g_mono_disp_multiscale/{j}/{mode}", mono_disp_all_stack, global_step)


                    #     mono_neg_warped_frame = output_endpoint[('mono_warped_image',-1,0)][j].detach().cpu() # 3 h w
                    #     mono_pos_warped_frame = output_endpoint[('mono_warped_image',1,0)][j].detach().cpu() # 3 h w

                    #     mono_neg_stack_img = torch.concat([mono_ref_disp_0_colmap,
                    #                                 ref_frame,
                    #                                 mono_neg_warped_frame,
                    #                                 fw_neg_warped_frame,
                    #                                 neg_src_frame,
                    #                                 flow_to_image(neg_object_motion).float()/255.0
                    #                                 ], dim=1)


                    #     mono_pos_stack_img = torch.concat([mono_ref_disp_0_colmap,
                    #                                 ref_frame,
                    #                                 mono_pos_warped_frame,
                    #                                 fw_pos_warped_frame,
                    #                                 pos_src_frame,
                    #                                 flow_to_image(pos_object_motion).float()/255.0
                    #                                 ], dim=1)
                    #     writer.add_image(f"a_mono_visualzaition_-1/{j}/{mode}", mono_neg_stack_img, global_step)
                    #     writer.add_image(f"b_mono_visualzaition_1/{j}/{mode}", mono_pos_stack_img, global_step)
                        

        # def manydepth_compute_matching_mask(outputs):
        #     """Generate a mask of where we cannot trust the cost volume, based on the difference
        #     between the cost volume and the teacher, monocular network"""

        #     mono_output = outputs[('mono_depth', 0, 0)]
        #     matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(device)

        #     # mask where they differ by a large amount
        #     mask = ((matching_depth - mono_output) / mono_output) < 1.0
        #     mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        #     return mask[:, 0]

    def predict_depth(self, inputs, outputs):
        predicted_depth_dict = {}
        if self.depth_net_type == 'corl2020':
            predicted_depth = self.depth_net(inputs[('color_aug',0,0)], self.global_step) # This is a list storing predicted at multi-scale. Each has shape of b 1 h w
        elif self.depth_net_type in ['packnet', 'monodepth2', 'diffnet', 'brnet']:
            predicted_depth = self.depth_net(inputs[('color_aug',0,0)])
        # elif depth_net_type == 'manydepth':
        #     relative_poses = [outputs[('cam_T_cam', idx, 0)].detach() for idx in matching_ids[1:]]
        #     relative_poses = torch.stack(relative_poses, dim=1)
        #     lookup_frames = [inputs[('color_aug', idx, 0)] for idx in matching_ids[1:]]
        #     lookup_frames = torch.stack(lookup_frames, dim=1) 
        #     # apply static frame and zero cost volume augmentation
        #     batch_size = len(lookup_frames)
        #     augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(device).float()
        #     if depth_net.training:
        #         for batch_idx in range(batch_size):
        #             rand_num = random.random()
        #             # static camera augmentation -> overwrite lookup frames with current frame
        #             if rand_num < 0.25:
        #                 replace_frames = [inputs[('color', 0, 0)][batch_idx] for _ in matching_ids[1:]]
        #                 replace_frames = torch.stack(replace_frames, 0)
        #                 lookup_frames[batch_idx] = replace_frames
        #                 augmentation_mask[batch_idx] += 1
        #             # missing cost volume augmentation -> set all poses to 0, the cost volume will
        #             # skip these frames
        #             elif rand_num < 0.5:
        #                 relative_poses[batch_idx] *= 0
        #                 augmentation_mask[batch_idx] += 1
        #     predicted_depth_dict['augmentation_mask'] = augmentation_mask
        #     predicted_depth, lowest_cost, confidence_mask = depth_net(inputs["color_aug", 0, 0],
        #                                                                 lookup_frames,
        #                                                                 relative_poses,
        #                                                                 inputs[('K', 2)],
        #                                                                 inputs[('inv_K', 2)],
        #                                                                 min_depth_bin=min_depth_tracker,
        #                                                                 max_depth_bin=max_depth_tracker) # This is a list storing predicted at multi-scale. Each has shape of b 1 h w
            
        #     try:
        #         predicted_depth_dict["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
        #                                             [height, width],
        #                                             mode="nearest")[:, 0]
        #         predicted_depth_dict["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
        #                                                     [height, width],
        #                                                     mode="nearest")[:, 0]
        #         outputs["lowest_cost"] = predicted_depth_dict["lowest_cost"] 
        #         predicted_depth_dict["consistency_mask"] = (predicted_depth_dict["consistency_mask"] * manydepth_compute_matching_mask(outputs))
        #     except: pass
        else:
            raise NotImplementedError
        # Format the predicted depth as a dictionary
        for s in range(self.num_scales):
            predicted_depth_dict[('depth',0,s)] = predicted_depth[s]
            predicted_depth_dict[('disp',0,s)] = 1/predicted_depth_dict[('depth',0,s)]
        # zero_cost_volume_mask = []
        # for i in range(len(predicted_depth_dict[('depth',0,0)])):
        #     zero_cost_volume_mask.append(torch.zeros(1,height,width)+zero_cost_volume_list[i])
        # predicted_depth_dict['zero_cost_augmentation'] = torch.stack(zero_cost_volume_mask)
        return predicted_depth_dict

    def predict_pose(self, inputs, predict_pos_pose=True):
        # !!! We only predict pose at the highest resolution
        # Following monodepth2, the motions features are stacked in temporal orders
        motion_features_neg = torch.concat([inputs['color_aug',-1,0], inputs['color_aug',0,0]],dim=1) # b 6 h w
        # Predict motion 
        rot_neg, trans_neg = self.pose_net(motion_features_neg) # b 3
        # Get the transformation matrix
        ref_to_src_neg, rot_neg, trans_neg = transformation_from_parameters(rot_neg, trans_neg, invert=True)
        if predict_pos_pose:
            motion_features_pos = torch.concat([inputs['color_aug',0,0], inputs['color_aug',1,0]],dim=1) # b 6 h w
            rot_pos, trans_pos = self.pose_net(motion_features_pos) # b 3
            ref_to_src_pos, rot_pos, trans_pos = transformation_from_parameters(rot_pos, trans_pos, invert=False)
        else:
            rot_pos, trans_pos,ref_to_src_pos = None, None, None
        cam_T_cam_dict = {  ('rot',-1,0): rot_neg,
                            ('rot',1,0): rot_pos,
                            ('trans',-1,0): trans_neg,
                            ('trans',1,0): trans_pos,
                            ('cam_T_cam',-1,0):ref_to_src_neg,
                            ('cam_T_cam',1,0):ref_to_src_pos}
        return cam_T_cam_dict


    def generate_images_pred(self, inputs, outputs, is_multi=False):
        # Get some data used for warping image
        K = inputs[('K', 0)]
        K_inv = inputs[('inv_K', 0)]
        src_neg = inputs[('color_aug',-1,0)]
        src_pos = inputs[('color_aug',1,0)]
        cam_T_cam_neg = outputs[('cam_T_cam',-1,0)]
        cam_T_cam_pos = outputs[('cam_T_cam',1,0)]
        image_pred_dict = {}
        for s in range(self.num_scales):
            # Resize depth from h/(2^s) w/(2^s) to h w
            # predicted_depth_s = torch.nn.functional.interpolate(outputs[('depth',0,s)], [height, width], mode="bilinear", align_corners=False) # b 1 h w 
            # if self.depth_net_type == 'manydepth' and not is_multi:
            #     raise NotImplementedError
            #     # predicted_depth_s = outputs[('mono_depth',0,s)] # b 1 h//2(**s) w//(2**s)
            # else:
            predicted_depth_s = outputs[('depth',0,s)] # b 1 h//2(**s) w//(2**s)
            object_pixel_wise_motion_neg_s = outputs[('object_pixel_wise_motion',-1,s)] # b 2 h//2(**s) w//(2**s)
            object_pixel_wise_motion_pos_s = outputs[('object_pixel_wise_motion',1,s)] # b 2 h//2(**s) w//(2**s)
            # We block the gradients backprogataed to RAFT (when using muit-frame depth)
            # if depth_net_type == 'manydepth' and is_multi:
            #     object_pixel_wise_motion_neg_s = object_pixel_wise_motion_neg_s.detach() # b 1 h//2(**s) w//(2**s)
            #     object_pixel_wise_motion_pos_s = object_pixel_wise_motion_pos_s.detach() # b 2 h//2(**s) w//(2**s)
            # Resize depth and object motion from h/(2^s) w/(2^s) to h w
            predicted_depth_s = torch.nn.functional.interpolate(predicted_depth_s,  [self.height, self.width], mode="bilinear", align_corners=False) # b 1 h w
            # Backproject a pixel in reference frame to 3D space 
            xyz = back_project_depth(predicted_depth_s[:,0], K_inv[:,:3,:3], self.grid) # 16, 3, 192, 512
            # Project a source pixel back to the source frame
            # We block the gradient for intermediate predictions. Otherwise the pixel-wise motion will be noisy
            # if s != 0:
            #     object_pixel_wise_motion_neg_s = object_pixel_wise_motion_neg_s.detach()
            #     object_pixel_wise_motion_pos_s = object_pixel_wise_motion_pos_s.detach()
            uv_neg = project_3d(K[:,:3,:3], cam_T_cam_neg, xyz, object_pixel_wise_motion_neg_s)
            uv_pos = project_3d(K[:,:3,:3], cam_T_cam_pos, xyz, object_pixel_wise_motion_pos_s)
            # Get the warped image
            warped_image_neg = resample(src_neg, uv_neg)
            warped_image_pos = resample(src_pos, uv_pos)
            # Store data in a dictionary
            image_pred_dict_s = {
                ('uv',-1,s): uv_neg,
                ('uv', 1,s): uv_pos,
                ('warped_image',-1,s): warped_image_neg,
                ('warped_image',1,s): warped_image_pos
            }
            image_pred_dict.update(image_pred_dict_s)
        
        return image_pred_dict
        

    def compute_losses(self, inputs, outputs, epoch, is_multi=False):
        # during raft initialization, we don't use sparsity loss (i.e set epoch_sparsity_loss_weight to 0) 
        epoch_sparsity_loss_weight = 0.0 if epoch < self.initial_raft_epochs[-1] else self.sparsity_loss_weight
        losses_dict = {}
        for s in range(self.num_scales):
            # Compute depth smoothness loss
            # if self.depth_net_type == 'manydepth' and not is_multi:
            #     raise NotImplementedError
            #     disp = outputs[('mono_disp',0,s)] 
            # else:
            disp = outputs[('disp',0,s)] 
            mean_disp = torch.mean(disp, dim=[1, 2, 3], keepdim=True)
            if self.depth_net_type == 'brnet' and s>0:
                color_smooth_s = inputs[('color_aug',0,s-1)]
            else:
                color_smooth_s = inputs[('color_aug',0,s)]
            depth_smoothness_loss = self.depth_smoothing_weight*joint_bilateral_smoothing(disp/mean_disp, color_smooth_s)
            depth_smoothness_loss = depth_smoothness_loss / (2 ** s) 
            # Compute photometric loss
            # if self.depth_net_type == 'manydepth' and not is_multi:
            #     raise NotImplementedError
            #     photometric_loss, stationary_mask_s = rgbd_consistency_loss(inputs[('color_aug',0,0)].repeat(2,1,1,1), 
            #                                                             torch.concat([inputs[('color_aug',-1,0)], inputs[('color_aug',1,0)]],dim=0), 
            #                                                             torch.concat([outputs[('mono_warped_image',-1,s)], outputs[('mono_warped_image',1,s)]],dim=0), 
            #                                                             ssim_weight, rgb_consistency_weight, c1=ssim_c1, c2=ssim_c2)
            # else:
            photometric_loss, stationary_mask_s = rgbd_consistency_loss(inputs[('color_aug',0,0)].repeat(2,1,1,1), 
                                                                    torch.concat([inputs[('color_aug',-1,0)], inputs[('color_aug',1,0)]],dim=0), 
                                                                    torch.concat([outputs[('warped_image',-1,s)], outputs[('warped_image',1,s)]],dim=0), 
                                                                    self.ssim_weight, self.rgb_consistency_weight, c1=self.ssim_c1, c2=self.ssim_c2)
            photometric_loss_mask_s = stationary_mask_s
            # photometric_loss = photometric_error_weight * photometric_loss 
            if s == 0: stationary_mask = stationary_mask_s

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                photometric_loss_mask_s = torch.ones_like(stationary_mask_s)
                photometric_loss_mask_s = (photometric_loss_mask_s * outputs['consistency_mask'].unsqueeze(1))
                photometric_loss_mask_s = (photometric_loss_mask_s * (1 - outputs['augmentation_mask']))
                consistency_mask = (1 - photometric_loss_mask_s).float()

            photometric_loss = torch.sum(photometric_loss_mask_s*photometric_loss) / torch.sum(photometric_loss_mask_s+1e-10)
            losses_dict_s = {f'depth_smooth_{s}': depth_smoothness_loss, f'photometric_loss_{s}': photometric_loss}
            losses_dict.update(losses_dict_s)
            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                raise NotImplementedError
                # multi_depth = outputs[("depth", 0, s)]
                # # no gradients for mono prediction!
                # mono_depth = outputs[("mono_depth", 0, s)].detach()
                # multi_depth = F.interpolate(multi_depth, [height, width], mode="bilinear", align_corners=False)
                # mono_depth = F.interpolate(mono_depth, [height, width], mode="bilinear", align_corners=False)
                # consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                # consistency_loss = torch.sum(consistency_loss) / torch.sum(consistency_mask)
                # # save for logging to tensorboard
                # consistency_target = (mono_depth.detach() * consistency_mask +
                #                         multi_depth.detach() * (1 - consistency_mask))
                # consistency_target = 1 / consistency_target
                # outputs["consistency_target/{}".format(s)] = consistency_target
                # losses_dict[f'consistency_loss_{s}'] = consistency_loss*consistency_loss_weight
            else:
                consistency_loss = 0


        # Compute motion sparsity loss (at the highest resolution only)
        object_pixel_wise_motion_stack = torch.concat([outputs[('object_pixel_wise_motion',f,0)] for f in self.loaded_frame_idxs[1:]])
        motion_sparsity_loss = epoch_sparsity_loss_weight*sqrt_motion_sparsity(object_pixel_wise_motion_stack)
        losses_dict['final_motion_sparsity_loss'] = motion_sparsity_loss
        losses_dict['final_photometric_loss'] = sum([losses_dict[f'photometric_loss_{s}'] for s in range(self.num_scales)]) / self.num_scales
        losses_dict['final_depth_smooth'] = sum([losses_dict[f'depth_smooth_{s}'] for s in range(self.num_scales)]) / self.num_scales
        
        # if is_multi and self.depth_net_type=='manydepth':
        #     losses_dict['final_consistency_loss'] = sum([losses_dict[f'consistency_loss_{s}'] for s in range(self.num_scales)]) / self.num_scales

        return losses_dict, stationary_mask

    def get_fw_warped_image(self, inputs, outputs):
        # We only get the forward warped image at the highest resolution
        # Note that in case of manydepth, we use depth of monodepth2 to get the fw_warped image
        # This help to avoid 0 object motion caused by manydepth predicting exactly the triangulated depth
        # if self.depth_net_type == 'manydepth':
        #     ref_depth = outputs[('mono_depth',0,0)] # b 1 h w
        # else:
        ref_depth = outputs[('depth',0,0)] # b 1 h w
        fw_warped_dict = {}
        for frame_idx in self.loaded_frame_idxs[1:]: 
            fw_warped_image, fw_warped_depth = forward_warp(ref_depth, inputs[('color_aug',0,0)], 
                                                            outputs[('cam_T_cam',frame_idx,0)], 
                                                            torch.inverse(outputs[('cam_T_cam', frame_idx,0)]), 
                                                            inputs[('K',0)], upscale=3)
            fw_warped_dict[('fw_warped_image', frame_idx, 0)] = fw_warped_image
            fw_warped_dict[('fw_warped_depth', frame_idx, 0)] = fw_warped_depth
        return fw_warped_dict

    def predict_object_pixel_wise_motion(self, samples, outputs, epoch, predict_object_motion=True):
        
        # We only predict pixel-wise object motion at the highest resolution
        object_pixel_wise_motion_dict = {}
        # Before the initial_depth_pose_epochs[-1], we do not use the object_pixel_wise_motion, thus initialized it as all 0
        if epoch < self.initial_depth_pose_epochs[-1] or not predict_object_motion:
            for frame_idx in self.loaded_frame_idxs[1:]: 
                object_pixel_wise_motion_dict[('object_pixel_wise_motion', frame_idx, 0)] = torch.zeros_like(samples[('color_aug',frame_idx,0)][:,:2])
        # Only predict pixel-wise motion at this stage
        else:
            object_motion_burn_in_weight = 1
            for frame_idx in self.loaded_frame_idxs[1:]: 
                object_pixel_wise_motion = self.raft(outputs[('fw_warped_image',frame_idx,0)]*255.0, samples[('color_aug',frame_idx,0)]*255.0, iters=1)[-1] # b 2 h w 
                # Poss process the predicted motion
                with torch.no_grad():
                    object_pixel_wise_motion_norm = torch.norm(object_pixel_wise_motion, dim=1, keepdim=True)
                    # Exclude around 50% of the predicted motion => We assume at least 50% is static
                    object_pixel_wise_motion_mask_mean = object_pixel_wise_motion_norm >= torch.mean(object_pixel_wise_motion_norm, dim=[1,2,3], keepdim=True)
                    # Mask out motion with very small magnitude
                    valid_pixel_wise_motion_mask = outputs[('fw_warped_depth',frame_idx,0)] != 0
                    object_pixel_wise_motion_mask = object_pixel_wise_motion_norm >= 1e-5
                # Not apply mean mask out when we initializing raft
                if epoch < self.initial_raft_epochs[-1]:
                    object_pixel_wise_motion = object_pixel_wise_motion * valid_pixel_wise_motion_mask.detach()
                else:
                    object_pixel_wise_motion = object_pixel_wise_motion * object_pixel_wise_motion_mask.detach() * object_pixel_wise_motion_mask_mean.detach() * valid_pixel_wise_motion_mask.detach()
                # Align the predicted pixel-wise motion to the reference frame
                object_pixel_wise_motion, _ = forward_warp(outputs[('fw_warped_depth',frame_idx,0)], 
                                                        object_pixel_wise_motion,
                                                        torch.inverse(outputs[('cam_T_cam', frame_idx,0)]),
                                                        outputs[('cam_T_cam', frame_idx,0)],
                                                        samples[('K',0)], upscale=3)

                object_pixel_wise_motion_dict[('object_pixel_wise_motion', frame_idx, 0)] = object_pixel_wise_motion * object_motion_burn_in_weight
            # Further post-processing
            overlapping_mask = (torch.sum(object_pixel_wise_motion_dict[('object_pixel_wise_motion',-1,0)],dim=1,keepdim=True)!=0) & (torch.sum(object_pixel_wise_motion_dict[('object_pixel_wise_motion',1,0)],dim=1,keepdim=True)!=0)
            for frame_idx in self.loaded_frame_idxs[1:]: 
                object_pixel_wise_motion_dict[('object_pixel_wise_motion', frame_idx, 0)] = object_pixel_wise_motion_dict[('object_pixel_wise_motion', frame_idx, 0)] * overlapping_mask 
        # Get multiscale pixel-wise motion
        for frame_idx in self.loaded_frame_idxs[1:]: 
            for s in range(self.num_scales):
                if s == 0: continue
                object_pixel_wise_motion_dict[('object_pixel_wise_motion', frame_idx, s)] = object_pixel_wise_motion_dict[('object_pixel_wise_motion', frame_idx, 0)]
        return object_pixel_wise_motion_dict

    def process_batch(self, inputs, epoch, get_prediction_only=False, predict_object_motion=True):
        # 0. Process data
        for key, ipt in inputs.items(): 
            if type(ipt) is torch.Tensor: inputs[key] = ipt.to(self.device)
        outputs = {}
        # 1. Predict camera pose
        # if self.depth_net_type=='manydepth' and epoch >= self.many_depth_teacher_freeze_epoch:
        #     raise NotImplementedError
        #     with torch.no_grad(): cam_T_cam_dict = predict_pose(inputs)
        # else:
        cam_T_cam_dict = self.predict_pose(inputs)
        outputs.update(cam_T_cam_dict)
        # 2. Predict depth
        # We need to predict additional depth for the teacher monodepth2
        # if self.depth_net_type=='manydepth':
        #     if epoch >= many_depth_teacher_freeze_epoch:
        #         with torch.no_grad(): mono_predicted_depth_dict_ = predict_depth(inputs, outputs, mono_depth_net, depth_net_type='monodepth2')
        #     else:  mono_predicted_depth_dict_ = predict_depth(inputs, outputs, mono_depth_net, depth_net_type='monodepth2')
        #     mono_predicted_depth_dict = {}
        #     for key in mono_predicted_depth_dict_: 
        #         mono_predicted_depth_dict[(f'mono_{key[0]}',key[1],key[2])] = mono_predicted_depth_dict_[key]
        #     outputs.update(mono_predicted_depth_dict)
        predicted_depth_dict = self.predict_depth(inputs, outputs)
        # At the first epoch of training manydepth, we only train the posenet because we already initialize the depth net from the pre-trained model
        # if depth_net_type=='manydepth' and epoch == 0: 
        #     predicted_depth_dict = {key: predicted_depth_dict[key].detach() for key in predicted_depth_dict}
        outputs.update(predicted_depth_dict)
        # 3. Get the fw-warped image
        # if self.depth_net_type=='manydepth' and epoch >= self.many_depth_teacher_freeze_epoch:
        #     raise NotImplementedError
        #     with torch.no_grad(): fw_warped_dict = get_fw_warped_image(inputs, outputs)
        # else: 
        fw_warped_dict = self.get_fw_warped_image(inputs, outputs)
        outputs.update(fw_warped_dict)
        # 4. Predict 2d pixel-wise motion
        # Note that in case of manydepth, we use depth of monodepth2 to get the fw_warped image
        # This help to avoid 0 object motion caused by manydepth predicting exactly the triangulated depth
        # if self.depth_net_type=='manydepth' and epoch >= self.many_depth_teacher_freeze_epoch:
        #     raise NotImplementedError
        #     with torch.no_grad(): object_pixel_wise_motion_dict = predict_object_pixel_wise_motion(inputs, outputs, epoch)
        # else: 
        object_pixel_wise_motion_dict = self.predict_object_pixel_wise_motion(inputs, outputs, epoch, predict_object_motion)
        outputs.update(object_pixel_wise_motion_dict)
        # 5. Get the warped image
        # if self.depth_net_type=='manydepth':
        #     raise NotImplementedError
        #     image_pred_dict = generate_images_pred(inputs, outputs, is_multi=True)
        #     outputs.update(image_pred_dict)
        #     # if depth_net_type=='manydepth' and epoch >= many_depth_teacher_freeze_epoch:
        #     #     with torch.no_grad(): mono_image_pred_dict_ = generate_images_pred(inputs, outputs, is_multi=False)
        #     # else: 
        #     mono_image_pred_dict_ = generate_images_pred(inputs, outputs, is_multi=False)
        #     mono_image_pred_dict = {}
        #     for key in mono_image_pred_dict_: 
        #         mono_image_pred_dict[(f'mono_{key[0]}',key[1],key[2])] = mono_image_pred_dict_[key]
        #     outputs.update(mono_image_pred_dict)
        # else:
        image_pred_dict = self.generate_images_pred(inputs, outputs)
        outputs.update(image_pred_dict)

        if get_prediction_only: 
            return inputs, outputs

        # 6. Compute photometric loss
        # if self.depth_net_type=='manydepth':
        #     raise NotImplementedError
        #     losses_dict = {}
        #     many_losses_dict, stationary_mask = compute_losses(inputs, outputs, is_multi=True)
        #     # if depth_net_type=='manydepth' and epoch >= many_depth_teacher_freeze_epoch:
        #     #     with torch.no_grad(): mono_losses_dict_, mono_stationary_mask = compute_losses(inputs, outputs, is_multi=False)
        #     # else: 
        #     mono_losses_dict_, mono_stationary_mask = compute_losses(inputs, outputs, is_multi=False)
        #     mono_losses_dict = {}
        #     for key in mono_losses_dict_: mono_losses_dict[f'mono_{key}'] = mono_losses_dict_[key]
        #     losses_dict.update(many_losses_dict)
        #     losses_dict.update(mono_losses_dict)
        # else:
        losses_dict, stationary_mask = self.compute_losses(inputs, outputs, epoch)
        outputs['losses'] = losses_dict
        outputs['stationary_mask'] = stationary_mask
        # (For many depth only) Update adaptive bins of many depth
        # if self.depth_net_type == 'manydepth' and self.depth_net.training:
        #     global min_depth_tracker
        #     global max_depth_tracker
        #     # We do not update depth range while initializing RAFT and freezing the teacher model
        #     if epoch not in initial_raft_epochs[:-1] and epoch < many_depth_teacher_freeze_epoch:
        #         min_depth_tracker, max_depth_tracker = manydepth_update_adaptive_depth_bins(outputs, min_depth_tracker, max_depth_tracker)
        return inputs, outputs

    def set_train(self):
        self.depth_net.train()
        # if self.depth_net_type == 'manydepth':
        #     raise NotImplementedError
        #     if epoch < many_depth_teacher_freeze_epoch:
        #         pose_net.train()
        #         raft.train()
        #         mono_depth_net.train()
        #     else:
        #         pose_net.eval()
        #         raft.eval()
        #         mono_depth_net.eval()
        # else:
        self.pose_net.train()
        self.raft.train()

    def set_eval(self):
        self.depth_net.eval()
        self.pose_net.eval()
        self.raft.eval()
        # if depth_net_type == 'manydepth': mono_depth_net.eval()
        

    def val(self, epoch):
        self.set_eval()
        try: val_inputs = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_loader)
            val_inputs = next(self.val_iter)
        with torch.no_grad(): val_inputs, val_outputs = self.process_batch(val_inputs, epoch)
        self.set_train()
        return val_inputs, val_outputs

    def get_depth_for_evaluation(self, depth_loader):
        print('Start depth evaluation')
        self.set_eval()
        pred_disp_list = []
        with torch.no_grad():
            for inputs in tqdm(depth_loader):
                for key, ipt in inputs.items(): 
                    if type(ipt) is torch.Tensor: inputs[key] = ipt.to(self.device)
                outputs = {}
                # if self.depth_net_type == 'manydepth':
                #     raise NotImplementedError
                #     # 1. Predict camera pose
                #     cam_T_cam_dict = self.predict_pose(inputs, predict_pos_pose=False)
                #     outputs.update(cam_T_cam_dict)
                # 2. Predict depth
                pred_disp = self.predict_depth(inputs, outputs)[('disp',0,0)]
                pred_disp_list += [disp.detach().cpu().numpy() for disp in pred_disp]
        self.set_train()
        return pred_disp_list

    def epoch_eval(self, epoch):
        self.set_eval()
        epoch_val_running_loss = 0
        with torch.no_grad():
            for val_inputs in tqdm(self.val_loader):
                val_inputs, val_outputs = self.process_batch(val_inputs, epoch)
                val_total_loss = val_outputs['losses']['final_photometric_loss'] + val_outputs['losses']['final_depth_smooth'] + val_outputs['losses']['final_motion_sparsity_loss']
                # if self.depth_net_type=='manydepth':
                #     val_total_loss = val_total_loss + val_outputs['losses']['final_consistency_loss']
                #     val_total_loss = val_total_loss + val_outputs['losses']['mono_final_photometric_loss'] + val_outputs['losses']['mono_final_depth_smooth'] 
                epoch_val_running_loss += val_total_loss.item()*self.batch_size
        self.set_train()
        return epoch_val_running_loss

    def epoch_learning_curve_log(self, epoch_train_loss, epoch_val_loss, errors, epoch):
        writers = self.writers
        if epoch_train_loss is not None: writers['train'].add_scalar("a_epoch_eval/_loss", epoch_train_loss, epoch)
        if epoch_val_loss is not None: writers['val'].add_scalar("a_epoch_eval/_loss", epoch_val_loss, epoch)
        
        if errors is not None:
            all_region_mean_errors, static_mean_errors, dynamic_mean_errors = errors
            
            all_region_abs_rel, all_region_sq_rel, all_region_rmse, all_region_rmse_log, all_region_a1, all_region_a2, all_region_a3  = all_region_mean_errors
            writers['val'].add_scalar("b_all_region_depth/abs_rel", all_region_abs_rel, epoch)
            writers['val'].add_scalar("b_all_region_depth/sq_rel", all_region_sq_rel, epoch)
            writers['val'].add_scalar("b_all_region_depth/rmse", all_region_rmse, epoch)
            writers['val'].add_scalar("b_all_region_depth/rmse_log", all_region_rmse_log, epoch)
            writers['val'].add_scalar("b_all_region_depth/a1", all_region_a1, epoch)
            writers['val'].add_scalar("b_all_region_depth/a2", all_region_a2, epoch)
            writers['val'].add_scalar("b_all_region_depth/a3", all_region_a3, epoch)

            static_abs_rel, static_sq_rel, static_rmse, static_rmse_log, static_a1, static_a2, static_a3 = static_mean_errors
            writers['val'].add_scalar("c_static_depth/abs_rel", static_abs_rel, epoch)
            writers['val'].add_scalar("c_static_depth/sq_rel", static_sq_rel, epoch)
            writers['val'].add_scalar("c_static_depth/rmse", static_rmse, epoch)
            writers['val'].add_scalar("c_static_depth/rmse_log", static_rmse_log, epoch)
            writers['val'].add_scalar("c_static_depth/a1", static_a1, epoch)
            writers['val'].add_scalar("c_static_depth/a2", static_a2, epoch)
            writers['val'].add_scalar("c_static_depth/a3", static_a3, epoch)

            dynamic_abs_rel, dynamic_sq_rel, dynamic_rmse, dynamic_rmse_log, dynamic_a1, dynamic_a2, dynamic_a3  = dynamic_mean_errors
            writers['val'].add_scalar("d_dynamic_depth/abs_rel", dynamic_abs_rel, epoch)
            writers['val'].add_scalar("d_dynamic_depth/sq_rel", dynamic_sq_rel, epoch)
            writers['val'].add_scalar("d_dynamic_depth/rmse", dynamic_rmse, epoch)
            writers['val'].add_scalar("d_dynamic_depth/rmse_log", dynamic_rmse_log, epoch)
            writers['val'].add_scalar("d_dynamic_depth/a1", dynamic_a1, epoch)
            writers['val'].add_scalar("d_dynamic_depth/a2", dynamic_a2, epoch)
            writers['val'].add_scalar("d_dynamic_depth/a3", dynamic_a3, epoch)

    def epoch_mono_learning_curve_log(self, epoch_train_loss, epoch_val_loss, errors, epoch):
        writers = self.writers
        if epoch_train_loss is not None: writers['train'].add_scalar("a_epoch_eval/_loss", epoch_train_loss, epoch)
        if epoch_val_loss is not None: writers['val'].add_scalar("a_epoch_eval/_loss", epoch_val_loss, epoch)
        
        all_region_mean_errors, static_mean_errors, dynamic_mean_errors = errors
        
        all_region_abs_rel, all_region_sq_rel, all_region_rmse, all_region_rmse_log, all_region_a1, all_region_a2, all_region_a3  = all_region_mean_errors
        writers['val'].add_scalar("b_mono_all_region_depth/abs_rel", all_region_abs_rel, epoch)
        writers['val'].add_scalar("b_mono_all_region_depth/sq_rel", all_region_sq_rel, epoch)
        writers['val'].add_scalar("b_mono_all_region_depth/rmse", all_region_rmse, epoch)
        writers['val'].add_scalar("b_mono_all_region_depth/rmse_log", all_region_rmse_log, epoch)
        writers['val'].add_scalar("b_mono_all_region_depth/a1", all_region_a1, epoch)
        writers['val'].add_scalar("b_mono_all_region_depth/a2", all_region_a2, epoch)
        writers['val'].add_scalar("b_mono_all_region_depth/a3", all_region_a3, epoch)

        static_abs_rel, static_sq_rel, static_rmse, static_rmse_log, static_a1, static_a2, static_a3 = static_mean_errors
        writers['val'].add_scalar("c_mono_static_depth/abs_rel", static_abs_rel, epoch)
        writers['val'].add_scalar("c_mono_static_depth/sq_rel", static_sq_rel, epoch)
        writers['val'].add_scalar("c_mono_static_depth/rmse", static_rmse, epoch)
        writers['val'].add_scalar("c_mono_static_depth/rmse_log", static_rmse_log, epoch)
        writers['val'].add_scalar("c_mono_static_depth/a1", static_a1, epoch)
        writers['val'].add_scalar("c_mono_static_depth/a2", static_a2, epoch)
        writers['val'].add_scalar("c_mono_static_depth/a3", static_a3, epoch)

        dynamic_abs_rel, dynamic_sq_rel, dynamic_rmse, dynamic_rmse_log, dynamic_a1, dynamic_a2, dynamic_a3  = dynamic_mean_errors
        writers['val'].add_scalar("d_mono_dynamic_depth/abs_rel", dynamic_abs_rel, epoch)
        writers['val'].add_scalar("d_mono_dynamic_depth/sq_rel", dynamic_sq_rel, epoch)
        writers['val'].add_scalar("d_mono_dynamic_depth/rmse", dynamic_rmse, epoch)
        writers['val'].add_scalar("d_mono_dynamic_depth/rmse_log", dynamic_rmse_log, epoch)
        writers['val'].add_scalar("d_mono_dynamic_depth/a1", dynamic_a1, epoch)
        writers['val'].add_scalar("d_mono_dynamic_depth/a2", dynamic_a2, epoch)
        writers['val'].add_scalar("d_mono_dynamic_depth/a3", dynamic_a3, epoch)

    def get_prediction(self):
        print("Start getting prediction")
        torch.cuda.empty_cache()
        # Load the best model based evaluation metrics
        best_model_load_path = f"{self.log_path}/models/weights"
        load_model([self.depth_net, self.pose_net, self.raft], self.models_to_load, self.optimizer, best_model_load_path)
        # Reinitiate the dataset without shuffle and drop_last
        self.train_set.is_train = False
        train_loader = torch.utils.data.DataLoader(dataset=self.train_set, 
                                                batch_size=self.batch_size, 
                                                shuffle=False, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset=self.val_set, 
                                                batch_size=self.batch_size, 
                                                shuffle=False, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=False)
        # Set some dummy value for epoch and global_step
        epoch = 1000
        global_step = (epoch * (len(self.train_set)//self.batch_size+1))
        # Start getting prediction
        dataset_dict = {'train': train_loader, 'val': val_loader} 
        self.set_eval()
        with torch.no_grad():
            for split in dataset_dict.keys():
                print('Get prediction in split: ', split)
                data_loader = dataset_dict[split]
                for i, inputs in enumerate(tqdm(data_loader)):
                    frame_id_list = inputs['frame_id']
                    # Process batch
                    _, outputs_obj = self.process_batch(inputs, epoch, get_prediction_only=True, predict_object_motion=True)
                    _, outputs = self.process_batch(inputs, epoch, get_prediction_only=True, predict_object_motion=False)
                    # Get predicted depth and flow
                    predicted_depth = outputs[('depth',0,0)].detach().cpu() # b 1 h w 
                    predicted_object_flow_motion_neg = (outputs_obj[('uv',-1,0)] - outputs[('uv',-1,0)]).detach().cpu() # b 1 h w 
                    predicted_object_flow_motion_pos = (outputs_obj[('uv',1,0)] - outputs[('uv',1,0)]).detach().cpu() # b 1 h w 
                    # # Exclude pixels with very small flow
                    predicted_object_flow_motion_neg = predicted_object_flow_motion_neg*(torch.sum(torch.abs(predicted_object_flow_motion_neg),dim=1,keepdim=True)>=0.01)
                    predicted_object_flow_motion_pos = predicted_object_flow_motion_pos*(torch.sum(torch.abs(predicted_object_flow_motion_pos),dim=1,keepdim=True)>=0.01)
                    # Save the prediction into hard disk
                    # Require large storage here!!!
                    save_dir = f'{self.log_path}/predictions'
                    os.makedirs(save_dir, exist_ok=True)
                    for j in range(len(frame_id_list)):
                        city_id, full_frame_id = frame_id_list[j].split(' ')
                        os.makedirs(f'{save_dir}/depth/{split}/{city_id}', exist_ok=True)
                        os.makedirs(f'{save_dir}/flow_neg/{split}/{city_id}', exist_ok=True)
                        os.makedirs(f'{save_dir}/flow_pos/{split}/{city_id}', exist_ok=True)

                        np.save(f'{save_dir}/depth/{split}/{city_id}/{full_frame_id}.npy', predicted_depth[j].numpy()) # 1 h w
                        np.save(f'{save_dir}/flow_neg/{split}/{city_id}/{full_frame_id}.npy', predicted_object_flow_motion_neg[j].numpy()) # 1 h w
                        np.save(f'{save_dir}/flow_pos/{split}/{city_id}/{full_frame_id}.npy', predicted_object_flow_motion_pos[j].numpy()) # 1 h w



    def train(self):
        self.global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {self.global_step}")
        print()

        for epoch in range(self.current_epoch, self.epochs):

            print(f"*********************************")
            print(f"************ Epoch {epoch} ************")
            print(f"*********************************")
            
            # Initialize optimizer 
            if epoch in [0, self.initial_depth_pose_epochs[-1], self.initial_raft_epochs[-1]]: 
                self.init_optimizer_based_on_epoch(epoch)
            # if self.depth_net_type == 'manydepth' and self.many_depth_teacher_freeze_epoch is not None and epoch == self.many_depth_teacher_freeze_epoch:
            #     raise NotImplementedError
                # self.manydepth_freeze_teacher_and_pose_net()
            # Adjust learning rate
            if epoch in self.reduce_lr_epochs_list:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
                print("Change learning rate to ", g['lr'])
            # Main training loop
            epoch_train_running_loss = 0
            for i, inputs in enumerate(tqdm(self.train_loader)):
                # Process batch
                inputs, outputs = self.process_batch(inputs, epoch)
                # Back propagation
                total_loss = outputs['losses']['final_photometric_loss'] + outputs['losses']['final_depth_smooth'] + outputs['losses']['final_motion_sparsity_loss']
                # if self.depth_net_type=='manydepth':
                #     total_loss = total_loss + outputs['losses']['final_consistency_loss']
                #     total_loss = total_loss + outputs['losses']['mono_final_photometric_loss'] + outputs['losses']['mono_final_depth_smooth'] 
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping
                for model in [self.raft, self.depth_net, self.pose_net]:  
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                self.optimizer.step()
                # Keep track of the running loss
                epoch_train_running_loss += total_loss.item()*self.batch_size
                # EVAL IF EVAL STEP REACHED
                __eval_steps__ = self.first_epoch_eval_steps if epoch==0 else self.eval_steps
                if self.global_step % __eval_steps__ == 0 and self.global_step > 0:
                    val_inputs, val_outputs = self.val(epoch)
                    self.log(inputs, outputs, val_inputs, val_outputs, self.global_step)
                    del val_inputs, val_outputs
                self.global_step += 1

            # Evaluate and save model every K epoch
            if epoch % self.eval_and_save_every_epoch == 0:
                # Epoch evaluation
                epoch_val_running_loss = self.epoch_eval(epoch)
                # # Epoch depth evaluation
                pred_disp_list = self.get_depth_for_evaluation(self.val_depth_loader)
                depth_errors = compute_errors_all_dataset(pred_disp_list, self.val_depth_loader, self.val_gt_depth_path, self.val_gt_mask_path)
                # # Epoch logging
                epoch_train_loss = epoch_train_running_loss / (len(self.train_loader)*self.batch_size)
                epoch_val_loss = epoch_val_running_loss / (len(self.val_loader)*self.batch_size)
                self.epoch_learning_curve_log(epoch_train_loss, epoch_val_loss, depth_errors, epoch)
                # Save model
                save_model([self.depth_net,self.pose_net,self.raft], ['depth_net', 'pose_net','raft'], self.optimizer, self.log_path, epoch)
                if depth_errors[0][0] < self.smallest_error:
                    self.smallest_error = depth_errors[0][0]
                    save_model([self.depth_net,self.pose_net,self.raft], ['depth_net', 'pose_net','raft'], self.optimizer, self.log_path, None)
                    print(f"Best model detected. Save model at epoch {epoch} with abs_error = {self.smallest_error.item()}")

            # if self.depth_net_type == 'manydepth' and epoch <= self.many_depth_teacher_freeze_epoch:
            #     raise NotImplementedError
            #     mono_pred_disp_list = get_depth_for_evaluation(mono_depth_net, depth_net_type='monodepth2')
            #     mono_depth_errors = compute_errors_all_dataset(mono_pred_disp_list, val_depth_loader, val_gt_depth_path, val_gt_mask_path)
            #     epoch_mono_learning_curve_log(writers, None, None, mono_depth_errors, epoch)

            # # Save model
            # if self.depth_net_type == 'manydepth':
            #     raise NotImplementedError
            #     np.savetxt(f'{log_path}/models/weights_{epoch}/depth_range.txt', np.array([min_depth_tracker, max_depth_tracker]))


    def eval_depth(self):
        if self.eval_model_load_path is None:
            print(f"Not model_load_path specified ==> Load the best model in {self.log_path}/models/weights/depth_net.pth")
            self.depth_net.load_state_dict(torch.load(f"{self.log_path}/models/weights/depth_net.pth"))
        else:
            print(f"Load model in {self.eval_model_load_path}")
            self.depth_net.load_state_dict(torch.load(self.eval_model_load_path))
        self.depth_net.eval()
        pred_disp_list = self.get_depth_for_evaluation(self.test_depth_loader)
        depth_errors = compute_errors_all_dataset(pred_disp_list, self.test_depth_loader, self.test_gt_depth_path, self.test_gt_mask_path)
        all_errors, static_errors, dynamic_errors = depth_errors
        print("********************************* All region *********************************")
        print("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*all_errors.tolist()) + "\\\\")
        print("********************************* Static region *********************************")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*static_errors.tolist()) + "\\\\")
        print("********************************* Dynamic region *********************************")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*dynamic_errors.tolist()) + "\\\\")
        return pred_disp_list, depth_errors


    
