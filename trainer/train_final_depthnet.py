
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

from datasets_mono import CityscapesPreprocessedDataset, CityscapesEvalDataset
from utils.utils import readlines, load_weights_from_tfResNet_to_depthEncoder, colormap, manydepth_update_adaptive_depth_bins, Namespace
from layers import transformation_from_parameters, back_project_depth, project_3d, resample, forward_warp
# from forward_warp import forward_warp
from model_zoo.model_zoo import *
from loss import joint_bilateral_smoothing, rgbd_consistency_loss, sqrt_motion_sparsity, depth_scale_invariant_loss
from utils.validation_utils import compute_errors_all_dataset
from utils.checkpoints_utils import *
from raft_core.raft import RAFT
from trainer.train_pixel_wise import PixelWiseTrainer
from matplotlib import pyplot as plt
import numpy as np
import cv2



class FinalDepthTrainer(PixelWiseTrainer):

    def __init__(self, cfg):
        super(FinalDepthTrainer, self).__init__(cfg)

        # Depth network architecture
        self.cfg = cfg
        self.project_path = cfg['log_path']
        self.log_path = f"{self.cfg['log_path']}/final_depthnet"
        # Log set up 
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=10)
        self.current_epoch = 0
        self.global_step = 0
        self.smallest_error = 10e10


    def get_training_filenames(self, split):
        original_files_path = self.cfg[f'{split}_file_path']
        dynamic_object_train_files_path = f'{self.project_path}/depth_scale_alignment/dynamic_object_training_files/{split}/dynamic_object_filenames.txt'
        train_files = sorted(np.loadtxt(original_files_path, dtype=object, delimiter='\n').tolist())
        dynamic_object_train_files = sorted(np.loadtxt(dynamic_object_train_files_path, dtype=object, delimiter='\n').tolist())
        dynamic_object_train_files_wo_object = [f"{file.split(' ')[0]} {file.split(' ')[1]}" for file in dynamic_object_train_files]
        difference = list(set(train_files) - set(dynamic_object_train_files_wo_object))
        for file in difference: dynamic_object_train_files.append(f'{file} -1')
        save_dir = f'{self.log_path}/extracted_filenames/{split}'
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(f'{save_dir}/dynamic_object_filenames.txt', sorted(dynamic_object_train_files), fmt="%s")

    def training_setup(self):
        # Dataloading
        # self.train_data_dir = cfg['train_data_dir']
        self.train_file_path = f'{self.log_path}/extracted_filenames/train/dynamic_object_filenames.txt'
        self.train_object_mask_path = f"{self.project_path}/object_depthnet/processed_mask/train"
        self.train_pseudo_label_path = f"{self.project_path}/depth_scale_alignment/pseudo_depth/train"
        # self.val_data_dir = cfg['val_data_dir']
        self.val_file_path = f'{self.log_path}/extracted_filenames/val/dynamic_object_filenames.txt'
        self.val_object_mask_path = f"{self.project_path}/object_depthnet/processed_mask/val"
        self.val_pseudo_label_path = f"{self.project_path}/depth_scale_alignment/pseudo_depth/val"

        # Training setting
        self.initial_depth_pose_epochs = [-3,-2] # Set to these value means we training all models
        self.initial_raft_epochs = [-2,-1] # Set to these value means we training all models
        self.reduce_lr_epochs_list = [15]
        self.epochs = self.cfg['final_depthnet_epochs']
        self.learning_rate = self.cfg['final_depthnet_learning_rate']
        self.initialized_weights_folder = f"{self.cfg['log_path']}/pixelwise_depthnet/models/weights"
        self.depth_eval_steps = self.cfg['final_depthnet_depth_eval_steps']

        self.dynamic_depth_loss_weight = self.cfg['dynamic_depth_loss_weight']

        # Logging
        # self.log_path = f"{self.cfg['log_path']}/final_depthnet"
        self.load_weights_folder = self.cfg['final_depthnet_load_weights_folder']

        # Dataloading
        self.train_file = sorted(readlines(self.train_file_path))
        self.val_file = sorted(readlines(self.val_file_path))
        self.train_set = CityscapesPreprocessedDataset(data_path = self.train_data_dir,
                                          pseudo_label_path = self.train_pseudo_label_path,
                                          object_mask_path=self.train_object_mask_path,
                                          filenames = self.train_file,
                                          height = self.height,
                                          width = self.width,
                                          frame_idxs = self.loaded_frame_idxs,
                                          num_scales = self.num_scales,
                                          is_train=True,
                                          img_ext=self.img_ext)
        self.val_set = CityscapesPreprocessedDataset(data_path = self.val_data_dir,
                                          pseudo_label_path = self.val_pseudo_label_path,
                                          object_mask_path=self.val_object_mask_path,
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
        self.val_iter = iter(self.val_loader)

        # Load model
        if self.initialized_weights_folder is not None and self.load_weights_folder is None:
            print("Initialize from pixelwise_depthnet succesfully")
            load_model([self.depth_net, self.pose_net, self.raft], self.models_to_load, self.optimizer, self.initialized_weights_folder)
        if self.load_weights_folder is not None:
            try: 
                self.current_epoch = int(self.load_weights_folder.split('_')[-1])

            except:
                self.current_epoch = 10000 # Set dummy for current_epoch when load the best model
            print(f"Load model succesfully at epoch {self.current_epoch}")
            load_model([self.depth_net, self.pose_net, self.raft], self.models_to_load, self.optimizer, self.load_weights_folder)
            self.current_epoch = self.current_epoch + 1

    def epoch_eval(self, epoch):
        self.set_eval()
        epoch_val_running_loss = 0
        with torch.no_grad():
            for val_inputs in tqdm(self.val_loader):
                val_inputs, val_outputs = self.process_batch(val_inputs, epoch)
                val_total_loss = val_outputs['losses']['final_photometric_loss'] + val_outputs['losses']['final_depth_smooth'] + val_outputs['losses']['final_motion_sparsity_loss'] + val_outputs['losses']['final_dynamic_depth_loss']
                # if self.depth_net_type=='manydepth':
                #     val_total_loss = val_total_loss + val_outputs['losses']['final_consistency_loss']
                #     val_total_loss = val_total_loss + val_outputs['losses']['mono_final_photometric_loss'] + val_outputs['losses']['mono_final_depth_smooth'] 
                epoch_val_running_loss += val_total_loss.item()*self.batch_size
        self.set_train()
        return epoch_val_running_loss


    def log(self, train_inputs, train_outputs, val_inputs, val_outputs, global_step):
        with torch.no_grad():
            losses = {"train": train_outputs['losses'], "val": val_outputs['losses']}
            endpoints = {"train": train_inputs, "val": val_inputs}
            output_endpoints = {"train": train_outputs, "val": val_outputs}

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

                    ref_disp_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('disp',0,s)][[j]].detach().cpu(), [self.height, self.width], mode="bilinear", align_corners=False) for s in range(self.num_scales)])
                    ref_disp_all_colmap = torch.from_numpy(colormap(ref_disp_all)) # 4 3 h w 
                    ref_disp_0_colmap = ref_disp_all_colmap[0]

                    teacher_disp_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('teacher_disp',0,s)][[j]].detach().cpu(), [self.height, self.width], mode="bilinear", align_corners=False) for s in range(self.num_scales)])
                    teacher_disp_all_colmap = torch.from_numpy(colormap(teacher_disp_all)) # 4 3 h w 
                    teacher_disp_0_colmap = teacher_disp_all_colmap[0]

                    try: neg_optical_flow = flow_to_image(neg_object_motion).float()/255.0
                    except: neg_optical_flow = torch.zeros(3,neg_object_motion.shape[1],neg_object_motion.shape[2])
                    neg_stack_img = torch.concat([teacher_disp_0_colmap,
                                                ref_disp_0_colmap,
                                                ref_frame,
                                                neg_warped_frame,
                                                fw_neg_warped_frame,
                                                neg_src_frame,
                                                neg_optical_flow
                                                ], dim=1)

                    try: pos_optical_flow = flow_to_image(pos_object_motion).float()/255.0
                    except: pos_optical_flow = torch.zeros(3,pos_object_motion.shape[1],pos_object_motion.shape[2])
                    pos_stack_img = torch.concat([teacher_disp_0_colmap,
                                                ref_disp_0_colmap,
                                                ref_frame,
                                                pos_warped_frame,
                                                fw_pos_warped_frame,
                                                pos_src_frame,
                                                pos_optical_flow
                                                ], dim=1)

                    disp_all_stack = torch.concat([disp for disp in ref_disp_all_colmap], dim=1)
                    teacher_disp_all_stack = torch.concat([disp for disp in teacher_disp_all_colmap], dim=1)

                    neg_motion_stack = torch.concat([flow_to_image(object_motion).float()/255.0 for object_motion in neg_object_motion_all], dim=1)
                    pos_motion_stack = torch.concat([flow_to_image(object_motion).float()/255.0 for object_motion in pos_object_motion_all], dim=1)

                    writer.add_image(f"a_visualzaition_-1/{j}/{mode}", neg_stack_img, global_step)
                    writer.add_image(f"b_visualzaition_1/{j}/{mode}", pos_stack_img, global_step)
                    writer.add_image(f"c_disp_multiscale/{j}/{mode}", disp_all_stack, global_step)
                    writer.add_image(f"c1_teacher_disp_multiscale/{j}/{mode}", teacher_disp_all_stack, global_step)
                    writer.add_image(f"d_neg_motion_multiscale/{j}/{mode}", neg_motion_stack, global_step)
                    writer.add_image(f"e_pos_motion_multiscale/{j}/{mode}", pos_motion_stack, global_step)
                    writer.add_image(f"f_stationary_mask/{j}/{mode}", output_endpoint['stationary_mask'][[j]].detach().cpu(), global_step)


    def compute_losses(self, inputs, outputs, epoch, is_multi=False):
        dynamic_object_mask = inputs[('object_mask',0,0)]
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
            else:
                consistency_loss = 0

        # Compute motion sparsity loss (at the highest resolution only)
        object_pixel_wise_motion_stack = torch.concat([outputs[('object_pixel_wise_motion',f,0)] for f in self.loaded_frame_idxs[1:]])
        motion_sparsity_loss = epoch_sparsity_loss_weight*sqrt_motion_sparsity(object_pixel_wise_motion_stack)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Compute depth loss in dynamic region
        dynamic_depth_dict = {}
        for s in range(self.num_scales):
            # if self.depth_net_type == 'manydepth' and not is_multi: 
            #     depth_s = outputs[('mono_depth',0,s)]
            # else:
            depth_s = outputs[('depth',0,s)]
            dynamic_depth_loss, valid_depth_loss_mask_list = depth_scale_invariant_loss(depth_s, outputs[('teacher_depth',0,s)], lambda_=0.5)
            valid_depth_loss_mask_list = valid_depth_loss_mask_list*torch.nn.functional.interpolate(dynamic_object_mask, (valid_depth_loss_mask_list.shape[2],valid_depth_loss_mask_list.shape[3]), mode='nearest')
            dynamic_depth_loss = torch.sum(dynamic_depth_loss*valid_depth_loss_mask_list) / (torch.sum(valid_depth_loss_mask_list)+1e-10)
            dynamic_depth_dict[f'dynamic_depth_loss_{s}'] = dynamic_depth_loss*self.dynamic_depth_loss_weight
            # final_dynamic_depth_loss += dynamic_depth_dict[('loss/dynamic_depth_loss',s)]
        losses_dict.update(dynamic_depth_dict)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Summ all losses
        losses_dict['final_motion_sparsity_loss'] = motion_sparsity_loss
        losses_dict['final_photometric_loss'] = sum([losses_dict[f'photometric_loss_{s}'] for s in range(self.num_scales)]) / self.num_scales
        losses_dict['final_depth_smooth'] = sum([losses_dict[f'depth_smooth_{s}'] for s in range(self.num_scales)]) / self.num_scales
        losses_dict['final_dynamic_depth_loss'] = sum([losses_dict[f'dynamic_depth_loss_{s}'] for s in range(self.num_scales)]) / self.num_scales
        
        # if is_multi and self.depth_net_type=='manydepth':
        #     losses_dict['final_consistency_loss'] = sum([losses_dict[f'consistency_loss_{s}'] for s in range(self.num_scales)]) / self.num_scales

        return losses_dict, stationary_mask

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
        predicted_depth_dict = self.predict_depth(inputs, outputs)
        outputs.update(predicted_depth_dict)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 2.5 Pre-process pseudo depth label
        pseudo_depth_label = inputs[('pseudo_depth_label',0,0)]
        for i in range(len(pseudo_depth_label)):
            # If there is no pseudo depth. We use the currently predicted as a dummy pseudo depth label
            if torch.sum(pseudo_depth_label[i]) == 0: pseudo_depth_label[i] = torch.clone(predicted_depth_dict[('depth',0,0)][i].detach())
        pseudo_depth_label_list = []
        # Get psedeu depth label at  multi-scales
        for s in range(self.num_scales):
            if self.depth_net_type == 'brnet':
                if s != 0: pseudo_depth_label_s = torch.nn.functional.interpolate(pseudo_depth_label, (self.height//(2**(s-1)),self.width//(2**(s-1))), mode='bilinear', align_corners=False)
                else: pseudo_depth_label_s = pseudo_depth_label
            else: pseudo_depth_label_s = torch.nn.functional.interpolate(pseudo_depth_label, (self.height//(2**s),self.width//(2**s)), mode='bilinear', align_corners=False)
            pseudo_depth_label_list.append(pseudo_depth_label_s)
        pseudo_depth_label = pseudo_depth_label_list
        for s in range(self.num_scales):
                outputs[('teacher_depth',0,s)] = pseudo_depth_label[s]
                outputs[('teacher_disp',0,s)] = 1/outputs[('teacher_depth',0,s)]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        if self.depth_net_type=='manydepth':
            raise NotImplementedError
            image_pred_dict = generate_images_pred(inputs, outputs, is_multi=True)
            outputs.update(image_pred_dict)
            # if depth_net_type=='manydepth' and epoch >= many_depth_teacher_freeze_epoch:
            #     with torch.no_grad(): mono_image_pred_dict_ = generate_images_pred(inputs, outputs, is_multi=False)
            # else: 
            mono_image_pred_dict_ = generate_images_pred(inputs, outputs, is_multi=False)
            mono_image_pred_dict = {}
            for key in mono_image_pred_dict_: 
                mono_image_pred_dict[(f'mono_{key[0]}',key[1],key[2])] = mono_image_pred_dict_[key]
            outputs.update(mono_image_pred_dict)
        else:
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
        return inputs, outputs

    def train(self):
        self.global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {self.global_step}")
        print()

        if self.current_epoch == 0:
            # # Epoch depth evaluation
            pred_disp_list = self.get_depth_for_evaluation(self.val_depth_loader)
            depth_errors = compute_errors_all_dataset(pred_disp_list, self.val_depth_loader, self.val_gt_depth_path, self.val_gt_mask_path)
            # # Epoch logging
            self.epoch_learning_curve_log(None, None, depth_errors, -1)

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
                total_loss = outputs['losses']['final_photometric_loss'] + outputs['losses']['final_depth_smooth'] + outputs['losses']['final_motion_sparsity_loss']  + outputs['losses']['final_dynamic_depth_loss']
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
                # Perform depth evaluation within an epoch for better convergence
                if self.global_step % self.depth_eval_steps == 0 and self.global_step>0:
                    # Epoch depth evaluation
                    pred_disp_list = self.get_depth_for_evaluation(self.val_depth_loader)
                    depth_errors = compute_errors_all_dataset(pred_disp_list, self.val_depth_loader, self.val_gt_depth_path, self.val_gt_mask_path)
                    self.epoch_learning_curve_log(None, None, depth_errors, self.global_step)
                    save_model([self.depth_net,self.pose_net,self.raft], ['depth_net', 'pose_net','raft'], self.optimizer, self.log_path, self.global_step)
                    if depth_errors[0][0] < self.smallest_error:
                        self.smallest_error = depth_errors[0][0]
                        save_model([self.depth_net,self.pose_net,self.raft], ['depth_net', 'pose_net','raft'], self.optimizer, self.log_path, None)
                        print(f"Best model detected. Save model at step {self.global_step} with abs_error = {self.smallest_error.item()}")
                self.global_step += 1

            # Evaluate and save model every K epoch
            if epoch % self.eval_and_save_every_epoch == 0:
                # Epoch evaluation
                epoch_val_running_loss = self.epoch_eval(epoch)
                # # Epoch logging
                epoch_train_loss = epoch_train_running_loss / (len(self.train_loader)*self.batch_size)
                epoch_val_loss = epoch_val_running_loss / (len(self.val_loader)*self.batch_size)
                self.epoch_learning_curve_log(epoch_train_loss, epoch_val_loss, None, epoch)
                # Save model
                # save_model([self.depth_net,self.pose_net,self.raft], ['depth_net', 'pose_net','raft'], self.optimizer, self.log_path, epoch)

            # if self.depth_net_type == 'manydepth' and epoch <= self.many_depth_teacher_freeze_epoch:
            #     raise NotImplementedError
                # mono_pred_disp_list = get_depth_for_evaluation(mono_depth_net, depth_net_type='monodepth2')
                # mono_depth_errors = compute_errors_all_dataset(mono_pred_disp_list, val_depth_loader, val_gt_depth_path, val_gt_mask_path)
                # epoch_mono_learning_curve_log(writers, None, None, mono_depth_errors, epoch)

            # # Save model
            # if self.depth_net_type == 'manydepth':
            #     raise NotImplementedError
                # np.savetxt(f'{log_path}/models/weights_{epoch}/depth_range.txt', np.array([min_depth_tracker, max_depth_tracker]))

