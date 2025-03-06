import os
import numpy as np
from matplotlib import pyplot as plt 
import torch
from tqdm import tqdm
import imageio
from joblib import Parallel, delayed
from utils.mask_utils import *
import pickle as pkl
from datasets_mono import CityscapesPreprocessedDataset
from datasets_mono.object_test_dataset import CityscapeObjectTestDataset
from utils.utils import readlines

import os
import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets_mono import CityscapesPreprocessedDataset
from utils.utils import readlines, load_weights_from_tfResNet_to_depthEncoder, colormap, manydepth_update_adaptive_depth_bins, Namespace
from layers import transformation_from_parameters, back_project_depth, project_3d, resample, forward_warp, matrix_from_angles
# from forward_warp import forward_warp
from model_zoo.model_zoo import *
from loss import object_disp_smoothing, rgbd_consistency_loss
from utils.validation_utils import compute_object_errors_all_dataset
from utils.checkpoints_utils import *
from utils.mask_utils import *

class ObjectDepthTrainer:
    def __init__(self, cfg, pixel_wise_trainer=None):
        # Depth network architecture
        self.device = cfg['device']
        self.cfg = cfg
        # self.depth_net_type = cfg['depth_net_type']
        # self.pose_net_type = cfg['pose_net_type']
        # self.depth_encoder_pretrained = cfg['depth_encoder_pretrained']
        # self.encoder_use_randomize_layernorm = cfg['encoder_use_randomize_layernorm']
        # self.tf_imageNet_checkpoint_path = cfg['tf_imageNet_checkpoint_path']
        # self.many_depth_teacher_freeze_epoch = cfg['many_depth_teacher_freeze_epoch']
        # # Loss weight
        # self.ssim_c1 = float('inf') if cfg['ssim_c1'] == 'inf' else cfg['ssim_c1']
        # self.ssim_c2 = cfg['ssim_c2']
        # self.rgb_consistency_weight = cfg['rgb_consistency_weight']
        # self.ssim_weight = cfg['ssim_weight']
        # self.photometric_error_weight = cfg['photometric_error_weight']
        # self.depth_smoothing_weight = cfg['depth_smoothing_weight']
        # self.sparsity_loss_weight = cfg['sparsity_loss_weight']
        # self.consistency_loss_weight = cfg['consistency_loss_weight']
        # Dataloading
        self.batch_size = cfg['batch_size']
        self.height = cfg['height']
        self.width = cfg['width']
        self.loaded_frame_idxs = cfg['loaded_frame_idxs']
        self.num_scales = cfg['num_scales']
        self.num_workers = cfg['num_workers']
        self.img_ext = cfg['img_ext']
        self.train_data_dir = cfg['train_data_dir']
        self.val_data_dir = cfg['val_data_dir']

        # Logging
        self.project_path = f"{cfg['log_path']}"
        self.log_path = f"{cfg['log_path']}/object_depthnet"
        # Log set up 
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=10)

        # Parmaters for object mask pre-processing
        self.use_pretrained_mask = cfg['use_pretrained_mask']
        self.raw_mask_path = cfg['raw_mask_path']
        self.raw_road_mask_path = cfg['raw_road_mask_path']
        self.is_dynamic_threshold = cfg['is_dynamic_threshold']
        self.flow_threshold = cfg['flow_threshold']
        self.object_mask_size_theshold = cfg['object_mask_size_theshold']
        self.middle_portion = cfg['middle_portion']
        self.num_cores = cfg['num_cores']  # Adjust this to the number of CPU cores you want to use
        # self.object_retrieving_batch_size = cfg['object_retrieving_batch_size']
        # if not self.use_pretrained_mask: raise NotImplementedError
        self.processed_mask_save_path = f"{self.log_path}/processed_mask/"
        self.predicted_flow_neg_path = f"{self.project_path}/pixelwise_depthnet/predictions/flow_neg"
        self.predicted_flow_pos_path = f"{self.project_path}/pixelwise_depthnet/predictions/flow_pos"
        self.pixel_wise_trainer = pixel_wise_trainer

    
    def process_and_classify_object_mask(self, split):
        if split == "train":
            frame_id_list = sorted(np.loadtxt(self.cfg['train_file_path'], dtype=object, delimiter='\n').tolist())
        elif split == "val":
            frame_id_list = sorted(np.loadtxt(self.cfg['val_file_path'], dtype=object, delimiter='\n').tolist())
        else: raise NotImplementedError
        results = Parallel(n_jobs=self.num_cores)(
            delayed(proprocess_and_classify_raw_mask)(
                full_frame_id,
                self.raw_mask_path,
                self.raw_road_mask_path,
                self.predicted_flow_neg_path,
                self.predicted_flow_pos_path,
                self.flow_threshold,
                self.is_dynamic_threshold,
                self.middle_portion,
                self.processed_mask_save_path,
                split=split
            ) for full_frame_id in tqdm(frame_id_list)
        )
        # Extract results
        static_dynamic_label_dict = {}
        for frame_id, labels in results:
            if labels is not None:
                static_dynamic_label_dict[frame_id] = labels
        # Save the result dictionary
        os.makedirs(f"{self.log_path}/dynamic_static_labels/{split}", exist_ok=True)
        with open(f"{self.log_path}/dynamic_static_labels/{split}/dynamic_label_dict.pkl", 'wb') as f :
            pkl.dump(static_dynamic_label_dict, f)
        return static_dynamic_label_dict


    def extract_training_object_file(self, static_dynamic_label_dict, split):
        frame_with_labels = sorted(list(static_dynamic_label_dict.keys()))

        extract_mask_file_id_list = []
        result_dict = {}
        for f,full_frame_id in enumerate(tqdm(frame_with_labels)):
            city_id = full_frame_id.split("_")[0]
            # Get ground mask
            # if not os.path.isfile(f'{raw_road_mask_path}/{split}/{city_id}/{full_frame_id}.png'): continue
            # Get object mask
            mask_file_list = os.listdir(f"{self.raw_mask_path}/{split}/{city_id}/{full_frame_id}")
            mask_file_list = sorted(mask_file_list, key=lambda x: int(x[:-4]))
            mask_list =  [plt.imread(f"{self.raw_mask_path}/{split}/{city_id}/{full_frame_id}/{mask_file}") for mask_file in mask_file_list]
            mask_list = np.stack(mask_list)
            # Extract samples with target mask_size, and dynamic labels
            static_dynamic_labels = static_dynamic_label_dict[full_frame_id]
            extracted_mask_idx_ = extract_valid_mask(mask_list, static_dynamic_labels, self.object_mask_size_theshold, (self.height, self.width))
            extracted_mask_idx = np.array(mask_file_list)[extracted_mask_idx_[0].flatten()]
            # city_id frame_id object_id static_dynamic_label
            extract_mask_file_id_list +=[f"{city_id} {full_frame_id} {mask_file_id.replace('.png','')} {static_dynamic_labels[int(mask_file_id.replace('.png',''))]}" for m,mask_file_id in enumerate(extracted_mask_idx)]

        extract_mask_file_id_list = np.array(extract_mask_file_id_list)

        os.makedirs(f'{self.log_path}/object_train_file/{split}', exist_ok=True)
        np.savetxt(f'{self.log_path}/object_train_file/{split}/object_files.txt', extract_mask_file_id_list, fmt='%s', delimiter='\n')
        return extract_mask_file_id_list
    
    def get_src_warped_object_masks(self, inputs, outputs):
        b, _, h, w = inputs[('color_aug',0,0)].shape
        ref_object_masks = inputs[('object_mask', 0, 0)].float().cuda()
        warped_object_masks_dict = {}
        for frame_idx in self.cfg['loaded_frame_idxs'][1:]:
            object_pixel_wise_motion = outputs[('object_pixel_wise_motion', frame_idx, 0)]
            deltaT = torch.bmm(inputs[("inv_K",0)][:,:3,:3], 
                                            torch.cat([object_pixel_wise_motion, torch.zeros_like(object_pixel_wise_motion[:,[0]])], dim=1).view(len(object_pixel_wise_motion),3,-1))
            deltaT_inv = -torch.bmm(torch.inverse(matrix_from_angles(outputs[('rot', frame_idx, 0)])), deltaT) # b 3 h*w
            deltaT_inv = deltaT_inv.view(len(deltaT_inv),3,h,w) # b 3 h w
            deltaT = deltaT.view(deltaT_inv.shape)
            (src_object_mask, _,) = forward_warp(outputs[('depth',0,0)], ref_object_masks, 
                                                outputs[('cam_T_cam',frame_idx,0)], 
                                                torch.inverse(outputs[('cam_T_cam', frame_idx,0)]), 
                                                inputs[('K',0)], upscale=3,
                                                deltaT=deltaT_inv, deltaT_inv=deltaT)
            src_object_mask = (src_object_mask >= 0.5).float()  
            warped_object_masks_dict[('warped_object_masks',frame_idx,0)] = src_object_mask
        return warped_object_masks_dict

    def retrieve_corresponding_mask(self, split):
        train_file = readlines(f'{self.log_path}/object_train_file/{split}/object_files.txt')
        if split == 'train': data_path = self.train_data_dir 
        if split == 'val': data_path = self.val_data_dir 
        # Define dataset used to find object correspondence
        object_dataset = CityscapesPreprocessedDataset(data_path = data_path,
                                                object_mask_path = f'{self.processed_mask_save_path}/{split}',
                                                filenames = train_file,
                                                height = self.height,
                                                width = self.width,
                                                frame_idxs = self.loaded_frame_idxs,
                                                num_scales = self.num_scales,
                                                is_train=False,
                                                img_ext=self.img_ext)
        object_dataloader = torch.utils.data.DataLoader(dataset=object_dataset, 
                                                batch_size=self.batch_size, 
                                                shuffle=False, 
                                                num_workers=self.num_workers, 
                                                pin_memory=True, 
                                                drop_last=False)

        print("Start retrieving object correspondences")
        with torch.no_grad():
            for i, inputs in enumerate(tqdm(object_dataloader)):
                # Process batch
                inputs, outputs = self.pixel_wise_trainer.process_batch(inputs, epoch=10000, get_prediction_only=True)
                warped_object_masks_dict = self.get_src_warped_object_masks(inputs, outputs)

                ref_mask = inputs[('object_mask', 0, 0)].detach().cpu().numpy() # b 1 h w 
                src_neg_mask = warped_object_masks_dict[('warped_object_masks', -1, 0)].detach().cpu().numpy()
                src_pos_mask = warped_object_masks_dict[('warped_object_masks', 1, 0)].detach().cpu().numpy()

                mask_stack = np.concatenate([src_neg_mask, ref_mask, src_pos_mask], axis=-1)[:,0]
                for m in range(len(mask_stack)):
                    if np.sum(ref_mask[m]) <= self.object_mask_size_theshold*self.height*self.width: continue
                    if np.sum(src_neg_mask[m]) <= self.object_mask_size_theshold*self.height*self.width: continue
                    if np.sum(src_pos_mask[m]) <= self.object_mask_size_theshold*self.height*self.width: continue
                    city_id, frame_id, mask_id, _ = inputs['frame_id'][m].split(' ')
                    os.makedirs(f'{self.log_path}/object_mask_triplet/{split}/{city_id}/{frame_id}', exist_ok=True)
                    imageio.imsave(f'{self.log_path}/object_mask_triplet/{split}/{city_id}/{frame_id}/{mask_id}.png', mask_stack[m])
            
    def get_valid_train_mask_file(self, split):
        def get_file_names_list(dir):
            city_list = os.listdir(dir)
            filenames_list = []
            for city_id in city_list:
                city_filenames_list = os.listdir(f'{dir}/{city_id}')
                for frame_id in city_filenames_list:
                    frame_mask_filenames_list = os.listdir(f'{dir}/{city_id}/{frame_id}')
                    frame_mask_filenames_list = [f"{city_id} {frame_id} {file.replace('.png','')}" for file in frame_mask_filenames_list]
                # city_filenames = [f'{city_id} {file}' for file in city_filenames]
                    filenames_list += frame_mask_filenames_list
            return filenames_list
        def filter_out_files(valid_mask_filenames, original_filenames):
            retrieved_valid_file_names = []
            # valid_mask_filenames = valid_mask_filenames.tolist()
            for file in tqdm(original_filenames):
                city_id, frame_id, mask_id, label = file.split(' ')
                file_names = f'{city_id} {frame_id} {mask_id}'
                if file_names in valid_mask_filenames: 
                    retrieved_valid_file_names.append(file)
            return np.array(retrieved_valid_file_names)
        if split == 'train': 
            file_name_list = self.pixel_wise_trainer.train_file
        elif split == 'val': 
            file_name_list = self.pixel_wise_trainer.val_file
        else: 
            raise NotImplementedError
        mask_dir = f'{self.log_path}/object_mask_triplet/{split}'
        file_name_list = readlines(f'{self.log_path}/object_train_file/{split}/object_files.txt')
        valid_mask_filenames = get_file_names_list(mask_dir)
        valid_file = filter_out_files(valid_mask_filenames, file_name_list)
        np.savetxt(f'{self.log_path}/object_train_file/{split}/valid_object_files.txt', valid_file, fmt='%s', delimiter='\n')
        return valid_file
    
    def training_setup(self):
        # Delete the models associated with pixel-wise training 
        if self.pixel_wise_trainer: 
            del self.pixel_wise_trainer
            torch.cuda.empty_cache()
        # Depth network architecture
        self.depth_net_type = self.cfg['depth_net_type']
        self.pose_net_type = self.cfg['pose_net_type']
        self.depth_encoder_pretrained = self.cfg['depth_encoder_pretrained']
        self.encoder_use_randomize_layernorm = self.cfg['encoder_use_randomize_layernorm']
        self.tf_imageNet_checkpoint_path = self.cfg['tf_imageNet_checkpoint_path']
        # self.many_depth_teacher_freeze_epoch = self.cfg['many_depth_teacher_freeze_epoch']
        # Loss weight
        self.ssim_c1 = float('inf') if self.cfg['ssim_c1'] == 'inf' else self.cfg['ssim_c1']
        self.ssim_c2 = self.cfg['ssim_c2']
        self.rgb_consistency_weight = self.cfg['rgb_consistency_weight']
        self.ssim_weight = self.cfg['ssim_weight']
        self.photometric_error_weight = self.cfg['photometric_error_weight']
        self.depth_smoothing_weight = self.cfg['object_depth_smoothing_weight']
        # Dataloading
        self.batch_size = self.cfg['batch_size']
        self.height = self.cfg['height']
        self.width = self.cfg['width']
        self.loaded_frame_idxs = self.cfg['loaded_frame_idxs']
        self.num_scales = 1
        self.num_workers = self.cfg['num_workers']
        self.img_ext = self.cfg['img_ext']
        self.train_data_dir = self.cfg['train_data_dir']
        self.train_file_path = f"{self.log_path}/object_train_file/train/valid_object_files.txt"
        self.val_data_dir = self.cfg['val_data_dir']
        self.val_file_path = f"{self.log_path}/object_train_file/val/valid_object_files.txt"
        self.val_depth_data_dir = self.cfg['val_depth_data_dir']
        self.val_depth_file_path = self.cfg['val_depth_file_path']
        # Training setting
        self.reduce_lr_epochs_list = self.cfg['reduce_lr_epochs_list']
        self.nb_gpus = self.cfg['nb_gpus']
        self.epochs = self.cfg['object_depthnet_epochs']
        self.learning_rate = self.cfg['object_learning_rate']
        self.optim_beta = self.cfg['object_optim_beta']
        self.current_epoch = 0
        self.global_step = 0
        self.smallest_error = 10e10
        self.initialized_weights_folder = f"{self.project_path}/pixelwise_depthnet/models/weights"
        # Evaluation
        self.first_epoch_eval_steps = self.cfg['first_epoch_eval_steps']
        self.eval_steps = self.cfg['eval_steps']
        self.eval_and_save_every_epoch = self.cfg['eval_and_save_every_epoch']
        # Logging
        # self.log_path = self.cfg['log_path'] + "/object_depthnet"
        self.models_to_load = ['depth_net', 'pose_net']
        self.load_weights_folder = self.cfg['object_depthnet_load_weights_folder']
        # Define dataset 
        self.train_file = sorted(readlines(self.train_file_path))
        self.val_file = sorted(readlines(self.val_file_path))
        self.train_set = CityscapesPreprocessedDataset(data_path = self.train_data_dir,
                                                object_mask_path = f"{self.log_path}/object_mask_triplet/train",
                                                filenames = self.train_file,
                                                height = self.height,
                                                width = self.width,
                                                frame_idxs = self.loaded_frame_idxs,
                                                num_scales = self.num_scales,
                                                is_train=True,
                                                img_ext=self.img_ext,
                                                load_mask_triplet=True)
        self.val_set = CityscapesPreprocessedDataset(data_path = self.val_data_dir,
                                                object_mask_path = f"{self.log_path}/object_mask_triplet/val",
                                                filenames = self.val_file,
                                                height = self.height,
                                                width = self.width,
                                                frame_idxs = self.loaded_frame_idxs,
                                                num_scales = self.num_scales,
                                                is_train=False,
                                                img_ext=self.img_ext,
                                                load_mask_triplet=True)
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

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!! Need to fix this later (currently this only works for Cityscapes dataset) !!!
        self.val_depth_data_dir = self.cfg['val_depth_data_dir'] + "/leftImg8bit/val"
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.val_depth_file_path = self.cfg['val_depth_file_path']
        self.val_gt_depth_path = self.cfg['val_gt_depth_path']
        self.val_gt_mask_path = self.cfg['val_gt_mask_path']
        self.val_depth_file = readlines(self.val_depth_file_path)
        self.test_dynamic_mask_set = CityscapeObjectTestDataset(file_names=self.val_depth_file, 
                                                            image_dir=self.val_depth_data_dir, 
                                                            mask_dir=self.val_gt_mask_path, 
                                                            mask_size_threshold=self.object_mask_size_theshold, 
                                                            height=self.height, width=self.width)
        self.test_dynamic_mask_loader = torch.utils.data.DataLoader(self.test_dynamic_mask_set, batch_size=1, shuffle=False)
        # Define depthnet
        if self.depth_net_type == 'corl2020':
            self.depth_net = CorlDepthNet({}) # We pass an empty parameters here => Use default setting of the CorlDepthnet
            if self.depth_encoder_pretrained and self.tf_imageNet_checkpoint_path is None: assert False, 'Please specify path to imagenet checkpoint for the CORLDepthNet'
            if self.depth_encoder_pretrained: load_weights_from_tfResNet_to_depthEncoder(self.tf_imageNet_checkpoint_path, self.depth_net.depth_encoder)
        elif self.depth_net_type == 'monodepth2':
            raise NotImplementedError
        elif self.depth_net_type == 'packnet':
            self.depth_net = PackNet01(dropout=None, version='1A')
        elif self.depth_net_type == 'diffnet':
            self.depth_encoder = DiffNetDepthEncoder.hrnet18(True)
            self.depth_encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
            self.depth_decoder = DiffNetHRDepthDecoder(self.depth_encoder.num_ch_enc, scales=list(range(self.cfg['num_scales'])))
            self.depth_net = DiffDepthNet(self.depth_encoder, self.depth_decoder)
        elif self.depth_net_type == 'brnet':
            self.depth_encoder = BrNetResnetEncoder(num_layers=18, pretrained=True)
            self.depth_decoder = BrNetDepthDecoder(self.depth_encoder.num_ch_enc, scales=list(range(self.cfg['num_scales'])))
            self.depth_net = BrDepthNet(self.depth_encoder, self.depth_decoder)
        # elif self.depth_net_type == 'manydepth':
        #     raise NotImplementedError
        # Define posenet
        if self.pose_net_type == 'corl2020':
            self.pose_net = CorlPoseNet(in_channels=6) 
        elif self.pose_net_type == 'packnet':
            self.pose_net = PackNetPoseResNet(version='18pt')
        elif self.pose_net_type in ['manydepth', "monodepth2", 'diffnet', 'brnet']:
            self.pose_encoder = ManyDepthResnetEncoder(num_layers=18, pretrained=True, num_input_images=2)
            self.pose_decoder = ManyPoseDecoder(num_ch_enc=self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
            self.pose_net = ManyPoseNet(self.pose_encoder, self.pose_decoder)
        # Bring models to cuda
        self.depth_net.to(self.device)
        self.depth_net = torch.nn.DataParallel(self.depth_net, device_ids=list(range(self.nb_gpus)))
        self.pose_net.to(self.device)
        self.pose_net = torch.nn.DataParallel(self.pose_net, device_ids=list(range(self.nb_gpus)))
        # Init optimizer
        self.parameters_to_train = list(self.depth_net.parameters()) + list(self.pose_net.parameters())
        self.optimizer = torch.optim.Adam(self.parameters_to_train, lr=self.learning_rate)
        # Define default grid
        self.grid = torch.squeeze(torch.stack(torch.meshgrid(torch.arange(0, end=self.height, dtype=torch.float),
                                        torch.arange( 0, end=self.width, dtype=torch.float),
                                        torch.tensor([1.0, ]))), dim=3)
        self.grid[[0,1]] = self.grid[[1,0]]
        self.grid = self.grid.type(torch.FloatTensor).to(device=self.device)
        # Load model
        if self.initialized_weights_folder is not None and self.load_weights_folder is None:
            load_model([self.depth_net, self.pose_net], self.models_to_load, self.optimizer, self.initialized_weights_folder)
        if self.load_weights_folder is not None:
            try: 
                self.current_epoch = int(self.load_weights_folder.split('_')[-1])
            except:
                self.current_epoch = 10000 # Set dummy for current_epoch when load the best model
            load_model([self.depth_net, self.pose_net], self.models_to_load, self.optimizer, self.load_weights_folder)
            self.current_epoch = self.current_epoch + 1
        
    def process_batch(self, inputs, epoch, get_prediction_only=False):
        # 0. Process data
        for key, ipt in inputs.items(): 
            if type(ipt) is torch.Tensor: inputs[key] = ipt.to(self.device)
        for frame_idx in self.loaded_frame_idxs:
            inputs[('color_aug',frame_idx,0)] = inputs[('color_aug',frame_idx,0)] * inputs[('object_mask',frame_idx,0)]
        outputs = {}
        # 1. Predict camera pose
        # if self.depth_net_type=='manydepth':
        #     raise NotImplementedError
        # else:
        cam_T_cam_dict = self.predict_pose(inputs)
        outputs.update(cam_T_cam_dict)
        # 2. Predict depth
        predicted_depth_dict = self.predict_depth(inputs, outputs)
        outputs.update(predicted_depth_dict)
        # 3. Get the warped image
        image_pred_dict = self.generate_images_pred(inputs, outputs)
        outputs.update(image_pred_dict)
        # 4. Compute photometric loss
        losses_dict, stationary_mask = self.compute_losses(inputs, outputs, epoch)
        outputs['losses'] = losses_dict
        outputs['stationary_mask'] = stationary_mask
        return inputs, outputs
    def predict_depth(self, inputs, outputs):
        predicted_depth_dict = {}
        if self.depth_net_type == 'corl2020':
            predicted_depth = self.depth_net(inputs[('color_aug',0,0)], self.global_step) # This is a list storing predicted at multi-scale. Each has shape of b 1 h w
        elif self.depth_net_type in ['packnet', 'monodepth2', 'diffnet', 'brnet']:
            predicted_depth = self.depth_net(inputs[('color_aug',0,0)])
        else: raise NotImplementedError
        # Format the predicted depth as a dictionary
        for s in range(self.num_scales):
            predicted_depth_dict[('depth',0,s)] = predicted_depth[s]
            predicted_depth_dict[('disp',0,s)] = 1/predicted_depth_dict[('depth',0,s)]
            predicted_depth_dict[('depth',0,s)] = inputs[('object_mask',0,0)] * torch.nn.functional.interpolate(predicted_depth_dict[('depth',0,s)], (192,512), mode='bilinear', align_corners=False)
            predicted_depth_dict[('disp',0,s)] = inputs[('object_mask',0,0)] * torch.nn.functional.interpolate(predicted_depth_dict[('disp',0,s)], (192,512), mode='bilinear', align_corners=False)
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
            # else:
            predicted_depth_s = outputs[('depth',0,s)] # b 1 h//2(**s) w//(2**s)
            # Resize depth and object motion from h/(2^s) w/(2^s) to h w
            predicted_depth_s = torch.nn.functional.interpolate(predicted_depth_s,  [self.height, self.width], mode="bilinear", align_corners=False) # b 1 h w
            # Backproject a pixel in reference frame to 3D space 
            xyz = back_project_depth(predicted_depth_s[:,0], K_inv[:,:3,:3], self.grid) # 16, 3, 192, 512
            uv_neg = project_3d(K[:,:3,:3], cam_T_cam_neg, xyz)
            uv_pos = project_3d(K[:,:3,:3], cam_T_cam_pos, xyz)
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
        losses_dict = {}
        for s in range(self.num_scales):
            # Compute depth smoothness loss
            disp = outputs[('disp',0,s)] 
            mean_disp = torch.mean(disp, dim=[1, 2, 3], keepdim=True)
            object_mask_s = torch.nn.functional.interpolate(inputs[('object_mask',0,0)], (192//(2**s),512//(2**s)), mode='nearest')
            # print(disp.shape)
            # print(object_mask_s.shape)
            depth_smoothness_loss = self.depth_smoothing_weight*object_disp_smoothing(disp/mean_disp, inputs[('object_mask',0,0)])
            depth_smoothness_loss = depth_smoothness_loss / (2 ** s) 
            # Compute photometric loss
            photometric_loss, stationary_mask_s = rgbd_consistency_loss(inputs[('color_aug',0,0)].repeat(2,1,1,1), 
                                                                    torch.concat([inputs[('color_aug',-1,0)], inputs[('color_aug',1,0)]],dim=0), 
                                                                    torch.concat([outputs[('warped_image',-1,s)], outputs[('warped_image',1,s)]],dim=0), 
                                                                    self.ssim_weight, self.rgb_consistency_weight, c1=self.ssim_c1, c2=self.ssim_c2)
            photometric_loss_mask_s = stationary_mask_s
            photometric_loss = self.photometric_error_weight * photometric_loss 
            photometric_loss = torch.sum(photometric_loss_mask_s*photometric_loss) / torch.sum(photometric_loss_mask_s+1e-10)

            if s == 0: stationary_mask = stationary_mask_s
            losses_dict_s = {f'depth_smooth_{s}': depth_smoothness_loss, f'photometric_loss_{s}': photometric_loss}
            losses_dict.update(losses_dict_s)
        # Compute motion sparsity loss (at the highest resolution only)
        losses_dict['final_photometric_loss'] = sum([losses_dict[f'photometric_loss_{s}'] for s in range(self.num_scales)]) / self.num_scales
        losses_dict['final_depth_smooth'] = sum([losses_dict[f'depth_smooth_{s}'] for s in range(self.num_scales)]) / self.num_scales
        return losses_dict, stationary_mask
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
                    neg_warped_frame = output_endpoint[('warped_image',-1,0)][j].detach().cpu() # 3 h w
                    pos_warped_frame = output_endpoint[('warped_image',1,0)][j].detach().cpu() # 3 h w

                    ref_disp_all = torch.concat([torch.nn.functional.interpolate(output_endpoint[('disp',0,s)][[j]].detach().cpu(), [self.height, self.width], mode="bilinear", align_corners=False) for s in range(self.num_scales)])
                    ref_disp_all_colmap = torch.from_numpy(colormap(ref_disp_all)) # 4 3 h w 
                    ref_disp_0_colmap = ref_disp_all_colmap[0]
                    neg_stack_img = torch.concat([ref_disp_0_colmap,
                                                ref_frame,
                                                neg_warped_frame,
                                                neg_src_frame,
                                                ], dim=1)
                    pos_stack_img = torch.concat([ref_disp_0_colmap,
                                                ref_frame,
                                                pos_warped_frame,
                                                pos_src_frame,
                                                ], dim=1)
                    disp_all_stack = torch.concat([disp for disp in ref_disp_all_colmap], dim=1)
                    writer.add_image(f"a_visualzaition_-1/{j}/{mode}", neg_stack_img, self.global_step)
                    writer.add_image(f"b_visualzaition_1/{j}/{mode}", pos_stack_img, self.global_step)
                    writer.add_image(f"c_disp_multiscale/{j}/{mode}", disp_all_stack, self.global_step)
                    writer.add_image(f"f_stationary_mask/{j}/{mode}", output_endpoint['stationary_mask'][[j]].detach().cpu(), self.global_step)
    def get_depth_for_evaluation(self, use_masked_image=True):
        object_predicted_disp_list = []
        object_predicted_frame_id_list = []
        self.depth_net.eval()
        with torch.no_grad():
            for samples in tqdm(self.test_dynamic_mask_loader):
                image, mask_list, frame_id = samples
                image = image.float().cuda()
                mask_list = mask_list.float().cuda()

                if torch.sum(mask_list)==0: continue
                if use_masked_image: input = image * mask_list[0]
                else: input = image
                predicted_depth = self.depth_net(input)[0] # 1 1 h w
                predicted_disp = 1/predicted_depth
                object_predicted_disp = mask_list[0] * predicted_disp 
                object_predicted_disp_list += [disp.detach().cpu().numpy() for disp in object_predicted_disp]
                object_predicted_frame_id_list += [frame_id[0]] * len(object_predicted_disp)
        self.depth_net.train()
        return object_predicted_disp_list, object_predicted_frame_id_list
    def val(self, epoch):
        self.set_eval()
        try: val_inputs = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_loader)
            val_inputs = next(self.val_iter)
        with torch.no_grad(): val_inputs, val_outputs = self.process_batch(val_inputs, epoch)
        self.set_train()
        return val_inputs, val_outputs
    def epoch_eval(self, epoch):
        self.set_eval()
        epoch_val_running_loss = 0
        with torch.no_grad():
            for val_inputs in tqdm(self.val_loader):
                val_inputs, val_outputs = self.process_batch(val_inputs, epoch)
                val_total_loss = val_outputs['losses']['final_photometric_loss'] + val_outputs['losses']['final_depth_smooth'] 
                epoch_val_running_loss += val_total_loss.item()*self.batch_size
        self.set_train()
        return epoch_val_running_loss
    def set_train(self):
        self.depth_net.train()
        self.pose_net.train()
    def set_eval(self):
        self.depth_net.eval()
        self.pose_net.eval()
    def epoch_learning_curve_log(self, writers, epoch_train_loss, epoch_val_loss, errors, epoch):
        if epoch_train_loss is not None: writers['train'].add_scalar("a_epoch_eval/_loss", epoch_train_loss, epoch)
        if epoch_val_loss is not None: writers['val'].add_scalar("a_epoch_eval/_loss", epoch_val_loss, epoch)
        
        all_region_mean_errors = errors
        
        all_region_abs_rel, all_region_sq_rel, all_region_rmse, all_region_rmse_log, all_region_a1, all_region_a2, all_region_a3  = all_region_mean_errors
        writers['val'].add_scalar("object_depth/abs_rel", all_region_abs_rel, epoch)
        writers['val'].add_scalar("object_depth/sq_rel", all_region_sq_rel, epoch)
        writers['val'].add_scalar("object_depth/rmse", all_region_rmse, epoch)
        writers['val'].add_scalar("object_depth/rmse_log", all_region_rmse_log, epoch)
        writers['val'].add_scalar("object_depth/a1", all_region_a1, epoch)
        writers['val'].add_scalar("object_depth/a2", all_region_a2, epoch)
        writers['val'].add_scalar("object_depth/a3", all_region_a3, epoch)
    def train(self):
        # Evaluate depth using the pre-trained pixel-wise model
        if self.current_epoch == 0:
            object_predicted_disp_list, object_predicted_frame_id_list = self.get_depth_for_evaluation(use_masked_image=False)
            object_depth_errors = compute_object_errors_all_dataset(object_predicted_frame_id_list, object_predicted_disp_list, self.test_dynamic_mask_loader, self.val_gt_depth_path)
            self.epoch_learning_curve_log(self.writers, None, None, object_depth_errors, -1)
        # Main training loop
        global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {global_step}")
        print()
        for epoch in range(self.current_epoch, self.epochs):
            print(f"*********************************")
            print(f"************ Epoch {epoch} ************")
            print(f"*********************************")
            if epoch in self.reduce_lr_epochs_list:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
                print("Change learning rate to ", g['lr'])

            epoch_train_running_loss = 0
            for i, inputs in enumerate(tqdm(self.train_loader)):
                # Process batch
                inputs, outputs = self.process_batch(inputs, epoch)
                # Back propagation
                total_loss = outputs['losses']['final_photometric_loss'] + outputs['losses']['final_depth_smooth']
                self.optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(raft.parameters(),10)
                torch.nn.utils.clip_grad_norm_(self.depth_net.parameters(),10)
                torch.nn.utils.clip_grad_norm_(self.pose_net.parameters(),10)
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
                # # Epoch evaluation
                epoch_val_running_loss = self.epoch_eval(epoch)
                # Epoch depth evaluation
                object_predicted_disp_list, object_predicted_frame_id_list = self.get_depth_for_evaluation(use_masked_image=True)
                object_depth_errors = compute_object_errors_all_dataset(object_predicted_frame_id_list, object_predicted_disp_list, self.test_dynamic_mask_loader, self.val_gt_depth_path)
                # Epoch logging
                epoch_train_loss = epoch_train_running_loss / (len(self.train_loader)*self.batch_size)
                epoch_val_loss = epoch_val_running_loss / (len(self.val_loader)*self.batch_size)
                self.epoch_learning_curve_log(self.writers, epoch_train_loss, epoch_val_loss, object_depth_errors, epoch)
                # Save model
                save_model([self.depth_net,self.pose_net], ['depth_net','pose_net'], self.optimizer, self.log_path, epoch)
                if object_depth_errors[0] < self.smallest_error:
                    self.smallest_error = object_depth_errors[0]
                    save_model([self.depth_net,self.pose_net], ['depth_net','pose_net'], self.optimizer, self.log_path, None)
                    print(f"Best model detected. Save model at epoch {epoch} with abs_error = {self.smallest_error.item()}")
            