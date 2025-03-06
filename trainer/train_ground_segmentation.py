
import os 
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from dataset_auxiliary.ground_mask_dataset import GroundMaskDataset
from model.ground_segmentation import GroundSegmentationNet, GroundMaskExtractor
from loss import joint_bilateral_smoothing
from utils.checkpoints_utils import *
import imageio


def reduce_learning_rate(optimizer, reduction_factor=0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= reduction_factor
    print(f'Reduce learning rate from {param_group["lr"]*2} to {param_group["lr"]}')

def get_linear_weight(global_step, start_step, end_step, min_weights, max_weights):
    ratio = (global_step - start_step) / (end_step - start_step)
    ratio = np.clip(ratio, 0, 1)
    weight = min_weights + ratio * (max_weights - min_weights)
    return weight

def log(train_outputs, val_outputs, writers, global_step, prefix=''):

    outputs = {'train': train_outputs, 'val': val_outputs}
    for key in train_outputs:
        if 'loss' not in key: continue
        writers['train'].add_scalar(f'{prefix}_loss/{key}', train_outputs[key].item(), global_step)
        writers['val'].add_scalar(f'{prefix}_loss/{key}', val_outputs[key].item(), global_step)

    for mode in ['train', 'val']:
        images, road_mask, predicted_road_mask = outputs[mode]['images'], outputs[mode]['road_mask'], outputs[mode]['predicted_road_mask']

        for i in range(min(2, images.shape[0])): # going through each sample in the batch, up to 2 samples
            # Stack images as per the requirement
            img = images[i].detach().cpu() # 3 H W
            gt_mask = road_mask[i].repeat(3,1,1).detach().cpu() # 3 H W
            pred_mask = predicted_road_mask[i].repeat(3,1,1).squeeze().detach().cpu() # 3 H W
            pred_mask_threshold = (pred_mask > 0.5).float()

            img_gt_mask = img * gt_mask
            img_pred_mask = img * pred_mask
            img_pred_mask_threshold = img * pred_mask_threshold
            img_gt_mask_subtract = img * (1-gt_mask)
            img_pred_mask_threshold_subtract = img * (1-pred_mask_threshold)

            # Stack masks along the height dimension
            stacked_image = torch.cat([
                    torch.cat([img, img], dim=2),
                    torch.cat([gt_mask, pred_mask], dim=2),
                    torch.cat([img_gt_mask, img_pred_mask], dim=2),
                    torch.cat([img_gt_mask, img_pred_mask_threshold], dim=2),
                    torch.cat([img_gt_mask_subtract, img_pred_mask_threshold_subtract], dim=2)],
                    dim=1
            )
            writers[mode].add_image(f'{prefix}_stacked_image/{i}', stacked_image, global_step)


class GroundSegmentationTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_path = self.cfg['log_path']
        self.log_path = f"{cfg['log_path']}/ground_segmentation"
        self.depth_net_type = cfg['depth_net_type']

        # HARDCODE SOME PARAMS
        self.current_epoch = 0
        self.global_step = 0
        # Training params
        self.lr = cfg['gs_lr']
        self.use_depth = cfg['gs_use_depth']
        self.batch_size = cfg['gs_batch_size']
        self.eval_step = cfg['gs_eval_step']
        self.epochs = cfg['gs_epochs']
        self.reduce_lr_epoch = cfg['gs_reduce_lr_epoch']
        # Loss params
        self.suppressed_road_mask_loss_weight = cfg['gs_suppressed_road_mask_loss_weight']
        self.self_supervision_loss_weight = cfg['gs_self_supervision_loss_weight']
        self.smooth_loss_weight = cfg['gs_smooth_loss_weight']
        self.do_self_supervision = cfg['gs_do_self_supervision']
        self.start_self_supervision_step = cfg['gs_start_self_supervision_step']
        self.use_full_self_supervision_loss_step = cfg['gs_use_full_self_supervision_loss_step']
        self.start_supress_noisy_mask_signal_step = cfg['gs_start_supress_noisy_mask_signal_step']
        self.end_supress_noisy_mask_signal_step = cfg['gs_end_supress_noisy_mask_signal_step']
        # Dataloading
        self.num_workers = cfg['num_workers']
        self.device = cfg['device']
        # Dataset variables
        self.height = cfg['height']
        self.width = cfg['width']
        # Training dataset variables
        self.train_file_path = cfg['train_file_path']
        self.train_image_path = cfg['train_data_dir']
        self.train_depth_dir = f'{self.project_path}/pixelwise_depthnet/predictions/depth/train'
        # Validation dataset variables
        self.val_file_path = cfg['val_file_path']
        self.val_image_path = cfg['val_data_dir']
        self.val_depth_dir = f'{self.project_path}/pixelwise_depthnet/predictions/depth/val'

        # Loading model parameters
        self.load_weights_folder = cfg['ground_segmentation_load_weights_folder']

        # Define dataset
        self.train_files = np.sort(np.loadtxt(self.train_file_path, dtype=object, delimiter='\n'))
        self.train_set = GroundMaskDataset(image_dir=self.train_image_path, depth_dir=self.train_depth_dir, file_list=self.train_files, height=self.height, width=self.width, is_train=True)
        self.train_loader = DataLoader(self.train_set, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers, 
                                pin_memory=True, 
                                drop_last=True)
        self.val_files = np.sort(np.loadtxt(self.val_file_path, dtype=object, delimiter='\n'))
        self.val_set = GroundMaskDataset(image_dir=self.val_image_path, depth_dir=self.val_depth_dir, file_list=self.val_files, height=self.height, width=self.width, is_train=False)
        self.val_loader = DataLoader(self.val_set, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers, 
                                pin_memory=True, 
                                drop_last=True)
        self.val_iter = iter(self.val_loader)


        # Define ground mask extractor
        self.ground_mask_extractor = GroundMaskExtractor(self.batch_size, self.height, self.width)
        self.ground_mask_extractor.to(device=self.device)
        # Define ground_segmentation net 
        self.ground_segmentation_net = GroundSegmentationNet(use_depth=self.use_depth).cuda()
        self.optimizer = torch.optim.Adam(self.ground_segmentation_net.parameters(), lr=self.lr)

        # Load model
        if self.load_weights_folder is not None:
            self.current_epoch = load_model([self.ground_segmentation_net], ['ground_segmentation'], self.optimizer, self.load_weights_folder) + 1
        # Log set up 
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=10)

    def eval(self):
        try:
            val_samples = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)  # Reset the iterator
            val_samples = next(self.val_iter)
        frame_id_list = val_samples[-1]
        val_samples = val_samples[:-1]
        with torch.no_grad():
            val_outputs, val_self_supervision_ouputs = self.process_batch(val_samples, do_self_supervision=self.do_self_supervision)
        return val_outputs, val_self_supervision_ouputs
        
    def epoch_eval(self):
        # Evaluation
        epoch_val_loss = 0
        with torch.no_grad():
            for val_samples in tqdm(self.val_loader):
                frame_id_list = val_samples[-1]
                val_samples = val_samples[:-1]
                val_outputs, val_self_supervision_ouputs = self.process_batch(val_samples, do_self_supervision=False)
                val_total_loss = val_outputs['total_loss'] + val_self_supervision_ouputs['self_supervision_total_loss']
                epoch_val_loss += val_total_loss * len(val_outputs['images'])
        return epoch_val_loss

    def process_batch(self, samples, do_self_supervision=False):
        images, depths, K = [item.float().cuda() for item in samples]
        with torch.no_grad(): 
            # Get features for predicted_road_mask
            disp = 1/depths
            disp = disp/torch.max(disp.view(len(disp),-1),dim=-1).values.view(len(disp),1,1,1)
            features = torch.cat([images, disp], dim=1) if self.use_depth else images
            # Get target noisy road mask
            road_mask = self.ground_mask_extractor(depths, K)

        # Predict road masks and compute loss
        predicted_road_mask = self.ground_segmentation_net(features)
        mask_loss = torch.nn.functional.binary_cross_entropy(predicted_road_mask, road_mask)
        smooth_loss = joint_bilateral_smoothing(predicted_road_mask, features)*self.smooth_loss_weight
        total_loss = mask_loss + smooth_loss
        outputs = {'images': images, 'depths':depths, 'road_mask': road_mask, 'predicted_road_mask': predicted_road_mask, 
                    'mask_loss': mask_loss, 'smooth_loss': smooth_loss, 'total_loss': total_loss}
        
        # If global_step >= start_supress_noisy_mask_signal_step, we start reduce the weights of the loss on the noisy target masks
        # At the same time, we prioritize self supervision loss
        noisy_mask_signal_loss_weight = get_linear_weight(self.global_step, start_step=self.start_supress_noisy_mask_signal_step, end_step=self.end_supress_noisy_mask_signal_step,
                                            min_weights=1, max_weights=self.suppressed_road_mask_loss_weight)
        outputs['mask_loss'] = outputs['mask_loss'] = outputs['mask_loss']*noisy_mask_signal_loss_weight
        outputs['total_loss'] = outputs['mask_loss'] + outputs['smooth_loss']

        self_supervision_ouputs = {'self_supervision_total_loss': torch.tensor(0).float().cuda()} 

        # In do_self_supervision, 
        # On the other hand, the middle 2/3 are supervised by the model's own predictions:
            # We first devide the image into 3 along the height dimesion. Then we resize the middle and the bottom part to the original size
            # The model then predicts road mask for the bottom part and the middle part
            # Loss 1: The bottom 1/3 is still supervised using the noisy target ground mask
            # Loss 2 (self-supervision): 
                # We concatenate the middle and the bottom part together
                # Then use it as target to supervise the predictions on the original image
                # ==> In this way, the model learn to predict part of the ground that are very far away 
                #                               (which is usually not included in the target noisy mask)
        if do_self_supervision and self.global_step >= self.start_self_supervision_step:
            self_supervision_loss_linear_weight = get_linear_weight(self.global_step, start_step=self.start_self_supervision_step,
                                                                    end_step=self.use_full_self_supervision_loss_step, min_weights=0, 
                                                                    max_weights=self.self_supervision_loss_weight)

            # Process the bottom 1/3
            bottom_image = images[:,:,int(2/3*images.shape[2]):,:] # b 3 h/3 w
            bottom_road_mask = road_mask[:,:,int(2/3*images.shape[2]):,:] # b 3 h/3 w
            bottom_depth = depths[:,:,int(2/3*images.shape[2]):,:] # b 1 h/3 w
            # Resize stuff
            bottom_image = torch.nn.functional.interpolate(bottom_image, (192,512), mode='bilinear', align_corners=False)
            bottom_road_mask = torch.nn.functional.interpolate(bottom_road_mask, (192,512), mode='bilinear', align_corners=False)
            bottom_depth = torch.nn.functional.interpolate(bottom_depth, (192,512), mode='bilinear', align_corners=False)
            # Predict bottom mask
            bottom_disp = 1/bottom_depth
            bottom_disp = bottom_disp/torch.max(bottom_disp.view(len(bottom_disp),-1),dim=-1).values.view(len(bottom_disp),1,1,1)
            bottom_features = torch.cat([bottom_image, bottom_disp], dim=1) if self.use_depth else images
            bottom_predicted_road_mask = self.ground_segmentation_net(bottom_features)
            # Loss 1 mentioned above. We have bce loss and smoothness loss
            # Calculate loss
            bottom_mask_loss = torch.nn.functional.binary_cross_entropy(bottom_predicted_road_mask, bottom_road_mask)*self_supervision_loss_linear_weight
            bottom_smooth_loss = joint_bilateral_smoothing(bottom_predicted_road_mask, bottom_features)*self.smooth_loss_weight
            bottom_total_loss = bottom_mask_loss + bottom_smooth_loss

            with torch.no_grad():
                middle_image = images[:,:,int(1/3*images.shape[2]):int(2/3*images.shape[2]),:] # b 3 h/3 w
                middle_road_mask = road_mask[:,:,int(1/3*images.shape[2]):int(2/3*images.shape[2]),:] # b 3 h/3 w
                middle_depth = depths[:,:,int(1/3*images.shape[2]):int(2/3*images.shape[2]),:] # b 1 h/3 w
                # Resize stuff
                middle_image = torch.nn.functional.interpolate(middle_image, (192,512), mode='bilinear', align_corners=False)
                middle_road_mask = torch.nn.functional.interpolate(middle_road_mask, (192,512), mode='bilinear', align_corners=False)
                middle_depth = torch.nn.functional.interpolate(middle_depth, (192,512), mode='bilinear', align_corners=False)
                # Predict middle mask
                middle_disp = 1/middle_depth
                middle_disp = middle_disp/torch.max(middle_disp.view(len(middle_disp),-1),dim=-1).values.view(len(middle_disp),1,1,1)
                middle_features = torch.cat([middle_image, middle_disp], dim=1) if self.use_depth else images
                middle_predicted_road_mask = self.ground_segmentation_net(middle_features)
                # middle_smooth_loss = joint_bilateral_smoothing(middle_predicted_road_mask, middle_features)*self.smooth_loss_weight

            # Resize it back
            bottom_two_third_road_mask = torch.cat([middle_predicted_road_mask.detach(), bottom_predicted_road_mask.detach()], dim=2)
            bottom_two_third_road_mask = torch.nn.functional.interpolate(bottom_two_third_road_mask, 
                                                                        (int(2/3*192),512), 
                                                                        mode='bilinear', align_corners=False)
            bottom_two_third_road_mask = (bottom_two_third_road_mask>0.5).float()
            # Loss 2 mentioned above. We have only have bce loss here
            self_supervision_loss = torch.nn.functional.binary_cross_entropy(predicted_road_mask[:,:,(int(1/3*192)):,:], bottom_two_third_road_mask.detach())*self_supervision_loss_linear_weight

            self_supervision_total_loss = bottom_total_loss + self_supervision_loss
            self_supervision_ouputs = {
                'bottom_mask_loss': bottom_mask_loss,
                'bottom_smooth_loss': bottom_smooth_loss,
                'self_supervision_loss': self_supervision_loss,
                'self_supervision_total_loss': self_supervision_total_loss,
                'images': images[:,:,(int(1/3*192)):,:],
                'road_mask': bottom_two_third_road_mask,
                'predicted_road_mask': predicted_road_mask[:,:,(int(1/3*192)):,:]
            }
        return outputs, self_supervision_ouputs



    def get_prediction(self):
            
        # return predicted_road_mask
        train_set = GroundMaskDataset(image_dir=self.train_image_path, depth_dir=self.train_depth_dir, file_list=self.train_files, height=self.height, width=self.width, is_train=False)
        train_loader = DataLoader(train_set, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=self.num_workers, 
                                pin_memory=True, 
                                drop_last=False)
        val_set = GroundMaskDataset(image_dir=self.val_image_path, depth_dir=self.val_depth_dir, file_list=self.val_files, height=self.height, width=self.width, is_train=False)
        val_loader = DataLoader(val_set, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=self.num_workers, 
                                pin_memory=True, 
                                drop_last=False)
        # Start getting prediction
        dataset_dict = {'train': train_loader, 'val': val_loader} 
        self.ground_segmentation_net.eval()
        with torch.no_grad():
            for split in dataset_dict.keys():
                print('Get prediction in split: ', split)
                data_loader = dataset_dict[split]
                for samples in tqdm(data_loader):
                    frame_id_list = samples[-1]
                    samples = samples[:-1]
                    images, depths, K = [item.float().cuda() for item in samples]
                    # Get features for predicted_road_mask
                    disp = 1/depths
                    disp = disp/torch.max(disp.view(len(disp),-1),dim=-1).values.view(len(disp),1,1,1)
                    features = torch.cat([images, disp], dim=1) if self.use_depth else images
                    # Predict road masks and compute loss
                    predicted_road_mask = self.ground_segmentation_net(features)
                    predicted_road_mask = predicted_road_mask.detach().cpu()[:,0]
                    # Save each predicted mask
                    for i, frame_id in enumerate(frame_id_list):
                        city = frame_id.split('_') [0]
                        frame = frame_id
                        directory = f'{self.log_path}/predictions/{split}/{city}'
                        os.makedirs(directory, exist_ok=True) 
                        save_file_path = os.path.join(directory, f"{frame}.png")
                        imageio.imsave(save_file_path, predicted_road_mask[i])

    def train(self):
        self.global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {self.global_step}")
        print()

        for epoch in range(self.current_epoch, self.epochs):
            print(f"---------------- Epoch {epoch} ----------------")
            if epoch in self.reduce_lr_epoch: reduce_learning_rate(self.optimizer, reduction_factor=0.5)
            epoch_train_loss = 0
            for samples in tqdm(self.train_loader):
                frame_id_list = samples[-1]
                samples = samples[:-1]
                train_outputs, train_self_supervision_ouputs = self.process_batch(samples, do_self_supervision=self.do_self_supervision)
                total_loss = train_outputs['total_loss'] + train_self_supervision_ouputs['self_supervision_total_loss']
                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                epoch_train_loss += total_loss.item() * len(train_outputs['images'])

                # Log
                if self.global_step % self.eval_step == 0:
                    val_outputs, val_self_supervision_ouputs = self.eval()
                    log(train_outputs, val_outputs, self.writers, self.global_step, prefix='')
                    if train_self_supervision_ouputs['self_supervision_total_loss'] != 0:
                        log(train_self_supervision_ouputs, val_self_supervision_ouputs, self.writers, self.global_step, prefix='self_supervision')
                self.global_step += 1
            # Epoch eval
            epoch_val_loss = self.epoch_eval()
            # Epoch logging
            epoch_train_loss = epoch_train_loss / (self.batch_size*len(self.train_loader))
            epoch_val_loss = epoch_val_loss / (self.batch_size*len(self.val_loader))
            self.writers['train'].add_scalar('epoch_loss', epoch_train_loss, epoch)
            self.writers['val'].add_scalar('epoch_loss', epoch_val_loss, epoch)
            # Logging
            save_model([self.ground_segmentation_net], ['ground_segmentation'], self.optimizer, self.log_path, epoch)

