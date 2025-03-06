
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import torch 
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
import pickle as pkl
import yaml
import os
from tqdm.auto import  tqdm
from utils.mask_utils import  extract_valid_dynamic_object_mask, extract_dynamic_object_filenames
from dataset_auxiliary.object_scale_dataset import ObjectScaleDataset, CityScapesObjectScaleEvalDataset
from model_zoo.model_zoo import *
from model.depth_scale_alignment_net import DepthScaleAlignmentNet
from tensorboardX import SummaryWriter
from utils.utils import  colormap
from utils.validation_utils import compute_errors_all_dataset
from utils.checkpoints_utils import *


def eval_dataset_collate(batch):
    images = [item[0] for item in batch]
    depth = [item[1] for item in batch]
    object_masks = [item[2] for item in batch]
    object_bbox = [item[3] for item in batch]
    road_mask = [item[4] for item in batch]
    dynamic_mask = [item[5] for item in batch]
    frame_id = [item[6] for item in batch]
    images = torch.stack(images)
    depth = torch.stack(depth)
    road_mask = torch.stack(road_mask)
    dynamic_mask = torch.stack(dynamic_mask)
    return images, depth, object_masks, object_bbox, road_mask, dynamic_mask, frame_id

class DepthScaleAlignmentTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_path = self.cfg['log_path']

        self.depth_net_type = cfg['depth_net_type']

        flow_threshold = 1.0
        # Training dataset variables
        self.train_image_path = cfg['train_data_dir']
        self.train_depth_path = f'{self.project_path}/pixelwise_depthnet/predictions/depth/train'
        self.train_object_mask_path = f'{self.project_path}/object_depthnet/processed_mask/train'
        self.train_road_mask_path = f'{cfg["raw_road_mask_path"]}/train'
        self.train_object_predicted_flow_neg_path = f'{self.project_path}/pixelwise_depthnet/predictions/flow_neg/train'
        self.train_object_predicted_flow_pos_path = f'{self.project_path}/pixelwise_depthnet/predictions/flow_pos/train'
        self.train_file_lists_path = f'{self.project_path}/object_depthnet/object_train_file/train/object_files.txt'
        # Validation dataset variables
        self.val_image_path = cfg['val_data_dir']
        self.val_depth_path = f'{self.project_path}/pixelwise_depthnet/predictions/depth/val'
        self.val_object_mask_path = f'{self.project_path}/object_depthnet/processed_mask/val'
        self.val_road_mask_path = f'{cfg["raw_road_mask_path"]}/val'
        self.val_object_predicted_flow_neg_path = f'{self.project_path}/pixelwise_depthnet/predictions/flow_neg/val'
        self.val_object_predicted_flow_pos_path = f'{self.project_path}/pixelwise_depthnet/predictions/flow_pos/val'
        self.val_file_lists_path = f'{self.project_path}/object_depthnet/object_train_file/val/object_files.txt'
        # Depth evaluation
        self.gt_depth_path = self.cfg['val_gt_depth_path']
        self.gt_mask_path = self.cfg['val_gt_mask_path']

        self.object_depth_net_load_path = f'{self.project_path}/object_depthnet/models/weights/depth_net.pth'
        self.load_weights_folder = cfg['dsa_load_path']
        self.models_to_load = ['depth_scale_alignment']

        # Model params
        self.num_scales = 1
        # Training variables
        self.current_epoch = 0
        self.epochs = cfg['dsa_epochs']
        self.batch_size = cfg['dsa_batch_size']
        self.lr = cfg['dsa_lr']
        self.reduce_lr_epoch = cfg['dsa_reduce_lr_epoch']
        self.eval_steps = 1000
        self.huber_delta =  0.005
        self.global_step = 0
        self.height = cfg['height']
        self.width = cfg['width']
        self.smallest_error = 10e10
        self.nb_gpus = cfg['nb_gpus']

        # Log_path
        self.log_path = self.cfg['log_path'] + "/depth_scale_alignment"
        self.num_workers = cfg['num_workers']

        # Set up dataset
        self.train_file_lists = np.loadtxt(self.train_file_lists_path, dtype=object, delimiter='\n')
        self.train_file_lists = [file for file in self.train_file_lists if file.split(' ')[-1]=='static']
        self.train_set = ObjectScaleDataset(self.train_image_path, self.train_depth_path, self.train_object_mask_path, self.train_road_mask_path, 
                        self.train_object_predicted_flow_neg_path, self.train_object_predicted_flow_pos_path, flow_threshold, self.height, self.width, self.train_file_lists, is_train=True)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

        self.val_file_lists = np.loadtxt(self.val_file_lists_path, dtype=object, delimiter='\n')
        self.val_file_lists = [file for file in self.val_file_lists if file.split(' ')[-1]=='static']
        self.val_set = ObjectScaleDataset(self.val_image_path, self.val_depth_path, self.val_object_mask_path, self.val_road_mask_path, 
                        self.val_object_predicted_flow_neg_path, self.val_object_predicted_flow_pos_path, flow_threshold, self.height, self.width, self.val_file_lists, is_train=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)

        # Define depthnet
        if self.depth_net_type == 'corl2020':
            self.object_depth_net = CorlDepthNet({}) # We pass an empty parameters here => Use default setting of the CorlDepthnet
            if self.depth_encoder_pretrained and self.tf_imageNet_checkpoint_path is None: assert False, 'Please specify path to imagenet checkpoint for the CORLDepthNet'
            if self.depth_encoder_pretrained: load_weights_from_tfResNet_to_depthEncoder(self.tf_imageNet_checkpoint_path, depth_net.depth_encoder)
        elif self.depth_net_type == 'monodepth2':
            raise NotImplementedError
        elif self.depth_net_type == 'packnet':
            self.object_depth_net = PackNet01(dropout=None, version='1A')
        elif self.depth_net_type == 'diffnet':
            self.depth_encoder = DiffNetDepthEncoder.hrnet18(True)
            self.depth_encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
            self.depth_decoder = DiffNetHRDepthDecoder(self.depth_encoder.num_ch_enc, scales=list(range(self.num_scales)))
            self.object_depth_net = DiffDepthNet(self.depth_encoder, self.depth_decoder)
        elif self.depth_net_type == 'brnet':
            self.depth_encoder = BrNetResnetEncoder(num_layers=18, pretrained=True)
            self.depth_decoder = BrNetDepthDecoder(self.depth_encoder.num_ch_enc, scales=list(range(self.cfg['num_scales'])))
            self.object_depth_net = BrDepthNet(self.depth_encoder, self.depth_decoder)
        # elif self.depth_net_type == 'manydepth':
        #     raise NotImplementedError
        self.object_depth_net.to(device='cuda')
        self.object_depth_net = torch.nn.DataParallel(self.object_depth_net, device_ids=list(range(self.nb_gpus)))
        self.object_depth_net.load_state_dict(torch.load(self.object_depth_net_load_path))
        self.object_depth_net.eval()
        print(f"Load object_depthnet successfully at {self.object_depth_net_load_path}")

        # Define
        self.depth_scale_alignment = DepthScaleAlignmentNet(replace_bn_by_gn=True)
        self.depth_scale_alignment = self.depth_scale_alignment.to(device='cuda')
        self.optimizer = torch.optim.Adam(self.depth_scale_alignment.parameters(), lr=self.lr)

        # # LOAD MODEL AND EPOCH IF THERE IS LOAD_WEIGHTS_FOLDER
        if self.load_weights_folder is not None:
            # load_epoch = load_model(self.depth_scale_alignment, self.optimizer, self.load_weights_folder)
            load_epoch = load_model([self.depth_scale_alignment], self.models_to_load, self.optimizer, self.load_weights_folder)
            self.current_epoch = load_epoch + 1

        # Log set up 
        if not os.path.isdir(self.log_path): os.makedirs(self.log_path)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=1)

    def extract_eval_dynamic_object_mask(self, split):
        # Define some variables
        depth_net_type = self.cfg['depth_net_type']
        with open(f"{self.project_path}/object_depthnet/dynamic_static_labels/{split}/dynamic_label_dict.pkl", 'rb' ) as f:
            static_dynamic_label_dict = pkl.load(f)
        mask_path = f'{self.project_path}/object_depthnet/processed_mask/{split}'
        road_mask_path = f'{self.cfg["raw_road_mask_path"]}/{split}'
        save_dir = f'./{self.log_path}/dynamic_object_training_files/{split}'
        os.makedirs(save_dir, exist_ok=True)
        object_mask_size_theshold = self.cfg['object_mask_size_theshold']
        # Retrieve file name of dynamic object (used for validation)
        extract_mask_file_id_list = extract_dynamic_object_filenames(static_dynamic_label_dict, depth_net_type, mask_path, road_mask_path, object_mask_size_theshold, split, save_dir)
        np.savetxt(f'{save_dir}/dynamic_object_filenames.txt', extract_mask_file_id_list, fmt='%s', delimiter='\n')

    def create_evaluation_dataset(self):
        height = self.height
        width = self.width
        test_dynamic_file_lists_path = f"{self.log_path}/dynamic_object_training_files/val/dynamic_object_filenames.txt"
        test_file_lists_path = self.cfg["val_depth_file_path"]
        test_image_path =  self.cfg["val_data_dir"]
        test_depth_path = f"{self.project_path}/pixelwise_depthnet/predictions/depth/val"
        test_object_mask_path = f"{self.project_path}/object_depthnet/processed_mask/val"
        test_road_mask_path = f"{self.cfg['raw_road_mask_path']}/val"
        test_object_predicted_flow_neg_path = f"{self.project_path}/pixelwise_depthnet/predictions/flow_neg/val"
        test_object_predicted_flow_pos_path = f"{self.project_path}/pixelwise_depthnet/predictions/flow_pos/val"

        test_dynamic_file_lists = np.loadtxt(test_dynamic_file_lists_path, dtype=object, delimiter='\n')
        test_file_lists = np.loadtxt(test_file_lists_path, dtype=object, delimiter='\n')
        # Filter out only filenames that is in test_file_lists
        test_dynamic_file_lists_ = [file_name.split(' ')[0] + ' ' + file_name.split(' ')[1] for file_name in test_dynamic_file_lists]
        test_file_lists_ = []
        for file_name in test_file_lists:
            city_id, frame_id = file_name.split(" ")
            if not os.path.isfile(f'{test_depth_path}/{city_id}/{frame_id}.npy'):
                continue
            if file_name in test_dynamic_file_lists_:
                dynamic_file_name = test_dynamic_file_lists[test_dynamic_file_lists_.index(file_name)]
            else:
                dynamic_file_name = file_name + ' -1'
            test_file_lists_.append(dynamic_file_name)
        test_file_lists = np.array(test_file_lists_)
        # np.array([test_dynamic_file_lists[test_dynamic_file_lists_.index(file_name)] if file_name in test_dynamic_file_lists_ else file_name + ' -1'  for file_name in test_file_lists])
        test_set = CityScapesObjectScaleEvalDataset(test_image_path, test_depth_path, test_object_mask_path, test_road_mask_path, 
                        test_object_predicted_flow_neg_path, test_object_predicted_flow_pos_path, 0.5, height, width, test_file_lists, middle_portion=20, is_train=False)
        test_loader = DataLoader(test_set, batch_size=self.cfg['dsa_batch_size'], num_workers=self.cfg['num_workers'], shuffle=False, pin_memory=True, drop_last=False, collate_fn=eval_dataset_collate)
        self.depth_test_file_lists, self.depth_test_set, self.depth_test_loader = test_file_lists, test_set, test_loader

    def log(self, train_outputs, val_outputs, global_step):
        outputs = {}
        outputs["train"] = train_outputs
        outputs["val"] = val_outputs
        with torch.no_grad():
            for mode in ["train", "val"]:
                # Show loss 
                self.writers[mode].add_scalar("loss", outputs[mode]['loss'], global_step) 
                images = outputs[mode]['images'].detach().cpu() # b 3 h w
                static_depth = outputs[mode]['static_depth'].detach().cpu()
                masked_static_depth = outputs[mode]['masked_static_depth'].detach().cpu() # b 1 h w
                dynamic_depth = outputs[mode]['dynamic_depth'].detach().cpu() # b 1 h w
                masked_dynamic_depth = outputs[mode]['masked_dynamic_depth'].detach().cpu() # b 1 h w
                predicted_scale = outputs[mode]['predicted_scale'].detach().cpu().view(-1) # b 1
                object_masks = outputs[mode]['object_masks'].detach().cpu() # b 1

                # Show depth maps: static, masked static, masked depth, combined depth, scaled combined depth
                for j in range(2):  # write a maxmimum of 4 images per mode
                    # LOG IMAGES
                    # Get static disp and masked static disp
                    static_disp = torch.from_numpy(colormap(1/static_depth[j]))[0] # 3 h w
                    masked_static_disp = static_disp*((masked_static_depth[j]!=0).float())
                    # Get masked dynamic depth
                    dynamic_disp = torch.from_numpy(colormap(1/(dynamic_depth[j]+1e-5)))[0] # 3 h w
                    masked_dynamic_disp = dynamic_disp*((masked_dynamic_depth[j]!=0).float())
                    # Get the combined disp image
                    combined_disp =  1 / ((1-object_masks[j])*static_depth[j] + object_masks[j]*masked_dynamic_depth[j]) # 1 h w
                    combined_disp = torch.from_numpy(colormap(combined_disp))[0] # 3 h w
                    combined_disp = combined_disp*(masked_static_depth[j]!=0) + combined_disp*(masked_dynamic_depth[j]!=0)
                    # Get the scaled combined disp image
                    scaled_combined_disp =  1 / ((1-object_masks[j])*static_depth[j] + object_masks[j]*masked_dynamic_depth[j] * predicted_scale[j]) # 1 h w
                    scaled_combined_disp = torch.from_numpy(colormap(scaled_combined_disp))[0] # 3 h w
                    scaled_combined_disp = scaled_combined_disp*(masked_static_depth[j]!=0) + scaled_combined_disp*(masked_dynamic_depth[j]!=0)
                    # Get the stacked image
                    image_stack = torch.concat([images[j], static_disp, masked_static_disp, masked_dynamic_disp, combined_disp, scaled_combined_disp], dim=1) # 3 h w*6
                    # Write image
                    self.writers[mode].add_image(f"image_stack/{j}/{mode}", image_stack, global_step)
                    
                    # LOG DISTRIBUTION
                    masked_object_static_depth = static_depth[j][masked_dynamic_depth[j]!=0]
                    masked_object_dynamic_depth = masked_dynamic_depth[j][masked_dynamic_depth[j]!=0]
                    ratio = (masked_object_static_depth / masked_object_dynamic_depth)
                    self.writers[mode].add_histogram(f"scale_distribution/{j}/{mode}", ratio, global_step, bins='auto')

                    ratio = ratio.numpy()
                    figure = plt.figure(figsize=(5,5))
                    plt.hist(ratio, bins='auto')
                    plt.title(f"Med={np.round(np.median(ratio*1000))/1000}\nPred={np.around(predicted_scale[j].item(),3)}")
                    plt.vlines(x=np.round(np.median(ratio*1000))/1000, ymin=0, ymax=plt.ylim()[1], colors='red', label="Median scale")
                    plt.vlines(x=np.around(predicted_scale[j].item(),3), ymin=0, ymax=plt.ylim()[1], colors='green', label="Pred scale")
                    plt.legend()
                    # plt.title('Title', fontsize=20)
                    self.writers[mode].add_figure(f"scale_distribution_fig/{j}/{mode}", figure, global_step)
        
    def process_batch(self, samples):
        # images, object_masks, dynamic_masks, object_bbox_mask = [item.cuda() for item in samples]
        full_frame_id = samples[-1]
        samples = samples[:-1]
        images, static_depth, object_masks, object_bbox_mask, road_masks, dynamic_masks  = [item.float().cuda() for item in samples]
        # Step 1: Get static and dynamic_depth_predictions
        with torch.no_grad():
            # Get road masks
            object_masks = object_masks * (1-road_masks)
            road_masks = road_masks * (1-object_bbox_mask)
            # Use road mask to mask out static depth 
            # masked_static_depth = road_masks*(1-dynamic_masks)*(1-object_masks)*static_depth
            masked_static_depth = road_masks*(1-object_masks)*static_depth
            # Get prediction for dynamic regions
            dynamic_depth = self.object_depth_net(images*object_masks)[0]
            masked_dynamic_depth = dynamic_depth*object_masks # b 1 h w
            masked_dynamic_depth_mean = torch.sum(masked_dynamic_depth, dim=[1,2,3]) / (torch.sum(object_masks, dim=[1,2,3])  + 1e-5)
            masked_dynamic_depth = masked_dynamic_depth / (masked_dynamic_depth_mean.view(len(masked_dynamic_depth),1,1,1)+1e-5)
        # Predict depth scale
        predicted_scale = self.depth_scale_alignment(images, object_masks, masked_static_depth, masked_dynamic_depth) # b 1
        # Calculate loss
        scaled_dynamic_depth = predicted_scale.unsqueeze(-1).unsqueeze(-1) * masked_dynamic_depth
        normalized_scaled_dynamic_depth_  = scaled_dynamic_depth # /(scaled_dynamic_depth+static_depth+1e-5)
        normalized_static_depth_depth_  = static_depth # /(static_depth+static_depth+1e-5)
        depth_error = torch.nn.HuberLoss(reduction='none', delta=self.huber_delta)(normalized_scaled_dynamic_depth_, normalized_static_depth_depth_)
        loss = torch.sum(depth_error*object_masks, dim=[1,2,3], keepdim=True) / (torch.sum(object_masks, dim=[1,2,3], keepdim=True) + 1e-5)
        loss = torch.sum(loss) / len(loss!=0)
        # Return outputs
        outputs = {}
        outputs['images'] = images
        outputs['object_masks'] = object_masks
        outputs['dynamic_masks'] = dynamic_masks
        outputs['static_depth'] = static_depth
        outputs['masked_static_depth'] = masked_static_depth
        outputs['dynamic_depth'] = dynamic_depth
        outputs['masked_dynamic_depth'] = masked_dynamic_depth
        outputs['predicted_scale'] = predicted_scale
        outputs['scaled_dynamic_depth'] = scaled_dynamic_depth
        outputs['loss'] = loss
        if torch.isnan(loss): assert False, 'Nan loss detected'
        return outputs

    def epoch_eval(self):
        self.depth_scale_alignment.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_samples in tqdm(self.val_loader):
                val_outputs = self.process_batch(val_samples) # no augmentaiton for validation
                val_loss = val_outputs['loss']
                epoch_val_loss += val_loss.item() * len(val_outputs['images'])
        self.depth_scale_alignment.train()
        return epoch_val_loss

    def get_depth_for_evaluation(self, use_scaled_depth=True):
        scaled_disp_list = []
        with torch.no_grad():
            for s,samples in enumerate(tqdm(self.depth_test_loader)):
                full_frame_id_list = samples[-1]
                samples = samples[:-1]
                images, static_depth_list, dynamic_object_mask_list, dynamic_object_bbox_list, road_masks_list, dynamic_region_mask_list = samples
                # Get prediction for full images (using StaticDepthNet)
                images = images.float().cuda()
                static_depth_list = static_depth_list.float().cuda()
                # static_depth_list = static_depth_list_[[s]]
                for b  in range(len(static_depth_list)): 
                    # Process input
                    dynamic_object_masks = dynamic_object_mask_list[b].float().cuda() # N 1 192 512
                    dynamic_object_bbox = dynamic_object_bbox_list[b].float().cuda() # N 1 192 512
                    road_masks = road_masks_list[[b]].repeat(len(dynamic_object_masks),1,1,1).float().cuda() # N 1 192 512
                    dynamic_object_masks = dynamic_object_masks * (1-road_masks)
                    road_masks = road_masks * (1-dynamic_object_bbox)
                    # Get masked depth for static regions
                    # Use road mask to mask out static depth 
                    masked_static_depth = road_masks*(1-dynamic_object_masks)*static_depth_list[[b]]
                    # Get prediction for dynamic regions
                    dynamic_depth = self.object_depth_net(images[[b]]*dynamic_object_masks)[0]
                    masked_dynamic_depth = dynamic_depth*dynamic_object_masks # b 1 h w
                    masked_dynamic_depth_mean = torch.sum(masked_dynamic_depth, dim=[1,2,3]) / (torch.sum(dynamic_object_masks, dim=[1,2,3])  + 1e-5)
                    masked_dynamic_depth = masked_dynamic_depth / (masked_dynamic_depth_mean.view(len(masked_dynamic_depth),1,1,1)+1e-5)
                    # Predict depth scale 
                    predicted_scale = self.depth_scale_alignment(images[[b]].repeat(len(dynamic_object_masks),1,1,1), dynamic_object_masks, masked_static_depth, masked_dynamic_depth) # n 1
                    # Scale dynamic depth
                    scaled_dynamic_depth = masked_dynamic_depth*predicted_scale.unsqueeze(-1).unsqueeze(-1) # n 1 h w
                    scaled_dynamic_depth[scaled_dynamic_depth==0] = torch.tensor(1e10).float().cuda()
                    # If there are overlapping object masks, use the one that has closer depth (i.e min depth)
                    all_scaled_dynamic_depth, _ = torch.min(scaled_dynamic_depth, dim=0, keepdim=True) # 1 1 h w
                    scaled_depth = torch.clone(static_depth_list[[b]]) # / torch.mean(masked_static_depth_mean+1e-5)
                    # scaled_depth[all_scaled_dynamic_depth!=1e10] = all_scaled_dynamic_depth[all_scaled_dynamic_depth!=1e10]
                    # Get back to the original static scale
                    # scaled_disp_list.append(1/scaled_depth.detach().cpu().numpy())
                    if use_scaled_depth:
                        scaled_depth[all_scaled_dynamic_depth!=1e10] = all_scaled_dynamic_depth[all_scaled_dynamic_depth!=1e10]
                    scaled_disp_list.append(1/scaled_depth.detach().cpu().numpy())
        return scaled_disp_list

    def epoch_learning_curve_log(self, epoch_train_loss, epoch_val_loss, errors, epoch):
        writers = self.writers
        if epoch_train_loss is not None: writers['train'].add_scalar("a_epoch_eval/_loss", epoch_train_loss, epoch)
        if epoch_val_loss is not None: writers['val'].add_scalar("a_epoch_eval/_loss", epoch_val_loss, epoch)
        
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

    def train(self):
        self.create_evaluation_dataset()
        if self.current_epoch == 0:
            # Depth evaluation
            scaled_disp_list = self.get_depth_for_evaluation(use_scaled_depth=False)
            # Epoch logging
            depth_errors = compute_errors_all_dataset(scaled_disp_list, self.depth_test_loader, self.gt_depth_path, self.gt_mask_path)
            self.epoch_learning_curve_log(None, None, depth_errors, -1)

        self.global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {self.global_step}")
        print()

        for epoch in range(self.current_epoch, self.epochs):
            print(f"----------- Epoch {epoch} -----------")

            if epoch in self.reduce_lr_epoch:
                # Reduce learning rate by half when the current epoch is in lr_reduction_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2
                    print(f"Reduce learning rate to {param_group['lr']}")
            self.depth_scale_alignment.train()
            epoch_train_loss = 0
            # Training
            for samples in tqdm(self.train_loader):
                outputs = self.process_batch(samples)
                loss = outputs['loss']
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item() * len(outputs['images'])
                # Logging
                if self.global_step % self.eval_steps == 0:
                    val_samples = next(self.val_iter)
                    self.depth_scale_alignment.eval()
                    val_outputs = self.process_batch(val_samples)
                    self.log(outputs, val_outputs, self.global_step)
                    self.depth_scale_alignment.train()
                self.global_step += 1

            # Validation
            self.depth_scale_alignment.eval()
            epoch_val_loss = self.epoch_eval()
            # Depth evaluation
            scaled_disp_list = self.get_depth_for_evaluation()
            # Epoch logging
            epoch_train_loss = epoch_train_loss/(len(self.train_loader)*self.batch_size)
            epoch_val_loss = epoch_val_loss/(len(self.val_loader)*self.batch_size)
            depth_errors = compute_errors_all_dataset(scaled_disp_list, self.depth_test_loader, self.gt_depth_path, self.gt_mask_path)
            self.depth_scale_alignment.train()
            self.epoch_learning_curve_log(epoch_train_loss, epoch_val_loss, depth_errors, epoch)
            # Save model
            save_model([self.depth_scale_alignment], ['depth_scale_alignment'], self.optimizer, self.log_path, epoch)

            if depth_errors[0][0] < self.smallest_error:
                self.smallest_error = depth_errors[0][0]
                # save_model([self.depth_net,self.pose_net,self.raft], ['depth_net', 'pose_net','raft'], self.optimizer, self.log_path, None)
                save_model([self.depth_scale_alignment], ['depth_scale_alignment'], self.optimizer, self.log_path, None)
                print(f"Best model detected. Save model at epoch {epoch} with abs_error = {self.smallest_error.item()}")

    def get_pseudo_depth(self, split):
        print(f"Start retrieving pseudo depth label. Split: {split}.")
        # 0. Load the best model
        best_model_path = f'{self.log_path}/models/weights/depth_scale_alignment.pth'
        self.depth_scale_alignment.load_state_dict(torch.load(best_model_path))
        self.depth_scale_alignment.eval()
        # 1. Construct dataset
        dynamic_file_lists_path = f"{self.log_path}/dynamic_object_training_files/{split}/dynamic_object_filenames.txt"
        file_lists = np.loadtxt(dynamic_file_lists_path, dtype=object, delimiter='\n')
        image_path = self.cfg[f"{split}_data_dir"]
        depth_path = f"{self.project_path}/pixelwise_depthnet/predictions/depth/{split}"
        object_mask_path = f"{self.project_path}/object_depthnet/processed_mask/{split}"
        road_mask_path = f"{self.cfg['raw_road_mask_path']}/{split}"
        object_predicted_flow_neg_path = f"{self.project_path}/pixelwise_depthnet/predictions/flow_neg/{split}"
        object_predicted_flow_pos_path = f"{self.project_path}/pixelwise_depthnet/predictions/flow_pos/{split}"
        dsa_datasset = CityScapesObjectScaleEvalDataset(image_path, depth_path, object_mask_path, road_mask_path, 
                        object_predicted_flow_neg_path, object_predicted_flow_pos_path, 0.5, self.height, self.width, file_lists, middle_portion=20, is_train=False)
        dsa_loader = DataLoader(dsa_datasset, batch_size=self.cfg['dsa_batch_size'], num_workers=self.cfg['num_workers'], shuffle=False, pin_memory=True, drop_last=False, collate_fn=eval_dataset_collate)
        # Get pseudo_depth labels
        save_dir = f"{self.log_path}/pseudo_depth"
        os.makedirs(f'{save_dir}/{split}', exist_ok=True)
        with torch.no_grad():
            for s,samples in enumerate(tqdm(dsa_loader)):
                full_frame_id_list = samples[-1]
                samples = samples[:-1]
                images, static_depth_list, dynamic_object_mask_list, dynamic_object_bbox_list, road_masks_list, dynamic_region_mask_list = samples
                # Get prediction for full images (using StaticDepthNet)
                images = images.float().cuda()
                static_depth_list = static_depth_list.float().cuda()
                # static_depth_list = static_depth_list_[[s]]
                scaled_depth_list = []
                for b  in range(len(static_depth_list)): 
                    # Process input
                    dynamic_object_masks = dynamic_object_mask_list[b].float().cuda() # N 1 192 512
                    dynamic_object_bbox = dynamic_object_bbox_list[b].float().cuda() # N 1 192 512
                    road_masks = road_masks_list[[b]].repeat(len(dynamic_object_masks),1,1,1).float().cuda() # N 1 192 512
                    dynamic_object_masks = dynamic_object_masks * (1-road_masks)
                    road_masks = road_masks * (1-dynamic_object_bbox)
                    # Get masked depth for static regions
                    # Use road mask to mask out static depth 
                    masked_static_depth = road_masks*(1-dynamic_object_masks)*static_depth_list[[b]]
                    # Get prediction for dynamic regions
                    dynamic_depth = self.object_depth_net(images[[b]]*dynamic_object_masks)[0]
                    masked_dynamic_depth = dynamic_depth*dynamic_object_masks # b 1 h w
                    masked_dynamic_depth_mean = torch.sum(masked_dynamic_depth, dim=[1,2,3]) / (torch.sum(dynamic_object_masks, dim=[1,2,3])  + 1e-5)
                    masked_dynamic_depth = masked_dynamic_depth / (masked_dynamic_depth_mean.view(len(masked_dynamic_depth),1,1,1)+1e-5)
                    # Predict depth scale 
                    predicted_scale = self.depth_scale_alignment(images[[b]].repeat(len(dynamic_object_masks),1,1,1), dynamic_object_masks, masked_static_depth, masked_dynamic_depth) # n 1
                    # Scale dynamic depth
                    scaled_dynamic_depth = masked_dynamic_depth*predicted_scale.unsqueeze(-1).unsqueeze(-1) # n 1 h w
                    scaled_dynamic_depth[scaled_dynamic_depth==0] = torch.tensor(1e10).float().cuda()
                    # If there are overlapping object masks, use the one that has closer depth (i.e min depth)
                    all_scaled_dynamic_depth, _ = torch.min(scaled_dynamic_depth, dim=0, keepdim=True) # 1 1 h w
                    scaled_depth = torch.clone(static_depth_list[[b]]) # / torch.mean(masked_static_depth_mean+1e-5)
                    scaled_depth[all_scaled_dynamic_depth!=1e10] = all_scaled_dynamic_depth[all_scaled_dynamic_depth!=1e10]
                    scaled_depth_list.append(scaled_depth[0].detach().cpu().numpy())

                for f,full_frame_id in enumerate(full_frame_id_list):
                    city_id, _, _ = full_frame_id.split('_')
                    os.makedirs(f'{save_dir}/{split}/{city_id}',exist_ok=True)
                    np.save(f'{save_dir}/{split}/{city_id}/{full_frame_id}.npy', scaled_depth_list[f])
