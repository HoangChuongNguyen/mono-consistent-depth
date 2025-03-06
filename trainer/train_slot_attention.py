
import os
from model.slot_attention import SlotAttentionModel
from dataset_auxiliary.slot_attention_dataset import SlotAttentionDataset
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np 
from torchvision.utils import flow_to_image
from matplotlib import pyplot as plt 
import shutil
from torchvision import utils as vutils
from utils.checkpoints_utils import *
import imageio


def reconstruct_log_visualization(num_slots, depth, recon_combined, recons, ref, flow_image_neg, masks, bg_mask_subtract=None):
    # scaling for better visualization
    disp = 1 / (depth + 1e-7)
    disp = disp / torch.max(disp.view(len(disp),1,-1), dim=-1).values.view(len(disp), 1, 1, 1)
    disp = (disp - 0.5) * 2

    pred_disp = 1 / (recon_combined[:,[-1]] + 1e-7)
    pred_disp = pred_disp / torch.max(pred_disp.view(len(disp),1,-1), dim=-1).values.view(len(disp), 1, 1, 1)
    pred_disp = (pred_disp - 0.5) * 2

    recons[:,:,[-1]] = 1 / (recons[:,:,[-1]] + 1e-7)
    recons[:,:,[-1]] = recons[:,:,[-1]] / torch.max(recons[:,:,[-1]].view(len(disp), num_slots, 1,-1), dim=-1).values.view(len(disp),  num_slots, 1, 1, 1)
    recons[:,:,[-1]] = (recons[:,:,[-1]] - 0.5) * 2

    recon_combined = torch.clip(recon_combined,0,1)
    recons = torch.clip(recons,0,1)

    if bg_mask_subtract == None:
        bg_mask_subtract = torch.zeros(ref.shape[0], 1, 1, ref.shape[2], ref.shape[3])
    else:
        bg_mask_subtract = bg_mask_subtract.detach().cpu()
    

    out = torch.cat([
        ref.unsqueeze(1), 
        flow_image_neg.unsqueeze(1),
        disp.unsqueeze(1).repeat(1,1,3,1,1),  # original images
        recon_combined[:,:3].unsqueeze(1).repeat(1,1,1,1,1),  # reconstructions
        pred_disp.unsqueeze(1).repeat(1,1,3,1,1),  # reconstructions
        (recons[:,:,:3] * masks + (1 - masks.repeat(1,1,3,1,1))),  # each slot,
        # (pred_disp.repeat(1,1,3,1,1) * masks + (1 - masks.repeat(1,1,3,1,1))),  # each slot,
        ref.unsqueeze(1) * bg_mask_subtract.repeat(1,1,3,1,1),
        ref.unsqueeze(1) * masks.repeat(1,1,3,1,1)
        ], dim=1)
    
    batch_size, num_slots, C, H, W = recons.shape
    out = vutils.make_grid(
        out.view(batch_size * out.shape[1], out.shape[2], H, W).cpu(), normalize=False, nrow=out.shape[1],
    )
    out = (out+1)/2
    return out


def step_logging(writers, train_out, train_loss, train_recon_loss, train_flow_recon_loss, train_depth_recon_loss, 
                    train_bg_mask_loss, train_bg_mask_subtract_loss, train_bg_recon_loss, train_flow_bg_recon_loss, train_depth_bg_recon_loss, train_z_loss,
                val_out, val_loss, val_recon_loss, val_flow_recon_loss, val_depth_recon_loss, 
                    val_bg_mask_loss, val_bg_mask_subtract_loss, val_bg_recon_loss, val_flow_bg_recon_loss, val_depth_bg_recon_loss, val_z_loss,
                optimizer, z_loss_w, global_step):

    writers["train"].add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
    writers["train"].add_scalar("z_weight", z_loss_w, global_step)

    writers["train"].add_image("train_image", train_out, global_step)
    writers["val"].add_image("val_image", val_out, global_step)

    writers["train"].add_scalar("loss", train_loss.item(), global_step)
    writers["val"].add_scalar("loss", val_loss.item(), global_step)
    writers["train"].add_scalar("recon_loss/total", train_recon_loss.item(), global_step)
    writers["val"].add_scalar("recon_loss/total", val_recon_loss.item(), global_step)
    writers["train"].add_scalar("recon_loss/flow", train_flow_recon_loss.item(), global_step)
    writers["val"].add_scalar("recon_loss/flow", val_flow_recon_loss.item(), global_step)
    writers["train"].add_scalar("recon_loss/depth", train_depth_recon_loss.item(), global_step)
    writers["val"].add_scalar("recon_loss/depth", val_depth_recon_loss.item(), global_step)

    writers["train"].add_scalar("bg_loss/bg_ce_loss", train_bg_mask_loss.item(), global_step)
    writers["val"].add_scalar("bg_loss/bg_ce_loss", val_bg_mask_loss.item(), global_step)
    writers["train"].add_scalar("bg_loss/bg_ce_loss_subtract", train_bg_mask_subtract_loss.item(), global_step)
    writers["val"].add_scalar("bg_loss/bg_ce_loss_subtract", val_bg_mask_subtract_loss.item(), global_step)
    writers["train"].add_scalar("bg_recon_loss/bg_recon_loss_total", train_bg_recon_loss.item(), global_step)
    writers["val"].add_scalar("bg_recon_loss/bg_recon_loss_total", val_bg_recon_loss.item(), global_step)
    writers["train"].add_scalar("bg_recon_loss/flow", train_flow_bg_recon_loss.item(), global_step)
    writers["val"].add_scalar("bg_recon_loss/flow", val_flow_bg_recon_loss.item(), global_step)
    writers["train"].add_scalar("bg_recon_loss/depth", train_depth_bg_recon_loss.item(), global_step)
    writers["val"].add_scalar("bg_recon_loss/depth", val_depth_bg_recon_loss.item(), global_step)

    writers["train"].add_scalar("z_loss", train_z_loss.item(), global_step)
    writers["val"].add_scalar("z_loss", val_z_loss.item(), global_step)



class SlotAttentionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.project_path = self.cfg['log_path']
        self.log_path = f"{cfg['log_path']}/slot_attention"
        self.depth_net_type = cfg['depth_net_type']

        # !!!! HARDCODED A FEW PARAMETERS HERE !!!!
        self.global_step = 0
        self.current_epoch = 0
        # Dataset parameter
        self.epochs = cfg['sa_epochs']
        self.flow_threshold = cfg['sa_flow_threshold']
        self.extracted_mask_threshold = cfg['sa_extracted_mask_threshold']
        self.batch_size = cfg['sa_batch_size']
        self.height = cfg['slot_height']
        self.width = cfg['slot_width']
        # Model parameters
        self.in_channels = cfg['sa_in_channels']
        self.out_channels = cfg['sa_out_channels']
        self.num_slots = cfg['sa_num_slots']
        self.num_iterations = cfg['sa_num_iterations']
        self.empty_cache = cfg['sa_empty_cache']
        # Loss parameters
        self.z_loss_w_max = cfg['sa_z_loss_w_max']
        self.z_loss_w_min = cfg['sa_z_loss_w_min']
        self.z_loss_w_start_steps = cfg['sa_z_loss_w_start_steps']
        self.z_loss_w_end_steps = cfg['sa_z_loss_w_end_steps']
        self.recons_loss_w = cfg['sa_recons_loss_w']
        self.flow_recons_loss_w = cfg['sa_flow_recons_loss_w']
        self.depth_recons_loss_w = cfg['sa_depth_recons_loss_w']
        self.bg_mask_loss_w = cfg['sa_bg_mask_loss_w']
        self.bg_recon_loss_w = cfg['sa_bg_recon_loss_w']
        self.bg_mask_loss_epoch = cfg['sa_bg_mask_loss_epoch']
        self.bg_recon_loss_epoch = cfg['sa_bg_recon_loss_epoch']
        # Optimizer parameters
        self.lr = cfg['sa_lr']
        self.warmup_steps_pct = cfg['sa_warmup_steps_pct']
        self.decay_steps_pct = cfg['sa_decay_steps_pct']
        self.weight_decay = cfg['sa_weight_decay']
        self.scheduler_gamma = cfg['sa_scheduler_gamma']
        self.eval_step = cfg['sa_eval_step']
        # Dataloading
        self.original_height = cfg['height']
        self.original_width = cfg['width']
        self.num_workers = cfg['num_workers']
        self.device = cfg['device']
        # Training dataset variables
        self.train_file_path = cfg['train_file_path']
        self.train_image_path = cfg['train_data_dir']
        self.train_depth_dir = f'{self.project_path}/pixelwise_depthnet/predictions/depth/train'
        self.train_flow_neg_dir = f'{self.project_path}/pixelwise_depthnet/predictions/flow_neg/train'
        self.train_flow_pos_dir = f'{self.project_path}/pixelwise_depthnet/predictions/flow_pos/train'
        # Validation dataset variables
        self.val_file_path = cfg['val_file_path']
        self.val_image_path = cfg['val_data_dir']
        self.val_depth_dir = f'{self.project_path}/pixelwise_depthnet/predictions/depth/val'
        self.val_flow_neg_dir = f'{self.project_path}/pixelwise_depthnet/predictions/flow_neg/val'
        self.val_flow_pos_dir = f'{self.project_path}/pixelwise_depthnet/predictions/flow_pos/val'
        # Loading model parameters
        self.load_weights_folder = cfg['slot_attention_load_weights_folder']
        # Define dataset
        self.train_files = np.sort(np.loadtxt(self.train_file_path, dtype=object, delimiter='\n'))
        self.train_set = SlotAttentionDataset(self.train_image_path, self.train_depth_dir, self.train_flow_neg_dir, self.train_flow_pos_dir, 
                                        self.train_files, self.height, self.width, self.flow_threshold, is_train=True)

        self.val_files = np.sort(np.loadtxt(self.val_file_path, dtype=object, delimiter='\n'))
        self.val_set = SlotAttentionDataset(self.val_image_path, self.val_depth_dir, self.val_flow_neg_dir, self.val_flow_pos_dir, 
                                        self.val_files, self.height, self.width, self.flow_threshold, is_train=False)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=self.num_workers)
        self.val_iter = iter(self.val_loader)

        # # Define model
        self.model = SlotAttentionModel(
                resolution=(self.height, self.width),
                decoder_resolution=(self.height//16,self.width//16),
                num_slots=self.num_slots,
                num_iterations=self.num_iterations,
                empty_cache=self.empty_cache,
                in_channels=self.in_channels,
                out_channels=self.out_channels
            )
        self.model = self.model.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.total_steps = self.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self.warm_and_decay_lr_scheduler)

        if self.load_weights_folder is not None:
            self.current_epoch = load_model([self.model, self.scheduler], ['slot_attention', 'scheduler'], self.optimizer, self.load_weights_folder) + 1
        # Log set up 
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=10)

    def warm_and_decay_lr_scheduler(self, step: int):
        warmup_steps = self.warmup_steps_pct * self.total_steps
        decay_steps = self.decay_steps_pct * self.total_steps
        # assert step < self.total_steps
        if step < warmup_steps:
            factor = step / warmup_steps
        else:
            factor = 1
        factor *= self.scheduler_gamma ** (step / decay_steps)
        return factor

    def get_prediction(self):
        def process_batch(samples, model):
            # Process input
            frame_id = samples[0]
            samples =  samples[1:]
            image, flow_image_neg, flow_image_pos, flow_neg_dynamic_mask, flow_pos_dynamic_mask, depth_neg_exp, depth_pos_exp = [item.float().cuda() for item in samples]
            image = image.repeat(2,1,1,1) # b*2 3 h w
            flow_image = torch.cat([flow_image_neg, flow_image_pos], dim=0) # b*2 3 h w
            dynamic_mask = torch.cat([flow_neg_dynamic_mask, flow_pos_dynamic_mask], dim=0) # b*2 1 h w
            depth_exp = torch.cat([depth_neg_exp, depth_pos_exp], dim=0) # b*2 1 h w
            bg_mask = 1 - dynamic_mask
            # Make predictions
            inputs_features = torch.cat([flow_image, depth_exp], dim=1)
            recon_combined, recons, masks, slots, z_value, bg_mask_subtract = model(inputs_features, use_z=False)
            inputs = (image, depth_exp, flow_image)
            outputs = (recon_combined, recons, masks, slots, z_value, bg_mask_subtract)
            return inputs, outputs
        # Define dataset
        train_set = SlotAttentionDataset(self.train_image_path, self.train_depth_dir, self.train_flow_neg_dir, self.train_flow_pos_dir, 
                                        self.train_files, self.height, self.width, self.flow_threshold, is_train=False)
        val_set = SlotAttentionDataset(self.val_image_path, self.val_depth_dir, self.val_flow_neg_dir, self.val_flow_pos_dir, 
                                        self.val_files, self.height, self.width, self.flow_threshold, is_train=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, pin_memory=True, shuffle=False, drop_last=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, pin_memory=True, shuffle=False, drop_last=False, num_workers=self.num_workers)
        # Get prediction
        dataset_dict = {'train': train_loader, 'val': val_loader} 
        self.model.eval()
        with torch.no_grad():
            for split in dataset_dict.keys():
                print('Get prediction in split: ', split)
                data_loader = dataset_dict[split]
                for samples in tqdm(data_loader): 
                    full_frame_id_list = samples[0]
                    # Get predictions
                    inputs, outputs = process_batch(samples, self.model)
                    recon_combined, recons, masks, slots, z_value, bg_mask_subtract = outputs
                    b, s, c, h, w = masks.shape
                    resized_mask = torch.nn.functional.interpolate(masks.view(b*s,c,h,w), (self.original_height, self.original_width)).view(b,s,c,self.original_height,self.original_width)
                    thresholded_mask_list = (resized_mask >= self.extracted_mask_threshold).detach().cpu()[:,:,0]
                    for f,full_frame_id in enumerate(full_frame_id_list):
                        city_id = full_frame_id.split("_")[0]
                        os.makedirs(f"{self.log_path}/predictions/{split}/{city_id}", exist_ok=True)
                        thresholded_mask = thresholded_mask_list[f]
                        thresholded_mask = torch.cat(list(thresholded_mask), dim=-1).numpy()
                        thresholded_mask = thresholded_mask.astype(int)
                        imageio.imsave(f"{self.log_path}/predictions/{split}/{city_id}/{full_frame_id}.png", thresholded_mask)
                        
    def process_batch(self, samples, global_step, epoch):
        # Process input
        frame_id = samples[0]
        samples =  samples[1:]
        image, flow_image_neg, flow_image_pos, flow_neg_dynamic_mask, flow_pos_dynamic_mask, depth_neg_exp, depth_pos_exp = [item.float().cuda() for item in samples]
        image = image.repeat(2,1,1,1) # b*2 3 h w
        flow_image = torch.cat([flow_image_neg, flow_image_pos], dim=0) # b*2 3 h w
        dynamic_mask = torch.cat([flow_neg_dynamic_mask, flow_pos_dynamic_mask], dim=0) # b*2 1 h w
        depth_exp = torch.cat([depth_neg_exp, depth_pos_exp], dim=0) # b*2 1 h w
        bg_mask = 1 - dynamic_mask
        # Make predictions
        inputs_features = torch.cat([flow_image, depth_exp], dim=1)
        recon_combined, recons, masks, slots, z_value, bg_mask_subtract = self.model(inputs_features, use_z=global_step>=self.z_loss_w_start_steps)

        # Loss 1: Reconstruction loss for flow and depth
        train_flow_recon_loss = torch.nn.functional.mse_loss(recon_combined[:,:3], inputs_features[:,:3]) * self.flow_recons_loss_w
        train_depth_recon_loss = torch.nn.functional.mse_loss(recon_combined[:,[-1]], inputs_features[:,[-1]]) * self.depth_recons_loss_w
        train_recon_loss = train_flow_recon_loss + train_depth_recon_loss
        # Loss 2: Cross entropy loss for background and the subtract background
        train_bg_mask_loss = torch.nn.BCEWithLogitsLoss()(masks[:,0], bg_mask)
        train_bg_mask_subtract_loss = torch.nn.BCEWithLogitsLoss()(bg_mask_subtract[:,0], bg_mask) if bg_mask_subtract != None else torch.tensor(0).cuda()
        # Loss 3: Reconstruction loss for background and the subtract background
        train_flow_bg_recon_loss = torch.nn.functional.mse_loss(recons[:,0,:3], torch.ones_like(recons[:,0,:3])) * self.flow_recons_loss_w * int(epoch >= self.bg_recon_loss_epoch)
        train_depth_bg_recon_loss = torch.nn.functional.mse_loss(recons[:,0,[-1]], torch.ones_like(recons[:,0,[-1]])) * self.depth_recons_loss_w * 0.1 * int(epoch >= self.bg_recon_loss_epoch)
        train_bg_recon_loss = train_flow_bg_recon_loss + train_depth_bg_recon_loss
        # Loss 4: Sparsity loss for the regularizer, which encourages empty slot.
        z_loss_w = np.clip(((global_step - self.z_loss_w_start_steps) / (self.z_loss_w_end_steps - self.z_loss_w_start_steps)) * (self.z_loss_w_max - self.z_loss_w_min) + self.z_loss_w_min, a_min=0.0, a_max=self.z_loss_w_max)
        if global_step < self.z_loss_w_start_steps: z_loss_w = torch.tensor(0).cuda()
        train_z_loss = torch.mean(torch.abs(z_value - 0.01))
        train_z_loss = train_z_loss * z_loss_w * int(global_step>=self.z_loss_w_start_steps)
        
        # The loss for background is only used after bg_mask_loss_epoch step
        train_bg_mask_loss = train_bg_mask_loss  * self.bg_mask_loss_w * int(epoch >= self.bg_mask_loss_epoch)
        train_bg_mask_subtract_loss = train_bg_mask_subtract_loss  * self.bg_mask_loss_w * int(epoch >= self.bg_mask_loss_epoch)
        train_bg_recon_loss = train_bg_recon_loss  * self.bg_recon_loss_w * int(epoch >= self.bg_recon_loss_epoch)

        # Return all loss, inputs and outputs
        loss = (train_recon_loss, train_flow_recon_loss, train_depth_recon_loss, train_bg_recon_loss, train_flow_bg_recon_loss, train_depth_bg_recon_loss, train_bg_mask_loss, train_bg_mask_subtract_loss, train_z_loss)
        inputs = (image, depth_exp, flow_image)
        outputs = (recon_combined, recons, masks, slots, z_value, bg_mask_subtract)

        return inputs, outputs, loss, z_loss_w

    def train(self):
        self.global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {self.global_step}")
        print()

        for epoch in range(self.current_epoch, self.epochs):
            print("Epoch ", epoch)
            self.model.train()
            epoch_train_loss = 0
            for samples in tqdm(self.train_loader):
                # Get predictions
                inputs, outputs, loss, z_loss_w = self.process_batch(samples, self.global_step, epoch)
                recon_combined, recons, masks, slots, z_value, bg_mask_subtract = outputs
                ref, depth, flow_image_neg = inputs
                # Calculate loss function
                train_recon_loss, train_flow_recon_loss, train_depth_recon_loss, train_bg_recon_loss, train_flow_bg_recon_loss, train_depth_bg_recon_loss, train_bg_mask_loss, train_bg_mask_subtract_loss, train_z_loss = loss 
                train_loss = train_recon_loss + train_bg_mask_loss + train_bg_recon_loss + train_bg_mask_subtract_loss + train_z_loss
                epoch_train_loss += train_loss * len(ref) * 2
                # Back propagation
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Evaluation and log
                if self.global_step % self.eval_step == 0:
                    out = reconstruct_log_visualization(self.num_slots, depth.detach().cpu(), recon_combined.detach().cpu(), recons.detach().cpu(), 
                                                    ref.detach().cpu(), flow_image_neg.detach().cpu(), masks.detach().cpu(), bg_mask_subtract)
                    self.model.eval()
                    with torch.no_grad():
                        try:
                            val_samples = next(self.val_iter)
                        except:
                            self.val_iter = iter(self.val_loader)
                            val_samples = next(self.val_iter)
                        val_inputs, val_outputs, val_loss, _ = self.process_batch(val_samples, self.global_step, epoch)
                        val_recon_combined, val_recons, val_masks, val_slots, val_z_value, val_bg_mask_subtract = val_outputs
                        val_ref, val_depth, val_flow_image_neg = val_inputs
                        # Calculate loss function
                        val_recon_loss, val_flow_recon_loss, val_depth_recon_loss, val_bg_recon_loss, val_flow_bg_recon_loss, val_depth_bg_recon_loss, val_bg_mask_loss, val_bg_mask_subtract_loss, val_z_loss = val_loss 
                        val_loss = val_recon_loss + val_bg_mask_loss + val_bg_recon_loss + val_bg_mask_subtract_loss + val_z_loss
                        val_out = reconstruct_log_visualization(self.num_slots, val_depth.detach().cpu(), val_recon_combined.detach().cpu(), val_recons.detach().cpu(), 
                                                                val_ref.detach().cpu(), val_flow_image_neg.detach().cpu(), val_masks.detach().cpu(), bg_mask_subtract)
                    self.model.train()
                    # Logging
                    step_logging(self.writers, out, train_loss, train_recon_loss, train_flow_recon_loss, train_depth_recon_loss, 
                                    train_bg_mask_loss, train_bg_mask_subtract_loss, train_bg_recon_loss, train_flow_bg_recon_loss, train_depth_bg_recon_loss, train_z_loss,
                                    val_out, val_loss, val_recon_loss, val_flow_recon_loss, val_depth_recon_loss, 
                                    val_bg_mask_loss, val_bg_mask_subtract_loss, val_bg_recon_loss, val_flow_bg_recon_loss, val_depth_bg_recon_loss, val_z_loss,
                                    self.optimizer, z_loss_w, self.global_step)
                self.global_step += 1
            # Log and save the model
            epoch_train_loss = epoch_train_loss / (self.batch_size * 2 * len(self.train_loader))
            self.writers["train"].add_scalar("epoch_loss", epoch_train_loss, epoch)
            save_model([self.model, self.scheduler], ['slot_attention', 'scheduler'], self.optimizer, self.log_path, epoch)


