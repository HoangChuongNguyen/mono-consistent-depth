
import os
import imageio
import numpy as np
import cv2
from tqdm import  tqdm
from matplotlib import  pyplot as plt
from utils.mask_utils import generate_bounding_box
from dataset_auxiliary.object_detection_dataset import ObjectDetectionDataset
import torch 
from utils.checkpoints_utils import  *
from tensorboardX import SummaryWriter
from torchvision.utils import draw_segmentation_masks
# import MaskRCNN 
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from joblib import Parallel, delayed
from model_zoo.model_zoo import *
import torch.nn.functional as F

def collate_fn(batch):
    return tuple(zip(*batch))
def get_file_list(data_dir):
    file_lists =  []
    for city in sorted(os.listdir(f"{data_dir}")):
        for frame_id in sorted(os.listdir(f"{data_dir}/{city}")):
            if '.png' in frame_id: frame_id = frame_id.replace('.png', '')
            file_lists.append(f"{city} {frame_id}")
    return np.array(sorted(file_lists))
def log(train_outputs, val_outputs, score_threhold, writers, global_step):
    outputs = {'train': train_outputs, 'val': val_outputs}
    for mode in ['train', 'val']:
        writer = writers[mode]
        output = outputs[mode]
        images = output['images']
        targets = output['targets']
        predictions = output['predictions']
        for p in range(2):
            image = (images[p]*255.0).to(torch.uint8)[:3]
            # Get the image annotated with ground truth mask
            gt_masks = targets[p]['masks'] == 1
            gt_annotated_image = draw_segmentation_masks(image.detach().cpu()[:3], masks=gt_masks.detach().cpu(), alpha=0.7)
            # Get the image annotated with predicted mask
            pred_masks = predictions[p]['masks']
            pred_score = predictions[p]['scores']
            best_score_idx = torch.flip(torch.argsort(pred_score), dims=[0])[:5]
            pred_masks = pred_masks[best_score_idx]
            pred_score = pred_score[best_score_idx]
            # pred_masks = pred_masks[pred_score>score_threhold]
            pred_masks = pred_masks[:,0] >= 0.5
            pred_annotated_image = draw_segmentation_masks(image.detach().cpu()[:3], masks=pred_masks.detach().cpu(), alpha=0.7)
            # Convert to numpy and opencv format
            pred_annotated_image_np = pred_annotated_image.permute(1, 2, 0).numpy()
            pred_annotated_image_np = cv2.cvtColor(pred_annotated_image_np, cv2.COLOR_RGB2BGR)
            # Write scores onto image
            for i in range(pred_masks.shape[0]):
                mask = pred_masks[i].detach().cpu().numpy()
                score = pred_score[i].detach().cpu().numpy()
                # calculate center of the mask
                y, x = np.where(mask)  # get the indices of non-zero elements
                center_x, center_y = np.mean(x), np.mean(y)
                try:
                    # Write the score on the center of the mask
                    cv2.putText(pred_annotated_image_np, "{:.2f}".format(score), (int(center_x), int(center_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except: pass
            # Convert back to torch format
            pred_annotated_image = torch.from_numpy(cv2.cvtColor(pred_annotated_image_np, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

            stacked_image = torch.concat([gt_annotated_image, pred_annotated_image], dim=1)
            writer.add_image(f"images/image_{p}", stacked_image, global_step)
def log_epoch(epoch_train_loss, epoch_val_loss, writers, epoch):
    writers['train'].add_scalar("epoch_loss", epoch_train_loss, epoch)
    writers['val'].add_scalar("epoch_loss", epoch_val_loss, epoch)

# Helper function for mask processing
def generate_bounding_box2(mask):
    # Find the coordinates of the bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # Create a bounding box mask
    bbox_mask = np.zeros_like(mask)
    bbox_mask[ymin:ymax+1, xmin:xmax+1] = 1
    return bbox_mask    

def get_unique_masks(masks, unique_threshold=0.25, connected_threshold=0.5):
    def get_area(mask):
        return np.sum(mask)
    def iou(target_mask, reference_mask):
        intersection = np.sum((target_mask==1) & (reference_mask==1))
        return intersection / np.sum(target_mask)
    extended_masks_list = []
    for i in range(len(masks)):
        iou_list = np.array([iou(masks[i], masks[j]) for j in range(len(masks))])
        iou_list[i] = 0
        if np.max(iou_list) >= connected_threshold:
            largest_masks_idx = np.where(iou_list >= unique_threshold)[0]
            largest_masks = masks[largest_masks_idx]
            largest_largest_masks_idx = np.argmax(np.sum(largest_masks.reshape(len(largest_masks),-1), axis=-1))
            masks[largest_masks_idx[largest_largest_masks_idx]] = np.clip(masks[i] + largest_masks[largest_largest_masks_idx],0,1)
            masks[i] = largest_masks[largest_largest_masks_idx]
        extended_masks_list.append(masks[i])
    extended_masks_list = np.stack(extended_masks_list)
    unique_masks_list = []
    for i in range(len(extended_masks_list)):
        iou_list = np.array([iou(extended_masks_list[i], extended_masks_list[j]) for j in range(len(extended_masks_list))])
        iou_list[i] = 0
        if np.max(iou_list) >= unique_threshold:
            largest_masks_idx = np.where(iou_list >= unique_threshold)[0]
            largest_largest_masks_idx = np.argmax(np.sum(extended_masks_list[largest_masks_idx].reshape(len(extended_masks_list[largest_masks_idx]),-1), axis=-1).reshape(-1))
            largest_masks = extended_masks_list[largest_masks_idx[largest_largest_masks_idx]]
            extended_masks_list[i] = largest_masks
        unique_masks_list.append(extended_masks_list[i])

    unique_masks_list = [mask for mask in unique_masks_list if np.sum(mask) != 0]
    unique_masks_list = np.unique(unique_masks_list, axis=0)
    return np.array(unique_masks_list)
    
# 1. Remove wrong masks which has more than 1 object
def remove_multiple_object_masks(mask_list, iou_threshold=0.2):
    processed_mask_list = []
    for mask in mask_list:
        bbox_mask = generate_bounding_box2(mask)
        wrong_region = (bbox_mask != mask).astype(float)
        if np.sum(wrong_region) / np.sum(mask) >= iou_threshold: 
            continue
        processed_mask_list.append(mask)
    processed_mask_list = np.array(processed_mask_list) # N H W
    return processed_mask_list

def mask_out_road(object_mask_list, road_mask):
    object_mask_list = object_mask_list*(1-np.expand_dims(road_mask, axis=0))
    return object_mask_list


def process_predicted_mask(mask_list, road_mask, iou_threshold=0.1, unique_threshold=0.25, connected_threshold=0.5):
    if len(mask_list) == 1: return mask_list
    processed_mask = remove_multiple_object_masks(mask_list, iou_threshold)
    if len(processed_mask) == 0: return processed_mask
    processed_mask = get_unique_masks(processed_mask, unique_threshold, connected_threshold)
    processed_mask = mask_out_road(processed_mask, road_mask)
    return processed_mask

def process_pred(pred, road_mask, frame_id, score_threshold, mask_size_threshold, processed_mask_epochs, log_path, epoch, split):
    pred_masks = pred["masks"].detach().cpu()[pred["scores"] > score_threshold]
    pred_masks = (pred_masks.detach().cpu().numpy() >= 0.5).astype(float)  # N H W
    pred_masks = [mask[0] for mask in pred_masks if np.sum(mask) >= mask_size_threshold*mask.shape[0]*mask.shape[1]]
    if len(pred_masks) == 0: return
    pred_masks = np.stack(pred_masks)
    # Save data            
    city_id = frame_id.split("_")[0]
    if epoch in processed_mask_epochs:
        if len(pred_masks) != 1:
            try:
                pred_masks = process_predicted_mask(pred_masks, road_mask[0].numpy(), iou_threshold=0.1, unique_threshold=0.25, connected_threshold=0.5)
            except: pass
        if len(pred_masks) == 0: return
    pred_masks = torch.from_numpy(pred_masks)
    pred_masks = [mask for mask in pred_masks if torch.sum(mask) >= mask_size_threshold*mask.shape[0]*mask.shape[1]]
    if len(pred_masks) == 0: return
    folder_path = f'{log_path}/predictions/epoch_{epoch}/{split}/{city_id}/{frame_id}'
    # if not os.path.exists(folder_path): 
    os.makedirs(folder_path, exist_ok=True)

    for i, mask in enumerate(pred_masks):
        imageio.imsave(f'{folder_path}/{i}.png', mask)
        


class MaskAugmentator:

    def __init__(self):
        self.augmentor_list = [self.identity_augmentor, self.crop_top_left, self.crop_top_right, self.crop_top_left_middle, self.crop_top_right_middle]
        self.inverse_augmentor_list = [self.identity_inverse_augmentor, self.inverse_crop_top_left, self.inverse_crop_top_right, self.inverse_crop_top_left_middle, self.inverse_crop_top_right_middle]

    @staticmethod
    def identity_augmentor(x,scale): return x

    @staticmethod
    def crop_bottom_left(images, scale):
        _, _, original_height, original_width = images.shape
        crop_height, crop_width = int(original_height * scale), int(original_width * scale)
        cropped = images[:, :, :crop_height, :crop_width]
        return F.interpolate(cropped, size=(original_height, original_width), mode='bilinear', align_corners=False)

    @staticmethod
    def crop_bottom_right(images, scale):
        _, _, original_height, original_width = images.shape
        crop_height, crop_width = int(original_height * scale), int(original_width * scale)
        cropped = images[:, :, :crop_height, -crop_width:]
        return F.interpolate(cropped, size=(original_height, original_width), mode='bilinear', align_corners=False)

    @staticmethod
    def crop_top_left(images, scale):
        _, _, original_height, original_width = images.shape
        crop_height, crop_width = int(original_height * scale), int(original_width * scale)
        cropped = images[:, :, -crop_height:, :crop_width]
        return F.interpolate(cropped, size=(original_height, original_width), mode='bilinear', align_corners=False)

    @staticmethod
    def crop_top_right(images, scale):
        _, _, original_height, original_width = images.shape
        crop_height, crop_width = int(original_height * scale), int(original_width * scale)
        cropped = images[:, :, -crop_height:, -crop_width:]
        return F.interpolate(cropped, size=(original_height, original_width), mode='bilinear', align_corners=False)

    @staticmethod
    def crop_middle(images, scale):
        _, _, original_height, original_width = images.shape
        crop_height, crop_width = int(original_height * scale), int(original_width * scale)
        start_y = (original_height - crop_height) // 2
        start_x = (original_width - crop_width) // 2
        cropped = images[:, :, start_y:start_y+crop_height, start_x:start_x+crop_width]
        return F.interpolate(cropped, size=(original_height, original_width), mode='bilinear', align_corners=False)

    @staticmethod
    def crop_top_left_middle(images, scale):
        intermediate = MaskAugmentator.crop_middle(images, scale=0.5)
        return MaskAugmentator.crop_top_left(intermediate, scale=scale)

    @staticmethod
    def crop_top_right_middle(images, scale):
        intermediate = MaskAugmentator.crop_middle(images, scale=0.5)
        return MaskAugmentator.crop_top_right(intermediate, scale=scale)

    @staticmethod
    def identity_inverse_augmentor(x,scale,original_height, original_width): return x

    @staticmethod
    def inverse_crop_top_left(images, scale, original_height, original_width):
        crop_height = int(original_height * scale)
        crop_width = int(original_width * scale)
        resized = F.interpolate(images, size=(crop_height, crop_width), mode='bilinear', align_corners=False)
        pad_top = original_height - crop_height
        pad_right = original_width - crop_width
        return F.pad(resized, (0, pad_right, pad_top, 0))

    @staticmethod
    def inverse_crop_top_right(images, scale, original_height, original_width):
        crop_height = int(original_height * scale)
        crop_width = int(original_width * scale)
        resized = F.interpolate(images, size=(crop_height, crop_width), mode='bilinear', align_corners=False)
        pad_top = original_height - crop_height
        pad_left = original_width - crop_width
        return F.pad(resized, (pad_left, 0, pad_top, 0))

    @staticmethod
    def inverse_crop_middle(images, scale, original_height, original_width):
        crop_height = int(original_height * scale)
        crop_width = int(original_width * scale)
        resized = F.interpolate(images, size=(crop_height, crop_width), mode='bilinear', align_corners=False)
        pad_top = (original_height - crop_height) // 2
        pad_bottom = original_height - crop_height - pad_top
        pad_left = (original_width - crop_width) // 2
        pad_right = original_width - crop_width - pad_left
        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))

    @staticmethod
    def inverse_crop_top_left_middle(images, scale, original_height, original_width):
        half_height, half_width = int(original_height * 0.5), int(original_width * 0.5)
        intermediate = MaskAugmentator.inverse_crop_top_left(images, scale, half_height, half_width)
        return MaskAugmentator.inverse_crop_middle(intermediate, 0.5, original_height, original_width)

    @staticmethod
    def inverse_crop_top_right_middle(images, scale, original_height, original_width):
        half_height, half_width = int(original_height * 0.5), int(original_width * 0.5)
        intermediate = MaskAugmentator.inverse_crop_top_right(images, scale, half_height, half_width)
        return MaskAugmentator.inverse_crop_middle(intermediate, 0.5, original_height, original_width)

class ObjectDetectionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.project_path = self.cfg['log_path']
        self.log_path = f"{cfg['log_path']}/object_detection"
        self.depth_net_type = cfg['depth_net_type']
        self.mask_augmentator = MaskAugmentator()

        self.current_epoch = 0
        self.global_step = 0
        # Training params
        self.flow_threshold = cfg['sa_flow_threshold']
        self.object_size_threshold = cfg['od_object_size_threshold']
        self.num_epochs = cfg['od_num_epochs']
        self.batch_size = cfg['od_batch_size']
        self.num_workers = cfg['od_num_workers']
        self.eval_step = cfg['od_eval_step']
        self.do_self_supervise = cfg['od_do_self_supervise']
        self.mask_size_threshold = cfg['od_mask_size_threshold']
        self.self_supervise_score_threshold_dict = cfg['od_self_supervise_score_threshold_dict']
        self.processed_mask_epochs = cfg['od_processed_mask_epochs']
        self.mask_augmentation_epochs = cfg['od_mask_augmentation_epochs']
        # Dataloading
        self.height = cfg['height']
        self.width = cfg['width']
        self.num_workers = cfg['num_workers']
        self.device = cfg['device']
        self.nb_gpus = cfg['nb_gpus']
        # Depthnet params
        self.num_scales = cfg['num_scales']
        self.depth_encoder_pretrained = cfg['depth_encoder_pretrained']
        self.encoder_use_randomize_layernorm = cfg['encoder_use_randomize_layernorm']
        self.tf_imageNet_checkpoint_path = cfg['tf_imageNet_checkpoint_path']
        # self.many_depth_teacher_freeze_epoch = cfg['many_depth_teacher_freeze_epoch']
        # Dataset variables
        self.flow_neg_dir = f'{self.project_path}/pixelwise_depthnet/predictions/flow_neg/'
        self.flow_pos_dir = f'{self.project_path}/pixelwise_depthnet/predictions/flow_pos/'
        self.slot_attention_mask_dir = f'{self.project_path}/slot_attention/predictions/'
        self.predicted_ground_mask_dir = f'{self.project_path}/ground_segmentation/predictions/'
        # Training dataset variables
        self.train_file_path = cfg['train_file_path']
        self.train_image_path = cfg['train_data_dir']
        self.train_files = np.sort(np.loadtxt(self.train_file_path, dtype=object, delimiter='\n'))
        # Validation dataset variables
        self.val_file_path = cfg['val_file_path']
        self.val_image_path = cfg['val_data_dir']
        self.val_files = np.sort(np.loadtxt(self.val_file_path, dtype=object, delimiter='\n'))
        # Training full dataset variables (for getting prediction)
        self.train_image_dir = f"./data/cityscapes/citiscapes_512x1024"
        self.train_depth_dir = f'{self.project_path}/pixelwise_depthnet/predictions/depth/train'
        self.train_ground_mask_dir =  f'{self.predicted_ground_mask_dir}/train'
        self.train_slot_mask_dir = f"{self.log_path}/processed_slot_mask/train"
        # Validation full dataset variables (for getting prediction)
        self.val_image_dir = f"./data/cityscapes/cityscapes_val_512x1024"
        self.val_depth_dir = f'{self.project_path}/pixelwise_depthnet/predictions/depth/val'
        self.val_ground_mask_dir =  f'{self.predicted_ground_mask_dir}/val'
        self.val_slot_mask_dir = f"{self.log_path}/processed_slot_mask/val"
        # Loading model parameters
        self.load_weights_folder = cfg['object_detection_load_weights_folder']
        # Model setup
        num_classes = 2
        backbone = resnet_fpn_backbone("resnet18", pretrained=True)
        # Modify the first layer to take 4-channel input
        new_conv = torch.nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = backbone.body.conv1.weight
            new_conv.weight[:, 3] = backbone.body.conv1.weight[:, 0] # Initialize new channel with weights from red channel
            new_conv.weight[:, 4] = backbone.body.conv1.weight[:, 0] # Initialize new channel with weights from red channel
        backbone.body.conv1 = new_conv
        # Define main model
        self.model = MaskRCNN(backbone, num_classes=num_classes)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        self.model.rpn.anchor_generator = anchor_generator
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        self.model.module.transform.image_mean = torch.tensor([0.4850, 0.4560, 0.4060, 0.0, 0.0])
        self.model.module.transform.image_std = torch.tensor([0.229, 0.224, 0.225, 1.0, 1.0])
        # Define optimizer and scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)

        if self.load_weights_folder is not None:
            self.dataset_setup()
            self.current_epoch = load_model([self.model, self.lr_scheduler], ['maskrcnn', 'scheduler'], self.optimizer, self.load_weights_folder) + 1
            epoch = self.current_epoch-1
            if self.do_self_supervise and epoch in list(self.self_supervise_score_threshold_dict.keys()):
                # Store prediction
                self.store_predictions(dataloader=self.train_full_loader, split='train', epoch=epoch)
                self.store_predictions(dataloader=self.val_full_loader, split='val', epoch=epoch)
                # Get new training dataset and data loaders
                self.generate_new_dataset(epoch)
                
        # Log set up 
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=10)

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
        else:
            assert False, f'Unknow depth_net_type {self.depth_net_type}. It should be in ["corl2020", "diffnet", "brnet", "packnet"]'
        self.depth_net = torch.nn.DataParallel(self.depth_net, device_ids=list(range(self.nb_gpus)))
        # Load the best depth_net model 
        load_model([self.depth_net], ['depth_net'], None, f"{self.project_path}/pixelwise_depthnet/models/weights")



    def dataset_setup(self):
        # Define dataset (with slot mask)
        self.train_mask_files = np.array(get_file_list(self.train_slot_mask_dir))
        self.train_set = ObjectDetectionDataset(file_lists=self.train_mask_files, image_dir=self.train_image_dir, depth_dir=self.train_depth_dir, 
                                            ground_mask_dir=self.train_ground_mask_dir, mask_dir=self.train_slot_mask_dir, height=self.height, width=self.width, is_train=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, 
                                                num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

        self.val_mask_files = np.array(get_file_list(self.val_slot_mask_dir))
        self.val_set = ObjectDetectionDataset(file_lists=self.val_mask_files, image_dir=self.val_image_dir, depth_dir=self.val_depth_dir, 
                                            ground_mask_dir=self.val_ground_mask_dir, mask_dir=self.val_slot_mask_dir, height=self.height, width=self.width, is_train=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, 
                                                num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
        self.val_iter = iter(self.val_loader)

        # Define dataset for getting prediction  
        self.train_full_set = ObjectDetectionDataset(file_lists=self.train_files, image_dir=self.train_image_dir, depth_dir=self.train_depth_dir, 
                                            ground_mask_dir=self.train_ground_mask_dir, mask_dir=None, height=self.height, width=self.width, is_train=False)
        self.train_full_loader = torch.utils.data.DataLoader(self.train_full_set, batch_size=self.batch_size, shuffle=False, 
                                                num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)
        self.val_full_set = ObjectDetectionDataset(file_lists=self.val_files, image_dir=self.val_image_dir, depth_dir=self.val_depth_dir, 
                                            ground_mask_dir=self.val_ground_mask_dir, mask_dir=None, height=self.height, width=self.width, is_train=False)
        self.val_full_loader = torch.utils.data.DataLoader(self.val_full_set, batch_size=self.batch_size, shuffle=False, 
                                                num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    def generate_bounding_box(self, mask):
        # Find the coordinates of the bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        # Create a bounding box mask
        bbox_mask = np.zeros_like(mask)
        bbox_mask[ymin:ymax+1, xmin:xmax+1] = 1
        return bbox_mask    

    def process_slot_maks(self, split):
        flow_neg_dir = f'{self.flow_neg_dir}/{split}'
        flow_pos_dir = f'{self.flow_pos_dir}/{split}'
        predicted_ground_mask_dir = f'{self.predicted_ground_mask_dir}/{split}'
        slot_attention_mask_dir = f'{self.slot_attention_mask_dir}/{split}'
        if split=='train': file_list = self.train_files 
        elif split=='val': file_list = self.val_files 
        else: assert False, 'Invalid split. Either train or val is accepted.'
        for file in tqdm(file_list):
            city_id, frame_id = file.split(" ")
            # if frame_id != 'darmstadt_000000_000016': continue
            # Read ground mask
            predicted_ground_mask = plt.imread(f'{predicted_ground_mask_dir}/{city_id}/{frame_id}.png')
            predicted_ground_mask = (predicted_ground_mask>0.5).astype(float) 
            # Read predicted dynamic mask
            predicted_flow_neg = np.load(f'{flow_neg_dir}/{city_id}/{frame_id}.npy')
            predicted_flow_pos = np.load(f'{flow_pos_dir}/{city_id}/{frame_id}.npy')
            predicted_dynamic_mask = (np.sum(np.abs(predicted_flow_neg) + np.abs(predicted_flow_pos), axis=0)/2 >= self.flow_threshold).astype(float)
            # Read the predicted object mask
            slot_mask = plt.imread(f'{slot_attention_mask_dir}/{city_id}/{frame_id}.png')
            slot_mask_list =  np.stack(np.split(slot_mask, axis=-1, indices_or_sections=5))
            slot_mask_list = slot_mask_list[np.flip(np.argsort(np.sum(slot_mask_list.reshape(5,-1),axis=-1)))]
            slot_mask_list = slot_mask_list[1:]
            # Process each mask
            retrieved_object_mask = []
            for slot_mask in slot_mask_list:
                slot_mask = slot_mask * (1-predicted_ground_mask) * predicted_dynamic_mask
                if np.sum(slot_mask) == 0: continue
                try: slot_mask_bbox = generate_bounding_box(slot_mask)
                except: continue
                # Skip if too large
                if np.sum(slot_mask_bbox) >= 0.5 * self.height * self.width: continue
                # Skip if do not intersect with the ground
                if np.sum(slot_mask_bbox*predicted_ground_mask) == 0: continue
                # Skip if ratio between mask if its bb is too small (they are expected to be quite similar)
                if np.sum(slot_mask) / np.sum(slot_mask_bbox) <= 0.5: continue
                object_mask = slot_mask_bbox * (1-predicted_ground_mask) * predicted_dynamic_mask
                # Skip if too small
                if np.sum(object_mask) <= self.object_size_threshold * self.height * self.width: continue
                object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((5,5)))
                object_mask = object_mask * (1-predicted_ground_mask) * predicted_dynamic_mask
                retrieved_object_mask.append(object_mask)
                # assert False
            if len(retrieved_object_mask) == 0: continue
            retrieved_object_mask = np.stack(retrieved_object_mask)
            retrieved_object_mask = retrieved_object_mask[np.flip(np.argsort(np.sum(retrieved_object_mask.reshape(len(retrieved_object_mask),-1),axis=-1)))]

            directory = f'{self.log_path}/processed_slot_mask/{split}/{city_id}/{frame_id}'
            os.makedirs(directory, exist_ok=True) 
            for i in range(len(retrieved_object_mask)):
                save_file_path  = os.path.join(directory, f"{i}.png")
                imageio.imsave(save_file_path, retrieved_object_mask[i])

    # 1. Remove wrong masks which has more than 1 object
    def remove_multiple_object_masks(self, mask_list, iou_threshold=0.2):
        processed_mask_list = []
        for mask in mask_list:
            bbox_mask = self.generate_bounding_box(mask)
            wrong_region = (bbox_mask != mask).astype(float)
            if np.sum(wrong_region) / np.sum(mask) >= iou_threshold: 
                continue
            processed_mask_list.append(mask)
        processed_mask_list = np.array(processed_mask_list) # N H W
        return processed_mask_list

    def get_unique_masks(self, masks, unique_threshold=0.25, connected_threshold=0.5):
        def get_area(mask):
            return np.sum(mask)
        def iou(target_mask, reference_mask):
            intersection = np.sum((target_mask==1) & (reference_mask==1))
            return intersection / np.sum(target_mask)
        extended_masks_list = []
        for i in range(len(masks)):
            iou_list = np.array([iou(masks[i], masks[j]) for j in range(len(masks))])
            iou_list[i] = 0
            if np.max(iou_list) >= connected_threshold:
                largest_masks_idx = np.where(iou_list >= unique_threshold)[0]
                largest_masks = masks[largest_masks_idx]
                largest_largest_masks_idx = np.argmax(np.sum(largest_masks.reshape(len(largest_masks),-1), axis=-1))
                masks[largest_masks_idx[largest_largest_masks_idx]] = np.clip(masks[i] + largest_masks[largest_largest_masks_idx],0,1)
                masks[i] = largest_masks[largest_largest_masks_idx]
            extended_masks_list.append(masks[i])
        extended_masks_list = np.stack(extended_masks_list)
        unique_masks_list = []
        for i in range(len(extended_masks_list)):
            iou_list = np.array([iou(extended_masks_list[i], extended_masks_list[j]) for j in range(len(extended_masks_list))])
            iou_list[i] = 0
            if np.max(iou_list) >= unique_threshold:
                largest_masks_idx = np.where(iou_list >= unique_threshold)[0]
                largest_largest_masks_idx = np.argmax(np.sum(extended_masks_list[largest_masks_idx].reshape(len(extended_masks_list[largest_masks_idx]),-1), axis=-1).reshape(-1))
                largest_masks = extended_masks_list[largest_masks_idx[largest_largest_masks_idx]]
                extended_masks_list[i] = largest_masks
            unique_masks_list.append(extended_masks_list[i])

        unique_masks_list = [mask for mask in unique_masks_list if np.sum(mask) != 0]
        unique_masks_list = np.unique(unique_masks_list, axis=0)
        return np.array(unique_masks_list)

    def mask_out_road(self, object_mask_list, road_mask):
        object_mask_list = object_mask_list*(1-np.expand_dims(road_mask, axis=0))
        return object_mask_list

    def process_predicted_mask(self, mask_list, road_mask, iou_threshold=0.1, unique_threshold=0.25, connected_threshold=0.5):
        if len(mask_list) == 1: return mask_list
        processed_mask = self.remove_multiple_object_masks(mask_list, iou_threshold)
        if len(processed_mask) == 0: return processed_mask
        processed_mask = self.get_unique_masks(processed_mask, unique_threshold, connected_threshold)
        processed_mask = self.mask_out_road(processed_mask, road_mask)
        return processed_mask

    def process_pred(self, pred, road_mask, frame_id, score_threshold, mask_size_threshold, processed_mask_epochs, log_path, epoch, split):
        pred_masks = pred["masks"].detach().cpu()[pred["scores"] > score_threshold]
        pred_masks = (pred_masks.detach().cpu().numpy() >= 0.5).astype(float)  # N H W
        pred_masks = [mask[0] for mask in pred_masks if np.sum(mask) >= mask_size_threshold*mask.shape[0]*mask.shape[1]]
        if len(pred_masks) == 0: return
        pred_masks = np.stack(pred_masks)
        # Save data            
        city_id = frame_id.split("_")[0]
        if epoch in processed_mask_epochs:
            if len(pred_masks) != 1:
                try:
                    pred_masks = self.process_predicted_mask(pred_masks, road_mask[0].numpy(), iou_threshold=0.1, unique_threshold=0.25, connected_threshold=0.5)
                except: pass
            if len(pred_masks) == 0: return
        pred_masks = torch.from_numpy(pred_masks)
        pred_masks = [mask for mask in pred_masks if torch.sum(mask) >= mask_size_threshold*mask.shape[0]*mask.shape[1]]
        if len(pred_masks) == 0: return
        folder_path = f'{log_path}/predictions/epoch_{epoch}/{split}/{city_id}/{frame_id}'
        # if not os.path.exists(folder_path): 
        os.makedirs(folder_path, exist_ok=True)

        for i, mask in enumerate(pred_masks):
            imageio.imsave(f'{folder_path}/{i}.png', mask)


    def combine_predictions(self, pred_list):
        combined_predictions = []
        for i in range(len(pred_list[0])):
            combined_pred_dict = {}
            for key in pred_list[0][0].keys():
                combined_pred_dict[key] = torch.cat([pred[i][key] for pred in pred_list], dim=0)
            combined_predictions.append(combined_pred_dict)
        return combined_predictions

    def store_predictions(self, dataloader, split, epoch):
        score_threshold = self.self_supervise_score_threshold_dict[epoch]
        with_augmentation = epoch in self.mask_augmentation_epochs
        print(f'Mask preprocessing: {epoch in self.processed_mask_epochs}, Mask augmentation: {with_augmentation}')
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            for batch_idx, (images, targets, road_mask, frame_id) in enumerate(tqdm(dataloader)):
                if with_augmentation:
                    images = torch.stack(list(image[[0,1,2,4]].cuda() for image in images)) # Contain images and ground mask
                    predictions_list = []
                    for a in range(len(self.mask_augmentator.augmentor_list)):
                        augmented_image = self.mask_augmentator.augmentor_list[a](images, scale=0.8)
                        disp = 1/self.depth_net(augmented_image[:,:3])[0]
                        disp = disp/torch.max(disp)
                        input_list = torch.concat([augmented_image[:,:3],disp,augmented_image[:,[-1]]],dim=1)
                        input_list = [input for input in input_list]
                        predictions = self.model(input_list)
                        predicted_mask = [self.mask_augmentator.inverse_augmentor_list[a](predictions[p]['masks'], scale=0.8, original_height=disp.shape[2], original_width=disp.shape[3]) for p in range(len(predictions))]
                        for i in range(len(predictions)): predictions[i]['masks'] = predicted_mask[i]
                        # input_list = torch.stack(input_list)
                        # input_list = [inverse_augmentor_list[a](input_list, scale=0.8, original_height=disp.shape[2], original_width=disp.shape[3]) for p in range(len(input_list))]
                        predictions_list.append(predictions)
                    predictions = self.combine_predictions(predictions_list)
                    predictions = [{k: v.detach().cpu() for k, v in t.items()} for t in predictions]
                    # os.makedirs(f'{self.log_path}/predictions/epoch', exist_ok=True)
                else:
                    images = list(image.cuda() for image in images)
                    # targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                    predictions = self.model(images)
                    predictions = [{k: v.detach().cpu() for k, v in t.items()} for t in self.model(images)]
                os.makedirs(f'{self.log_path}/predictions/epoch_{epoch}', exist_ok=True)
                # for p,pred in enumerate(predictions):
                #     self.process_pred(pred, road_mask[p], frame_id[p], score_threshold, self.mask_size_threshold, self.processed_mask_epochs, self.log_path, epoch, split)
                Parallel(n_jobs=16)(delayed(process_pred)(pred, road_mask[p], frame_id[p], score_threshold, self.mask_size_threshold, self.processed_mask_epochs, self.log_path, epoch, split) for p,pred in enumerate(predictions))
        self.model.train()

    def generate_new_dataset(self, epoch):
        self.train_mask_dir = f'{self.log_path}/predictions/epoch_{epoch}/train'
        self.train_mask_files = np.array(get_file_list(self.train_mask_dir))
        self.train_set = ObjectDetectionDataset(file_lists=self.train_mask_files, image_dir=self.train_image_dir, mask_dir=self.train_mask_dir, 
                                            ground_mask_dir=self.train_ground_mask_dir, depth_dir=self.train_depth_dir, height=self.height, width=self.width, is_train=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, 
                                                num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

        self.val_mask_dir = f'{self.log_path}/predictions/epoch_{epoch}/val'
        self.val_mask_files = np.array(get_file_list(self.val_mask_dir))
        self.val_set = ObjectDetectionDataset(file_lists=self.val_mask_files, image_dir=self.val_image_dir, mask_dir=self.val_mask_dir, 
                                            ground_mask_dir=self.val_ground_mask_dir, depth_dir=self.val_depth_dir, height=self.height, width=self.width, is_train=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, 
                                                num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
        self.val_iter = iter(self.val_loader)


    def combine_predictions(self, pred_list):
        combined_predictions = []
        for i in range(len(pred_list[0])):
            combined_pred_dict = {}
            for key in pred_list[0][0].keys():
                combined_pred_dict[key] = torch.cat([pred[i][key] for pred in pred_list], dim=0)
            combined_predictions.append(combined_pred_dict)
        return combined_predictions

    def process_batch(self, samples):
        images, targets, road_mask, frame_id = samples
        images = torch.stack(images).float().cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        return images, targets, loss_dict

    def train(self):
        self.global_step = (self.current_epoch * (len(self.train_set)//self.batch_size+1))
        print(f"Start training at epoch: {self.current_epoch}, global step: {self.global_step}")
        print()

        for epoch in range(self.current_epoch, self.num_epochs):
            print(f" --------------- Epoch {epoch} --------------- ", )
            self.model.train()
            epoch_train_loss = 0
            for samples in tqdm(self.train_loader):
                images, targets, loss_dict = self.process_batch(samples)
                losses = sum(loss for loss in loss_dict.values())
                epoch_train_loss += losses * len(images)
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if self.global_step % self.eval_step == 0:
                    self.model.eval()
                    with torch.no_grad():
                        train_predictions = self.model(images)
                        try: val_samples = next(self.val_iter)
                        except:
                            self.val_iter = iter(self.val_loader)
                            val_samples = next(self.val_iter)
                        val_images, val_targets, val_road_mask, val_frame_id = val_samples
                        val_predictions = self.model(val_images)
                        train_outputs = {'images': images, 'targets' : targets, 'predictions': train_predictions}
                        val_outputs = {'images': val_images, 'targets' : val_targets, 'predictions': val_predictions}
                        # TODO: Logging
                        log(train_outputs, val_outputs, 0.8, self.writers, self.global_step)
                        self.writers['train'].add_scalar("loss", losses.item(), self.global_step)
                    self.model.train()
                self.global_step += 1
            
            # # Epoch evaluation
            with torch.no_grad():
                epoch_val_loss = 0
                for val_samples in tqdm(self.val_loader):
                    val_images, val_targets, val_loss_dict = self.process_batch(val_samples)
                    val_losses = sum(loss for loss in val_loss_dict.values())
                    epoch_val_loss += val_losses * len(val_images)

            # # Log epoch
            epoch_train_loss = epoch_train_loss/(len(self.train_loader)*self.batch_size)
            epoch_val_loss = epoch_val_loss/(len(self.val_loader)*self.batch_size)
            log_epoch(epoch_train_loss, epoch_val_loss, self.writers, epoch)
            self.lr_scheduler.step()
            save_model([self.model, self.lr_scheduler], ['maskrcnn', 'scheduler'], self.optimizer, self.log_path, epoch)

            # # After each epoch, store the predictions of the model and use it to train the next model
            if self.do_self_supervise and epoch in list(self.self_supervise_score_threshold_dict.keys()):
                print(f"Start stroring predictions at epoch {epoch}")
                # Store prediction
                self.store_predictions(dataloader=self.train_full_loader, split='train', epoch=epoch)
                self.store_predictions(dataloader=self.val_full_loader, split='val', epoch=epoch)
                # Get new training dataset and data loaders
                self.generate_new_dataset(epoch)
                

