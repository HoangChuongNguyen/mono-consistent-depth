import numpy as np 
import torch
from tqdm import tqdm
import os
import cv2
from matplotlib import pyplot as plt


def gerenate_bounding_box_mask(bbox_2d, height, width):
    center_x, center_y, box_length, box_width = bbox_2d
    # Create an empty mask image
    mask = np.zeros((height, width),dtype=np.uint8)
    top_left_x = int(center_x-box_length/2)
    top_left_y = int(center_y-box_width/2)
    bottom_right_x = int(center_x+box_length/2)
    bottom_right_y = int(center_y+box_width/2)
    # Set pixels inside the bbox to 1
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
    return mask


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 


def compute_errors_all_dataset(pred_disp_list, test_loader, gt_depth_path, gt_mask_path):
    
    # Start evaluation
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    all_region_errors = []
    all_region_ratios = []
    static_errors = []
    # static_ratios = []
    dynamic_errors = []
    # dynamic_i = []
    # dynamic_ratios = []
    # gt_depths = os.path.join(splits_dir, "cityscapes", "val_gt_depths")
    for i in tqdm(range(len(pred_disp_list))):
        # city_id, full_frame_id = test_loader.dataset.filenames[i].split(" ")
        split_file = test_loader.dataset.filenames[i].split(" ")
        city_id, full_frame_id = split_file[0], split_file[1]
        # city_id, seq_id, frame_id = full_frame_id.split("_")

        if 'cityscapes' in type(test_loader.dataset).__name__.lower(): 
            # try:
            if os.path.isdir(f"{gt_mask_path}/{city_id}/{full_frame_id}"):
                mask_files_list = sorted(os.listdir(f"{gt_mask_path}/{city_id}/{full_frame_id}"))
                if 'label.txt' in mask_files_list:
                    labels = np.loadtxt(f"{gt_mask_path}/{city_id}/{full_frame_id}/label.txt", dtype=object).reshape(-1,)
                    mask_files_list.remove("label.txt")
                if 'labels.txt' in mask_files_list:
                    labels = np.loadtxt(f"{gt_mask_path}/{city_id}/{full_frame_id}/labels.txt", dtype=object).reshape(-1,)
                    mask_files_list.remove("labels.txt")
                mask_list = []
                for j,mask_file in enumerate(mask_files_list):
                    mask = plt.imread(f"{gt_mask_path}/{city_id}/{full_frame_id}/{j}.png")
                    # if labels[j] != "car": continue
                    if np.sum(mask) >= 0.005 * 1024 * 2048:
                        mask_list.append(mask)
                if len(mask_list) == 0:
                    object_mask = np.zeros((1024, 2048))
                else:
                    object_mask = np.clip(np.sum(np.stack(mask_list, axis=0), axis=0),0,1)
            else:
                object_mask = np.zeros((1024, 2048))
            # gt_depth = np.load(os.path.join(gt_depth_path, str(i).zfill(3) + '_depth.npy'))
            gt_depth = np.load(f'{gt_depth_path}/{full_frame_id}.npy')
            gt_height, gt_width = gt_depth.shape[:2]
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
            object_mask = object_mask[:gt_height]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = np.squeeze(pred_disp_list[i])
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]
            # mask = mask[256:, 192:1856]
            all_region_mask = np.ones_like(object_mask)[256:, 192:1856]
            static_mask = (1 - object_mask)[256:, 192:1856]
            dynamic_mask = object_mask[256:, 192:1856]

        elif 'waymo' in type(test_loader.dataset).__name__.lower(): 
            # Load gt depth
            gt_depth = np.load(f'{gt_depth_path}/{city_id}/{full_frame_id}.npy')
            # Crop out the top 30% of the original depth map
            gt_depth = gt_depth[1280-896:] # 896 1920 
            gt_height, gt_width = gt_depth.shape
            # Load object masks stats
            object_dynamic_label_list = np.loadtxt(f'{gt_mask_path}/{city_id}/{full_frame_id}/object_dynamic_label.txt', delimiter='\n',dtype=object).reshape(-1)
            object_visibility_label_list = np.loadtxt(f'{gt_mask_path}/{city_id}/{full_frame_id}/object_visibility.txt', delimiter='\n',dtype=object).reshape(-1)
            object_label_list = np.loadtxt(f'{gt_mask_path}/{city_id}/{full_frame_id}/labels.txt', delimiter='\n',dtype=object).reshape(-1)
            object_2d_bbox = np.loadtxt(f'{gt_mask_path}/{city_id}/{full_frame_id}/bbox_2d.txt', delimiter=',').reshape(-1,4)
            # Get object dynamic bbox 
            is_dynamic_object =  (object_dynamic_label_list=='dynamic') & (object_visibility_label_list=='visible') & (object_label_list!='SIGN')
            dynamic_object_2d_bbox = object_2d_bbox[is_dynamic_object]
            # Here we only keep 80% of the gt bbox. The reason is because gt bbox is larger than the actual object size in the image
            dynamic_object_2d_bbox[:,2:] = dynamic_object_2d_bbox[:,2:] * 0.9
            if len(dynamic_object_2d_bbox) == 0: 
                dynamic_mask = np.zeros((gt_height, gt_width))
            else:
                dynamic_object_2d_bbox_mask_list = np.stack([gerenate_bounding_box_mask(bbox_2d, 1280, 1920) for bbox_2d in dynamic_object_2d_bbox]) # n 1280 1920
                # Crop out the top 30% of the original object masks
                dynamic_object_2d_bbox_mask_list = dynamic_object_2d_bbox_mask_list[:,1280-896:]
                # Filter out objects with very small size
                dynamic_object_2d_bbox_mask_list = [object_bbox_mask for object_bbox_mask in dynamic_object_2d_bbox_mask_list if np.sum(object_bbox_mask) >= 0.005*896*1280]
                # If there is no moving objects with valid object size
                if len(dynamic_object_2d_bbox_mask_list) == 0: 
                    dynamic_mask = np.zeros((gt_height, gt_width))
                else:
                    dynamic_object_2d_bbox_mask_list = np.stack(dynamic_object_2d_bbox_mask_list)
                    dynamic_mask = np.clip(np.sum(dynamic_object_2d_bbox_mask_list, axis=0),0,1)
            all_region_mask = np.ones_like(dynamic_mask)
            static_mask = (1 - dynamic_mask)
            # Get the predicted depth
            pred_disp = np.squeeze(pred_disp_list[i])
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp
        
        elif 'kitti' in type(test_loader.dataset).__name__.lower(): 
            raise NotImplementedError


        all_region_mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH) & (all_region_mask==1)
        static_mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH) & (static_mask==1)
        dynamic_mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH) & (dynamic_mask==1)

        all_region_pred_depth = pred_depth[all_region_mask]
        all_region_gt_depth = gt_depth[all_region_mask]
        static_pred_depth = pred_depth[static_mask]
        static_gt_depth = gt_depth[static_mask]
        dynamic_pred_depth = pred_depth[dynamic_mask]
        dynamic_gt_depth = gt_depth[dynamic_mask]

        # Use the same scale for all regions
        if len(all_region_pred_depth) != 0: 
            # Median scale
            all_region_ratio = np.median(all_region_gt_depth) / np.median(all_region_pred_depth)
            all_region_ratios.append(all_region_ratio)
            all_region_pred_depth *= all_region_ratio
            all_region_pred_depth[all_region_pred_depth < MIN_DEPTH] = MIN_DEPTH
            all_region_pred_depth[all_region_pred_depth > MAX_DEPTH] = MAX_DEPTH
            all_region_errors.append(compute_errors(all_region_gt_depth, all_region_pred_depth))
        if len(static_pred_depth) != 0: 
            # Median scale
            static_pred_depth *= all_region_ratio
            static_pred_depth[static_pred_depth < MIN_DEPTH] = MIN_DEPTH
            static_pred_depth[static_pred_depth > MAX_DEPTH] = MAX_DEPTH
            static_errors.append(compute_errors(static_gt_depth, static_pred_depth))
        if len(dynamic_pred_depth) != 0: 
            # Median scale
            dynamic_pred_depth *= all_region_ratio
            dynamic_pred_depth[dynamic_pred_depth < MIN_DEPTH] = MIN_DEPTH
            dynamic_pred_depth[dynamic_pred_depth > MAX_DEPTH] = MAX_DEPTH
            dynamic_errors.append(compute_errors(dynamic_gt_depth, dynamic_pred_depth))

    all_region_mean_errors = np.array(all_region_errors).mean(0)
    static_mean_errors = np.array(static_errors).mean(0)
    dynamic_mean_errors = np.array(dynamic_errors).mean(0)
    return all_region_mean_errors, static_mean_errors, dynamic_mean_errors
    

def compute_object_errors_all_dataset(object_predicted_frame_id_list, object_predicted_disp_list, test_dynamic_mask_loader, gt_depth_path):
    # Start evaluation
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    dynamic_errors = []
    for k in tqdm(range(len(object_predicted_frame_id_list))):
        full_frame_id = object_predicted_frame_id_list[k]
        # if full_frame_id != 'munster_000132_000019': continue
        city_id, seq_id, frame_id = full_frame_id.split("_")

        index = test_dynamic_mask_loader.dataset.file_names.index(f'{city_id} {full_frame_id}')
        # plt.imshow(object_mask)
        # plt.show()
        # assert False
        # print(os.path.join(gt_depth_path, str(i).zfill(3) + '_depth.npy'))
        # gt_depth = np.load(os.path.join(val_gt_depth_path, str(index).zfill(3) + '_depth.npy'))
        gt_depth = np.load(f'{gt_depth_path}/{full_frame_id}.npy')
        gt_height, gt_width = gt_depth.shape[:2]
        # crop ground truth to remove ego car -> this has happened in the dataloader for input

        # when evaluating cityscapes, we centre crop to the middle 50% of the image.
        # Bottom 25% has already been removed - so crop the sides and the top here
        # crop ground truth to remove ego car -> this has happened in the dataloader for input
        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]
        gt_depth = cv2.resize(gt_depth, (512, 192), interpolation=cv2.INTER_NEAREST)
        # object_mask = object_mask[:gt_height]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = np.squeeze(object_predicted_disp_list[k])
        # pred_disp = cv2.resize(pred_disp, (gt_width, gt_height), interpolation=cv2.INTER_NEAREST)
        # object_mask = (pred_disp) 
        object_mask = (pred_disp!=0).astype(float)
        pred_depth = (1 / pred_disp) * object_mask

        gt_depth = gt_depth[64:, 48:464]
        pred_depth = pred_depth[64:, 48:464]
        # mask = mask[256:, 192:1856]

        dynamic_mask = object_mask[64:, 48:464]
        dynamic_mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH) & (dynamic_mask==1)
        if np.sum(dynamic_mask) == 0: continue


        dynamic_pred_depth = pred_depth[dynamic_mask]
        dynamic_gt_depth = gt_depth[dynamic_mask]

        dynamic_ratio = np.median(dynamic_gt_depth) / np.median(dynamic_pred_depth)
        dynamic_pred_depth *= dynamic_ratio
        dynamic_pred_depth[dynamic_pred_depth < MIN_DEPTH] = MIN_DEPTH
        dynamic_pred_depth[dynamic_pred_depth > MAX_DEPTH] = MAX_DEPTH
        dynamic_errors.append(compute_errors(dynamic_gt_depth, dynamic_pred_depth))
    return np.array(dynamic_errors).mean(0)


