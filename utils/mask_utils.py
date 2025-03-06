
import os
import numpy as np
from matplotlib import pyplot as plt 
from joblib import Parallel, delayed
import imageio
from tqdm.auto import  tqdm
import cv2


def calculate_iou(mask1, mask2):
    intersection = np.sum((mask1==1) & (mask2==1))
    iou = intersection / (np.sum(mask1)+1e-5)
    return iou

def check_middle_mask(mask, k=5):
    height, width = mask.shape
    # Calculate the range of the section
    start_height, end_height = height * (k - 1) // (2 * k), height * (k + 1) // (2 * k)
    start_width, end_width = width * (k - 1) // (2 * k), width * (k + 1) // (2 * k)
    # Sum over the middle kth of the mask
    sum_section = np.sum(mask[:, start_width:end_width])
    # If the sum is greater than 0, there is at least one mask portion in the section
    return sum_section > 0

def extract_valid_mask(mask_list, static_dynamic_labels, mask_size_theshold, image_size):
    # A mask is valid if its size > threhold
    # and its label is either static or dynamic
    target_label_lists = ['static', 'dynamic']
    height, width = image_size
    mask_size_list = np.sum(mask_list.reshape(len(mask_list),-1), axis=-1)
    is_target_label = np.array([label in target_label_lists for label in static_dynamic_labels])
    # extracted_mask_list = mask_list[(mask_size_list>mask_size_theshold*192*512)&(is_target_label)]
    extracted_mask_idx = np.where((mask_size_list>mask_size_theshold*height*width)&(is_target_label))
    return extracted_mask_idx


def proprocess_and_classify_raw_mask(full_frame_id, mask_path, road_mask_path, predicted_flow_neg_path, predicted_flow_pos_path,
                  flow_threshold, is_dynamic_threshold, middle_portion, save_path, split):
    
    city_id, full_frame_id = full_frame_id.split(" ")
    # Get object mask
    if not os.path.isdir(f"{mask_path}/{split}/{city_id}/{full_frame_id}"):
        return full_frame_id, None

    if not os.path.isfile(f'{road_mask_path}/{split}/{city_id}/{full_frame_id}.png'):
        return full_frame_id, None

    predicted_road_mask = plt.imread(f'{road_mask_path}/{split}/{city_id}/{full_frame_id}.png')
    predicted_road_mask = (predicted_road_mask > 0.5).astype(float)
    mask_file_list = os.listdir(f"{mask_path}/{split}/{city_id}/{full_frame_id}")
    mask_file_list = sorted(mask_file_list, key=lambda x: int(x[:-4]))
    
    mask_list = [plt.imread(f"{mask_path}/{split}/{city_id}/{full_frame_id}/{mask_file}") for mask_file in mask_file_list]
    mask_list = np.stack(mask_list)  # Exclude the zero labels

    predicted_flow_neg = np.load(f'{predicted_flow_neg_path}/{split}/{city_id}/{full_frame_id}.npy')
    predicted_flow_pos = np.load(f'{predicted_flow_pos_path}/{split}/{city_id}/{full_frame_id}.npy')
    predicted_dynamic_mask = (np.sum(np.abs(predicted_flow_neg) + np.abs(predicted_flow_pos), axis=0) / 2 >= flow_threshold).astype(float)

    iou_list = np.array([calculate_iou(object_mask, predicted_dynamic_mask) for object_mask in mask_list])
    is_dynamic = iou_list >= is_dynamic_threshold
    is_middle_mask = [check_middle_mask(object_mask, k=middle_portion) for object_mask in mask_list]

    mask_list = mask_list * (1 - predicted_road_mask)

    for m in range(mask_list.shape[0]):
        os.makedirs(f'{save_path}/{split}/{city_id}/{full_frame_id}', exist_ok=True)
        imageio.imsave(f'{save_path}/{split}/{city_id}/{full_frame_id}/{m}.png', mask_list[m])

    is_dynamic_label = ["dynamic" if (is_dynamic[i] or is_middle_mask[i]) else 'static' for i in range(len(is_dynamic))]
    return full_frame_id, is_dynamic_label



def extract_valid_dynamic_object_mask(mask_list, mask_size_theshold, static_dynamic_labels, target_label_lists):
    mask_size_list = np.sum(mask_list.reshape(len(mask_list),-1), axis=-1)
    is_target_label = np.array([label in target_label_lists for label in static_dynamic_labels])
    # extracted_mask_list = mask_list[(mask_size_list>mask_size_theshold*192*512)&(is_target_label)]
    extracted_mask_idx_ = np.where((mask_size_list>mask_size_theshold*192*512)&(is_target_label))
    return extracted_mask_idx_
    
def extract_dynamic_object_filenames(static_dynamic_label_dict, depth_net_type, mask_path, road_mask_path, object_mask_size_theshold, split, save_dir):
    frame_id_list = sorted(list(static_dynamic_label_dict.keys()))
    extract_mask_file_id_list = []
    result_dict = {}
    target_label_lists = ['dynamic']
    for f,full_frame_id in enumerate(tqdm(frame_id_list)):
        city_id = full_frame_id.split("_")[0]
        if not os.path.isfile(f'{road_mask_path}/{city_id}/{full_frame_id}.png'): continue
        # Get object mask
        mask_file_list = os.listdir(f"{mask_path}/{city_id}/{full_frame_id}")
        if 'labels.txt' in mask_file_list: mask_file_list.remove('labels.txt')
        mask_file_list = sorted(mask_file_list, key=lambda x: int(x[:-4]))
        # if len(mask_file_list) == 1: continue # If there is no object continue
        # mask_file_list = mask_file_list[1:]
        mask_list =  [plt.imread(f"{mask_path}/{city_id}/{full_frame_id}/{mask_file}") for mask_file in mask_file_list]
        mask_list = np.stack(mask_list)
        # Extract samples with target mask_size, and dynamic labels
        static_dynamic_labels = static_dynamic_label_dict[full_frame_id]
        static_dynamic_labels = static_dynamic_labels
        assert len(static_dynamic_labels) == len(mask_list)
        extracted_mask_idx_ = extract_valid_dynamic_object_mask(mask_list, object_mask_size_theshold, static_dynamic_labels, target_label_lists)
        extracted_mask_idx = np.array(mask_file_list)[extracted_mask_idx_]
        extracted_mask_idx = np.append(-1, extracted_mask_idx.flatten())
        # Extract mask id so that it is like this -1,2,3,4. 
        # -1 mean empty masks. This is the placeholder for images for no moving objects
        extracted_mask_idx_str = (str([f"{int(mask_file_id.replace('.png',''))}" for mask_file_id in extracted_mask_idx])).replace(' ', '').replace('\'', '')[1:-1]
        extract_mask_file_id_list.append(f"{city_id} {full_frame_id} {extracted_mask_idx_str}")
    extract_mask_file_id_list = np.array(extract_mask_file_id_list)
    return extract_mask_file_id_list

def generate_bounding_box(mask):
    mask_ = np.copy(mask)
    eroded_mask = cv2.erode(mask_, np.ones((5,5)), iterations=5)
    dilated_mask = cv2.dilate(eroded_mask, np.ones((5,5)), iterations=5)
    mask = dilated_mask
    # find the coordinates where the mask is True
    y, x = np.where(mask)
    # find min and max coordinates
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    # create a new mask of same size and draw the bounding box
    bounding_mask = np.zeros_like(mask)
    bounding_mask[ymin:ymax+1, xmin:xmax+1] = 1
    return bounding_mask