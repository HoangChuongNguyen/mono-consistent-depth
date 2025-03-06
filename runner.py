
import argparse
import yaml
from trainer.train_pixel_wise import PixelWiseTrainer
from trainer.train_ground_segmentation import GroundSegmentationTrainer
from trainer.train_slot_attention import SlotAttentionTrainer
from trainer.train_object_detection import ObjectDetectionTrainer
from trainer.train_object_depthnet import ObjectDepthTrainer
from trainer.train_depth_scale_alignment import DepthScaleAlignmentTrainer
from trainer.train_final_depthnet import FinalDepthTrainer
import torch

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config', type=str, help='Config file path')
    args = parser.parse_args()

    cfg = load_config('configs/default.yaml')
    cfg.update(load_config(args.config))

    # if not cfg['use_pretrained_mask']: cfg['log_path'] = cfg['log_path'] + "_predMask"

    stage_count = 1

    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print(f"--- STAGE {stage_count}: SCENE-DEPTH TRAINING WITH PIXEL-WISE MOTION ----")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    pixel_wise_trainer = PixelWiseTrainer(cfg)
    # 1.1. Main training loop
    pixel_wise_trainer.train()
    # 1.2. Use the pretrained model to predict and store prediction in hard disk for later use
    # Note: This requires large storage !!!!
    pixel_wise_trainer.get_prediction()
    print("\n\n\n")
    stage_count += 1

    if not cfg['use_pretrained_mask']:

        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        print(f"--- STAGE {stage_count}: SLOT ATTENTION ----")
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        slot_attention_trainer = SlotAttentionTrainer(cfg=cfg)
        slot_attention_trainer.train()
        slot_attention_trainer.get_prediction()
        del slot_attention_trainer
        torch.cuda.empty_cache()
        print("\n\n\n")
        stage_count += 1


        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        print(f"--- STAGE {stage_count}: GROUND SEGMENTATION ----")
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        ground_segmentation_trainer = GroundSegmentationTrainer(cfg=cfg)
        ground_segmentation_trainer.train()
        ground_segmentation_trainer.get_prediction()
        del ground_segmentation_trainer
        torch.cuda.empty_cache()
        print("\n\n\n")
        stage_count += 1


        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        print(f"--- STAGE {stage_count}: OBJECT DETECTION ----")
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        object_detection_trainer = ObjectDetectionTrainer(cfg=cfg)
        if object_detection_trainer.load_weights_folder is None:
            # Slot mask pre-processing
            for split in ['train', 'val']:
                object_detection_trainer.process_slot_maks(split=split)
            object_detection_trainer.dataset_setup()
        object_detection_trainer.train()
        last_object_detection_epoch = object_detection_trainer.num_epochs - 1
        del object_detection_trainer
        torch.cuda.empty_cache()
        print("\n\n\n")
        stage_count += 1

        project_path = cfg['log_path']
        cfg["raw_mask_path"] = f"{project_path}/object_detection/predictions/epoch_{last_object_detection_epoch}"
        cfg["raw_road_mask_path"] = f"{project_path}/ground_segmentation/predictions"

    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print(f"-- STAGE {stage_count}: OBJECT-DEPTH TRAINING WITH OBJECT RIGID MOTION --")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    object_depth_trainer = ObjectDepthTrainer(cfg, pixel_wise_trainer)
    # # 2.1. Mask pre-processing
    for split in ['train', 'val']:
        static_dynamic_label_dict = object_depth_trainer.process_and_classify_object_mask(split)
        object_depth_trainer.extract_training_object_file(static_dynamic_label_dict, split)
        object_depth_trainer.retrieve_corresponding_mask(split)
        object_depth_trainer.get_valid_train_mask_file(split)
    # # 2.2. Main training loop
    object_depth_trainer.training_setup()
    object_depth_trainer.train()
    # Empty cuda cache
    del object_depth_trainer
    del pixel_wise_trainer
    torch.cuda.empty_cache()
    print("\n\n\n")
    stage_count += 1



    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print(f"------ STAGE {stage_count}: DEPTH-SCALE-ALIGNMENT MODULE TRAINING -------")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    DSA_trainer = DepthScaleAlignmentTrainer(cfg)
    # 3.1. Pre-process data
    for split in ['train', 'val']:
        DSA_trainer.extract_eval_dynamic_object_mask(split)
    # 3.2. Main training loop
    # DSA_trainer.create_evaluation_dataset()
    DSA_trainer.train()
    # 3.3. Get pseudo depth label
    for split in ['train', 'val']:
        DSA_trainer.get_pseudo_depth(split)
    # Empty cuda cache
    del DSA_trainer
    torch.cuda.empty_cache()
    print("\n\n\n")
    stage_count += 1


    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print(f"-------------- STAGE {stage_count}: FINAL DEPTHNET TRAINING -------------")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    final_depthnet_trainer = FinalDepthTrainer(cfg)
    # 4.1 Pre-processing
    for split in ['train', 'val']:
        final_depthnet_trainer.get_training_filenames(split)
    # 4.2 Main training loop
    final_depthnet_trainer.training_setup()
    final_depthnet_trainer.train()
    print("\n\n\n")
