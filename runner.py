

import argparse
import yaml
from train_pixel_wise import PixelWiseTrainer
from train_object_depthnet import ObjectDepthTrainer
from train_depth_scale_alignment import DepthScaleAlignmentTrainer
from train_final_depthnet import FinalDepthTrainer
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

    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print("--- STAGE 1: SCENE-DEPTH TRAINING WITH PIXEL-WISE MOTION ----")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    trainer_pixel_wise = PixelWiseTrainer(cfg)
    # 1.1. Main training loop
    trainer_pixel_wise.train()
    # 1.2. Use the pretrained model to predict and store prediction in hard disk for later use
    # Note: This requires large storage !!!!
    trainer_pixel_wise.get_prediction()
    print("\n\n\n")


    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print("-- STAGE 2: OBJECT-DEPTH TRAINING WITH OBJECT RIGID MOTION --")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    object_depth_trainer = ObjectDepthTrainer(cfg, trainer_pixel_wise)
    # 2.1. Mask pre-processing
    for split in ['train', 'val']:
        static_dynamic_label_dict = object_depth_trainer.process_and_classify_object_mask(split)
        object_depth_trainer.extract_training_object_file(static_dynamic_label_dict, split)
        object_depth_trainer.retrieve_corresponding_mask(split)
        object_depth_trainer.get_valid_train_mask_file(split)
    # 2.2. Main training loop
    object_depth_trainer.training_setup()
    object_depth_trainer.train()
    # Empty cuda cache
    del object_depth_trainer
    torch.cuda.empty_cache()
    print("\n\n\n")


    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print("------ STAGE 3: DEPTH-SCALE-ALIGNMENT MODULE TRAINING -------")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    DSA_trainer = DepthScaleAlignmentTrainer(cfg)
    # 3.1. Pre-process data
    for split in ['train', 'val']:
        DSA_trainer.extract_eval_dynamic_object_mask(split)
    DSA_trainer.create_evaluation_dataset()
    # 3.2. Main training loop
    DSA_trainer.train()
    # 3.3. Get pseudo depth label
    for split in ['train', 'val']:
        DSA_trainer.get_pseudo_depth(split)
    # Empty cuda cache
    del DSA_trainer
    torch.cuda.empty_cache()
    print("\n\n\n")


    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print("-------------- STAGE 4: FINAL DEPTHNET TRAINING -------------")
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
