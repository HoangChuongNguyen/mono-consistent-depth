

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

    if not cfg['use_pretrained_mask']: cfg['log_path'] = cfg['log_path'] + "_predMask"

    print("-------------------------------------------------------------")
    print("--------------- EVALUATE PIXEL-WISE DEPTHNET ----------------")
    print("-------------------------------------------------------------")
    trainer_pixel_wise = PixelWiseTrainer(cfg)
    pixelwise_pred_disp_list, pixelwise_depth_errors = trainer_pixel_wise.eval_depth()
    print("\n\n")

    print("-------------------------------------------------------------")
    print("------------------ EVALUATE FINAL DEPTHNET ------------------")
    print("-------------------------------------------------------------")
    final_depthnet_trainer = FinalDepthTrainer(cfg)
    final_pred_disp_list, final_depth_errors = final_depthnet_trainer.eval_depth()

    