
# [CVPR2024] Mining Supervision for Dynamic Regions in Self-Supervised Monocular Depth Estimation

Hoang Chuong Nguyen, Tianyu Wang, Jose M. Alvarez, Miaomiao Liu.

\[[Link to paper](https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_Mining_Supervision_for_Dynamic_Regions_in_Self-Supervised_Monocular_Depth_Estimation_CVPR_2024_paper.html)\]

We propose a framework to address this the depth scale ambiguity issue in dynamic region and mine reliable supervision for learning depth in dynamic region. Our solution is:

- ✅ **Scene Decomposition**: Decompose the scene into static regions and individual moving objects, whose depths are estimated independently.
- ✅ **Depth Scale Alignment (DSA) module**: Introduce a novel DSA module to solve the scale ambiguity among each dynamic object and the static background.
- ✅ **Pseudo Depth Label**: Use the scale-consistent depth produced by the DSA module as supervision to train a depth network. 

<p align="center">
  <a
href="https://www.youtube.com/watch?v=E4jPf_wCQvk&t=160s">
  <img src="assets/video_thumbnail.png" alt="5 minute CVPR presentation video link" width="400">
  </a>
</p>



## ⚙️ Setup
```bash
conda create --name mono_consistent_depth python=3.8
conda activate mono_consistent_depth
pip install -r requirements.txt
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
```

## 📦 Dataset

Download the pre-processed Cityscapes dataset (~250Gb) using the following commands. Unzip all zip files and place them under data/cityscapes

```bash
cd data && mkdirs cityscapes && cd cityscapes
gdown 19jRLWDr79UdAFT4fwmflXZHobFdr51Bm && unzip gt_depths.zip && rm -f gt_depths.zip
gdown 1Hw3ZrAvnV8qkfmONdud9vcB7HmmmH96v && unzip gt_dynamic_object_mask.zip && rm -f gt_dynamic_object_mask.zip
gdown 1K07r-eQkCPUQsn04E1eobA8Ew5UgzGDO && unzip SEEM_mask.zip && rm -f SEEM_mask.zip
gdown 1cZHOH4qI8IxDSnXjOHrv3ErEkukzVBXz && unzip leftImg8bit_sequence.zip && rm -f leftImg8bit_sequence.zip
gdown 1DfDSegD4fG2wm_rvcM9y_TPox-2uTSR6 && unzip cityscapes_val_512x1024.zip && rm -f cityscapes_val_512x1024.zip
gdown 1ynuPmMVigVfuNb3vKsscCi4JT__X2ufZ && unzip citiscapes_512x1024.zip && rm -f citiscapes_512x1024.zip
```

If the commands above do not work, the dataset can be manually downloaded using this [link](https://drive.google.com/drive/folders/1juAb0NPYKEsDGw5m-OZ8mZNGuJQHJNrZ?usp=drive_link).

## 👀 Reproducing Paper Results

Note that we store model's predictions in hard disks for dataloading. Thus, the training requires large storage (~250Gb).
To reproduce the results from our paper on CityScape dataset, run:

```bash
# For training using pre-train segmentation model
python runner.py ./config/cityscapes/diffnet_pretrained_mask.yaml
# For fully unsupervised training
python runner.py ./config/cityscapes/diffnet_pred_mask.yaml
```

To resume training at a particular stage, specify the model_load_path in the config file, and modify the runner.py file accordingly. 

For depth evaluation, run:. 

```bash
# For training using pre-train segmentation model
python depth_eval.py ./config/cityscapes/diffnet_pretrained_mask.yaml
# For fully unsupervised training
python depth_eval.py ./config/cityscapes/diffnet_pred_mask.yaml
```



When evaluate the model, remember to specify the pretrained model path (i.e. eval_model_load_path in the config file). Otherwise, the best model is automatically loaded. 

## 👀 Pre-trained Models
Our pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1-p6Bfa-6GQR3BirAhefE0UwjF5sHzrLF?usp=sharing). To evaluate these models, you need to specify eval_model_load_path in the config file. 

## ✏️ Citation

If you find our work useful or interesting, please consider citing our paper:

```latex
@inproceedings{nguyen2024mining,
  title={Mining Supervision for Dynamic Regions in Self-Supervised Monocular Depth Estimation},
  author={Nguyen, Hoang Chuong and Wang, Tianyu and Alvarez, Jose M and Liu, Miaomiao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10446--10455},
  year={2024}
}
```

## Acknowledgement

Parts of our code are inspired from [MonoDepth2](https://github.com/nianticlabs/monodepth2), [InstaDM](https://github.com/SeokjuLee/Insta-DM). We appreciate and thank the authors for providing their excellent code.

This github page template is from [ManyDepth](https://github.com/nianticlabs/manydepth/blob/master/README.md).
