# <p align=center>Ultrasound SAM Adapter: Adapting SAM for Breast Lesion Segmentation in Ultrasound Images</p>

<p align=center>Zhengzheng Tu, Le Gu, Xixi Wang, Bo Jiang, and Jin Tang</p>

## Introduction
This repository is the official implementation for "Ultrasound SAM Adapter: Adapting SAM for Breast Lesion Segmentation in Ultrasound Images".

![image](BUSSAM.png)

## Abstract
Segment Anything Model (SAM) has recently achieved amazing results in the field of natural image segmentation. However, it is not effective for medical image segmentation, owing to the large domain gap between natural and medical images. In this paper, we mainly focus on ultrasound image segmentation. As we know that it is very difficult to train a foundation model for ultrasound image data due to the lack of large-scale annotated ultrasound image data. To address these issues, in this paper, we develop a novel Breast Ultrasound SAM Adapter, termed Breast Ultrasound Segment Anything Model (BUSSAM), which migrates the SAM to the field of breast ultrasound image segmentation by using the adapter technique. To be specific, we first design a novel CNN image encoder, which is fully trained on the BUS dataset. Our CNN image encoder is more lightweight, and focuses more on features of local receptive field, which provides the complementary information to the ViT branch in SAM. Then, we design a novel Cross-Branch Adapter to allow the CNN image encoder to fully interact with the ViT image encoder in SAM module. 
Finally, we add both of the Position Adapter and the Feature Adapter to the ViT branch to fine-tune the original SAM. The experimental results on AMUBUS and BUSI datasets demonstrate that our proposed model outperforms other medical image segmentation models significantly. 
Our code is available at: https://github.com/bscs12/BUSSAM.

## Getting Started
### Prepare Environment
1. Clone repository
```
git clone https://github.com/bscs12/BUSSAM.git
cd BUSSAM
```

2. Create environment
```
conda create -n BUSSAM python=3.8
conda activate BUSSAM
```

3. Install PyTorch

```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu111
```
or
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

4. Install other pakages
```
pip install -r requirements.txt
```

### Prepare Checkpoint
You can download the pretrained ```vit_b``` checkpoint of SAM from [https://github.com/facebookresearch/segment-anything] or [https://pan.baidu.com/s/1winNpGFv4z-gBqsfK5UoaA?pwd=cheh] [Password: ```cheh```].

### Prepare Dataset
1. Download dataset

You can download the AMUBUS dataset and BUSI dataset from [https://pan.baidu.com/s/1winNpGFv4z-gBqsfK5UoaA?pwd=cheh] [Password: ```cheh```].

2. Prepare your own dataset

Make sure the dataset folder structure like this:
```
datasets
    ├── AMUBUS
    │   ├── img
    │   │   ├── xxx.png
    │   │   ├── xxx.png
    │   │   ├── ...
    │   ├── label
    │   │   ├── xxx.png
    │   │   ├── xxx.png
    │   │   ├── ...
    ├── BUSI
    │   ├── img
    │   │   ├── xxx.png
    │   │   ├── xxx.png
    │   │   ├── ...
    │   ├── label
    │   │   ├── xxx.png
    │   │   ├── xxx.png
    │   │   ├── ...
    ├── MainPatient
    │   ├── AMUBUS_train.txt
    │   ├── AMUBUS_test.txt
    │   ├── AMUBUS_val.txt
    │   ├── BUSI_train.txt
    │   ├── BUSI_test.txt
    │   ├── BUSI_val.txt
    │   ├── class.json
```
### Train
```
python train.py
```
### Test
```
python test.py
```
## Cite
If you find this work useful for your, please consider citing our paper. Thank you!
