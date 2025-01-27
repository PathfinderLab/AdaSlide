# AdaSlide Project Instructions

Before starting, note that the training and evaluation processes can be quite complex. If you encounter any difficulties, feel free to reach out for assistance. The AdaSlide framework requires the following modules:

## Required Modules
### 1. Compression Decision Agent (CDA)
- **CompressionDecisionAgent**: Identifies significant and insignificant image regions.

### 2. Foundation Image Enhancer (FIE)
- **BasicSR**: For image reconstruction using a super-resolution model.
- **VAE**: For image reconstruction using a Variational Autoencoder (VAE) model.
- **VQVAE**: For image reconstruction using a Vector Quantized VAE (VQVAE) model.

## General Tips for Running Commands
To ensure smooth execution, use the `nohup` command for running processes in the background. Some steps may take several hours or even days. For example:

**Foreground Computing**:
```bash
python ./my_code.py --arg1 arg1 --arg2 arg2
```

**Background Computing**:
```bash
nohup python ./my_code.py --arg1 arg1 --arg2 arg2 > log.out &
```
Replace log.out with your preferred output log filename.

## Step 1: WSI Preprocessing
Use the `./Preprocessing/patch_generation.ipynb` file to generate patch instances for training FIEs and CDAs. Ensure that the folder hierarchy and file paths are correctly structured.

This file generates multiple patch instances and splits them into training, validation, and test sets. For training AdaSlide, we employed 910 WSIs across 31 projects (approximately 30 slides per project).

The `LR-x4` folder contains low-resolution images (128x128), and the `HR` folder contains high-resolution images (512x512). Ensure that `HR` and `LR-x4` splits (train, valid, test) are aligned with each other.

## Step 2: Traing FIEs
For ESRGAN, only one model needs to be trained. However, for VAE and VQVAE, two models are required based on the input size: one for 224x224 and another for 512x512.

**VAE**
In the VAE folder, run:

```bash
python ./train.py --config config/vanilla_vae_224.yaml
python ./train.py --config config/vanilla_vae_512.yaml
```

Please modify the following content in the configuration file to match your dataset:
```yaml
data_params:
  train_path: "my_dataset_folder/CLAM_prepare/HR_train/*.png"
  valid_path: "my_dataset_folder/CLAM_prepare/HR_valid/*.png"
  test_path: "my_dataset_folder/CLAM_prepare/HR_test/*.png"
```

**VQVAE**
In the vq-vae-2-pytorch folder, run:

```bash
python ./train_vqvae.py --n_gpu 1 --size 512 --epoch 5 --path my_dataset_folder/CLAM_prepare/HR_train
python ./train_vqvae.py --n_gpu 1 --size 224 --epoch 5 --path my_dataset_folder/CLAM_prepare/HR_train
```

**ESRGAN**
To use pretrained model weights, download ESRGAN_SRx4.pth and place it in the BasicSR/experiments/pretrained_models directory. [Download the pretrained model weight here](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth).

In the `BasicSR` folder, run:

```bash
python ./basicsr/train.py -opt options/train/ESRGAN/train_ESRGAN_FIE.yml
```

And ESRGAN_x4_foundation, you need to match below variables:

```yaml
datasets:
  train:
    dataroot_gt: my_dataset_folder/CLAM_prepare/HR_train
    dataroot_lq: /project/kimlab_tcga/AdaSlide_dataset/LR-x4/train

  val:
    dataroot_gt: /project/kimlab_tcga/AdaSlide_dataset/HR/valid
    dataroot_lq: /project/kimlab_tcga/AdaSlide_dataset/LR-x4/valid
```

## Step 3: Generate pseudo mask for CDA training
For use hover_net, you need to make another environment for hover_net. To install hover_net, please take a look at [hover_net](https://github.com/vqdang/hover_net).

In `hover_net` folder, execute inference.

```bash
python run_infer.py --gpu 1 --nr_types 5 --type_info_path type_info.json --model_path weights/hovernet_original_kumar_notype_tf2pytorch.tar --model_mode original --batch_size 16 --nr_inference_workers 16 --nr_post_proc_workers 16  tile --input_dir /data/SR-Hist-Foundation/LR-x4_valid_up --output_dir /data/SR-Hist-Foundation/LR-x4_valid_up_hover
```

For Hover-Net, inference may not be executable for large-scale files. In this project, we divided the data into subsets of 1,000 files each, performed inference on each subset, and then combined the results.

## Step 4: Training CDA
Navigate to the CompressAgent folder and execute following command:

```bash
python ./run_hparam_search.py --config config/hparam_tuner_config.yaml
```

To select optimal CDA model for each labmda condition, based on test results, we manually select based on the performance and compression ratio balance. 

## Step 5: Inference and evaluation
Please see the [AdaSlide_demo](https://github.com/PathfinderLab/AdaSlide_demo/tree/main).