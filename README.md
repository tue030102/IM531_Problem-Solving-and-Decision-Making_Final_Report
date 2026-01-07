# EGE-UNet
This is the official code repository for "EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation", which is accpeted by *26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI2023)* as a regular paper!

**0. Main Environments**
- python 3.8
- [pytorch 1.8.0](https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-win_amd64.whl)
- [torchvision 0.9.0](https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl)

**1. Prepare the dataset.**

- The FIVES datasets

- './data/fives/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

**2. Train the EGE-UNet.**
```
cd EGE-UNet
```
```
python train.py
```

**3. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
