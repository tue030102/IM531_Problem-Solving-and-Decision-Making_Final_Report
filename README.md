# EGE-UNet for Retinal Vessel Segmentation (Final Project)

This repository contains the source code and experimental results for the final course project.
The project applies the EGE-UNet architecture to the task of retinal vessel segmentation on the FIVES dataset.

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
## Final Results
See: `results/readme.md`  
(Outputs/plots are in `results/egeunet_fives_STR2_Tuesday_30_December_2025_02h_01m_28s/`)
   