# EuroSAT Land Use Classification

## Project
- ResNet50 built from scratch (no pretrained weights), manual layer-by-layer implementation
- Dataset: EuroSAT_RGB, 10 classes, ~27K images in Dataset/EuroSAT_RGB/
- Notebook: notebooks/eurosat_resnet50.ipynb

## Training Environment
- Primary: Kaggle T4 x2, batch_size=64, DATA_DIR="/kaggle/input/datasets/marwanmesbah19/eurosat-rgb-1/EuroSAT_RGB"
- Local GPU is 3.6GB VRAM — too small for batch_size>8 with ResNet50 at 224x224
- Kaggle Persistence setting: Files Only (prevents losing saved models on timeout)
- Keep Kaggle browser tab active during training to avoid session disconnect

## Architecture
- Bottleneck blocks (1x1→3x3→1x1 + skip connection), 3+4+6+3 blocks
- Single FC head: Linear(2048, 10)
- Kaiming init for conv, constant init for batchnorm
- ~23.5M parameters

## Hyperparameters
- epochs=100, lr=3e-4, optimizer=AdamW, scheduler=CosineAnnealingLR, patience=8
- Mixed precision (AMP) enabled
- Augmentation: satellite-specific (vertical flips, rotation, color jitter)

## Results
- Reached 97.65% val accuracy at epoch 50 on Kaggle T4
- Each epoch ~120s on Kaggle T4
