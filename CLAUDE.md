# EuroSAT Land Use Classification

## Project
- ResNet50 built from scratch (no pretrained weights), manual layer-by-layer implementation
- Dataset: EuroSAT_RGB, 10 classes, ~27K images in Dataset/EuroSAT_RGB/
- Notebooks: V2 (97.65%), V3 (97.58%), V4 (experiments), V5 (98.32% with TTA)
- Explanation docs: notebooks/notebook_explanation.md, model_architecture_explained.md, training_loop_explained.md
- User is preparing for a presentation/discussion on this project

## Training Environment
- Primary: Kaggle T4 x2, batch_size=64, DATA_DIR="/kaggle/input/datasets/marwanmesbah19/eurosat-rgb-1/EuroSAT_RGB"
- Local GPU is 3.6GB VRAM — too small for batch_size>8 with ResNet50 at 224x224
- No PyTorch installed locally — can only verify syntax, not run training
- Kaggle Persistence setting: Files Only (prevents losing saved models on timeout)
- Keep Kaggle browser tab active during training to avoid session disconnect

## Architecture
- Bottleneck blocks (1x1→3x3→1x1 + skip connection), 3+4+6+3 blocks
- Single FC head: Linear(2048, 10)
- Kaiming init for conv, constant init for batchnorm
- ~23.5M parameters
- Original images are 64×64, resized to 224×224 for ResNet50 compatibility

## Hyperparameters
- V2: epochs=100, lr=3e-4, optimizer=AdamW, scheduler=CosineAnnealingLR, patience=12, ImageNet normalization
- V5: same base + label_smoothing=0.1, CutMix(alpha=1.0, prob=0.5), TTA(5 views), patience=12
- Mixed precision (AMP) enabled
- Augmentation: satellite-specific (vertical flips, rotation, color jitter)

## Key Findings
- ImageNet normalization works better than dataset-specific for this dataset (wider range helps separate hard classes)
- Using both WeightedRandomSampler AND weighted loss causes severe overfitting — use one or the other
- Weighted loss alone doesn't help much when imbalance is mild (~1.5x) — V2 without it is still best baseline
- PermanentCrop is the hardest class — confused with AnnualCrop and HerbaceousVegetation
- CutMix causes temporary loss spikes but improves final accuracy significantly
- Label smoothing lowers training accuracy but improves generalization

## Results
| Version | Val Acc | Test Acc | Key Change |
|---|---|---|---|
| V2 | 97.65% | 97.65% | Baseline, ImageNet norm, patience=12 |
| V3 | 97.58% | 97.58% | Dataset-specific normalization |
| V4 | 97.31% | 97.14% | Weighted loss (hurt more than helped) |
| V5 | 98.54% | 98.32% (TTA) | CutMix + label smoothing + TTA |
- Each epoch ~120s on Kaggle T4
