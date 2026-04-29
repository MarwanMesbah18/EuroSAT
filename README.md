# EuroSAT Land Use Classification

Land use and land cover classification of Sentinel-2 satellite image patches using deep learning. Given a 64x64 RGB satellite image, the model predicts one of 10 land use classes.

## Dataset

The [EuroSAT](https://github.com/phelber/EuroSAT) dataset contains 27,000 labeled Sentinel-2 satellite images across 10 classes:

| Class | Images |
|---|---|
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |
| **Total** | **27,000** |

Images are 64x64 pixel RGB JPEGs derived from Sentinel-2 multispectral satellite data.

## Model Architecture

ResNet50 implemented from scratch — every layer built manually using `nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`. No pretrained weights.

**Bottleneck block** (3-layer residual block):
```
1x1 Conv → BN → ReLU    (reduce channels)
3x3 Conv → BN → ReLU    (spatial features)
1x1 Conv → BN            (expand channels x4)
+ skip connection        → ReLU
```

**Full network:**
```
Stem:  Conv2d(3→64, 7x7) → BN → ReLU → MaxPool(3x3)
Layer1: 3 Bottleneck blocks  (64 → 256 channels)
Layer2: 4 Bottleneck blocks  (128 → 512 channels)
Layer3: 6 Bottleneck blocks  (256 → 1024 channels)
Layer4: 3 Bottleneck blocks  (512 → 2048 channels)
Head:   AdaptiveAvgPool → Linear(2048, 10)
```

Weights initialized with Kaiming initialization. Input images resized from 64x64 to 224x224.

## Setup

### Kaggle (Recommended)

1. Upload the `EuroSAT_RGB` folder as a Kaggle dataset
2. Create a new notebook and attach a GPU accelerator (T4)
3. Copy the notebook from `notebooks/eurosat_resnet50.ipynb`
4. Set `DATA_DIR` to your Kaggle dataset path (e.g., `/kaggle/input/eurosat-rgb/EuroSAT_RGB/`)
5. Run all cells

Training completes in under 1 hour on a T4 GPU with mixed precision.

### Local

```bash
pip install -r requirements.txt
```

Set `DATA_DIR` in the notebook to `./Dataset/EuroSAT_RGB/`.

## Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Epochs | 100 |
| Weight decay | 1e-4 |
| Mixed precision | Yes (AMP) |
| Split ratio | 70 / 15 / 15 (train/val/test) |
| Early stopping | Patience = 12 |

## Data Augmentation

**Training:** RandomResizedCrop(224), RandomHorizontalFlip, RandomVerticalFlip (valid for satellite imagery), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02), ImageNet normalization.

**Validation/Test:** Resize(256), CenterCrop(224), ImageNet normalization.

## Results

Results will be filled in after training.

| Metric | Value |
|---|---|
| Overall Accuracy | — |
| Macro F1-Score | — |
| Weighted F1-Score | — |

Expected accuracy range: 90–95% when training from scratch.

## Project Structure

```
EuroSAT/
├── Dataset/
│   └── EuroSAT_RGB/            # 10 class folders with images
├── notebooks/
│   └── eurosat_resnet50.ipynb  # Kaggle training notebook
├── models/                      # Saved model checkpoints
├── outputs/                     # Plots and reports
├── README.md
└── requirements.txt
```

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217-2226. [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)
