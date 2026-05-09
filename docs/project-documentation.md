# EuroSAT Project — Complete Documentation

Consolidated from: `notebooks/notebook_explanation.md`, `training_loop_explained.md`, `model_architecture_explained.md`, `Reviews/review_1.md`, and `docs/superpowers/specs/2026-05-08-presentation-and-demo-design.md`.

---

# Part 1: Notebook Cell-by-Cell Explanation (V3)

This section explains every cell in the EuroSAT ResNet50 notebook with diagrams and visual walkthroughs.

---

## Cell 0: Title (Markdown)

The notebook title and a one-line description. It tells us: we're classifying Sentinel-2 satellite images into 10 land use categories using ResNet50 built from scratch (no pretrained weights).

---

## Cell 1: Section Header — "Configuration" (Markdown)

Just a section divider. No code.

---

## Cell 2: Configuration + Seeds

This cell does three things: **imports**, **settings**, and **reproducibility**.

### Imports

```python
import os, random, numpy as np, torch
```

| Library | What it does |
|---|---|
| `os` | File paths (find dataset, create output folder) |
| `random` | Python's random number generator (seeded for reproducibility) |
| `numpy` | Array math (used for splitting dataset indices) |
| `torch` | PyTorch — the deep learning framework |

### Paths

```
DATA_DIR = "/kaggle/input/.../EuroSAT_RGB"   ← where images live
OUTPUT_DIR = "/kaggle/working"                ← where to save results
```

On Kaggle, data is read-only (input), results are saved to `/kaggle/working`.

### Hyperparameters

```
BATCH_SIZE    = 64       ← images processed together in one step
NUM_EPOCHS    = 100      ← max passes through entire dataset
LEARNING_RATE = 3e-4     ← how fast the model updates weights (0.0003)
WEIGHT_DECAY  = 1e-4     ← regularization to prevent overfitting
NUM_CLASSES   = 10       ← 10 land use categories
SEED          = 42       ← random seed for reproducibility
TRAIN/VAL/TEST = 70/15/15% split
PATIENCE      = 8        ← stop if no improvement for 8 epochs
```

### Device + AMP

```
DEVICE = "cuda" (GPU) or "cpu"
USE_AMP = True (only on GPU)
```

**AMP (Automatic Mixed Precision)** uses float16 for some calculations instead of float32. This:
- Cuts GPU memory usage ~50%
- Speeds up training ~30-50%
- Keeps accuracy the same (the important parts still use float32)

### Seed Function

```python
def seed_everything(seed):
    random.seed(seed)           # Python random
    np.random.seed(seed)        # NumPy random
    torch.manual_seed(seed)     # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
```

Why? So you get the **same results** every time you run the notebook. The last line (`benchmark = True`) lets cuDNN auto-tune for the fixed 224x224 input size, which is safe and faster.

---

## Cell 3: Section Header — "Imports" (Markdown)

---

## Cell 4: Remaining Imports

```python
import copy, time, matplotlib, seaborn, pandas, tqdm
import torch.nn, torch.optim, torchvision.transforms, torchvision.datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
```

| Library | Role |
|---|---|
| `torch.nn` | Neural network layers (Conv2d, Linear, BatchNorm, etc.) |
| `torch.optim` | Optimizers (AdamW) and schedulers (CosineAnnealingLR) |
| `torchvision.transforms` | Image preprocessing (resize, crop, flip, normalize) |
| `torchvision.datasets` | Auto-loads images from folder structure |
| `sklearn.metrics` | Evaluation (accuracy, F1, confusion matrix) |
| `matplotlib / seaborn` | Plotting (training curves, heatmaps) |
| `tqdm` | Progress bars during training |
| `copy` | Deep copy for saving best model weights |
| `time` | Timing each epoch |

---

## Cell 5: Section Header — "Data Loading and Preprocessing" (Markdown)

---

## Cell 6: Compute Normalization + Define Transforms

### Part 1: Compute Dataset-Specific Mean and Std

Instead of using ImageNet's mean/std, V3 computes them from the actual EuroSAT data:

```
For every image in the dataset:
    Resize to 256 → CenterCrop to 224 → ToTensor
    Sum up all pixel values per channel
    Sum up all squared pixel values per channel

Then: mean = total_sum / num_pixels
      std  = sqrt(total_sq_sum / num_pixels - mean²)
```

Result:
```
Mean: [0.345, 0.381, 0.409]    ← EuroSAT images are darker than ImageNet
Std:  [0.201, 0.136, 0.114]    ← EuroSAT has less pixel variation
```

vs ImageNet values (used in V2/V4/V5):
```
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

**Why normalize?** Neural networks work best when input values are centered around 0 with similar scale. Without normalization, pixel values range [0, 1] after ToTensor, but the model converges faster and more stably when values are ~[-2, 2].

```
Before normalization:           After normalization:
    [0, 0, 0] ──┐                   [-1.7, -1.9, -2.0] ──┐
 Pixels         │                   Centered              │
 [0.5, 0.5, 0.5]─┤─ range [0, 1]   [0.0, 0.0, 0.0]  ─────┤─ range ~[-2, 2]
                 │                                         │
    [1, 1, 1] ──┘                   [1.7, 1.5, 1.3]  ─────┘
```

### Part 2: Training Augmentations

```python
train_transform = transforms.Compose([
    transforms.Resize(256),                          # ①
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # ②
    transforms.RandomHorizontalFlip(p=0.5),          # ③
    transforms.RandomVerticalFlip(p=0.5),            # ④
    transforms.RandomRotation(15),                   # ⑤
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1, hue=0.02),  # ⑥
    transforms.ToTensor(),                           # ⑦
    transforms.Normalize(mean=..., std=...),         # ⑧
])
```

**Why these augmentations for satellite imagery?**
- **VerticalFlip**: Satellites look straight down — a forest is still a forest upside down. This is NOT true for photos of people/cars.
- **HorizontalFlip**: Same reasoning — land looks the same mirrored.
- **Rotation(15°)**: Satellite angles vary slightly. Small rotation makes the model robust.
- **ColorJitter**: Atmospheric conditions change color slightly. The model should rely on texture/shape, not exact color.
- **RandomResizedCrop**: Simulates different zoom levels and positions.

### Part 3: Validation/Test Transforms

```python
val_transform = transforms.Compose([
    transforms.Resize(256),        # same resize
    transforms.CenterCrop(224),    # deterministic center crop (no randomness)
    transforms.ToTensor(),
    transforms.Normalize(mean=..., std=...),
])
```

No augmentation for validation — we want **consistent, reproducible** evaluation.

---

## Cell 7: Load Dataset + Create Splits

### How ImageFolder Works

The dataset folder structure tells PyTorch the class labels automatically:

```
Dataset/EuroSAT_RGB/
├── AnnualCrop/
│   ├── AnnualCrop_1.jpg    ← class label = "AnnualCrop" (auto-detected from folder name)
│   └── ...
├── Forest/
│   ├── Forest_1.jpg        ← class label = "Forest"
│   └── ...
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
    └── ...
```

### Train/Val/Test Split

```
Total: 27,000 images
├── Train: 18,900 (70%)  ← model learns from these
├── Val:    4,050 (15%)  ← model is evaluated during training (for early stopping)
└── Test:   4,050 (15%)  ← final evaluation (never seen during training)
```

---

## Cells 8-10: Dataset Visualization

- **Cell 9:** Class distribution bar chart — shows dataset balance
- **Cell 10:** Sample images — 5 per class in a 10×5 grid

---

## Cells 11-12: Augmentation Visualization

Shows the effect of each augmentation (Horizontal Flip, Vertical Flip, Rotation, Color Jitter, Random Crop) on a sample image.

---

## Cells 13-16: Model Definition

### The Bottleneck Block

The building block of ResNet50. Each block has **3 convolutions** + a **skip connection**:

```
                    Input (256 channels)
                    ┌───────┐
                    │       │
                    │  (copy for skip connection)
                    │       │
                    ▼       │
            ┌─────────────┐│
            │ 1×1 Conv    ││  ← squeeze: 256 → 64 channels (reduce by 4×)
            │ BatchNorm   ││     cheaper computation
            │ ReLU        ││
            └──────┬──────┘│
                   ▼       │
            ┌─────────────┐│
            │ 3×3 Conv    ││  ← learn: 64 → 64 channels
            │ BatchNorm   ││     this is where the real feature extraction happens
            │ ReLU        ││
            └──────┬──────┘│
                   ▼       │
            ┌─────────────┐│
            │ 1×1 Conv    ││  ← expand: 64 → 256 channels (restore by 4×)
            │ BatchNorm   ││
            └──────┬──────┘│
                   │       │
                   ▼       ▼
                   ┌───┐
                   │ + │    ← ADD skip connection to output
                   └─┬─┘
                     ▼
                   ┌─────────┐
                   │  ReLU   │
                   └────┬────┘
                        ▼
                   Output (256 channels)
```

### ResNet50 Full Architecture

```
Input image (3 × 224 × 224)
        │
        ▼
┌─────────────────── Stem ───────────────────────┐
│  7×7 Conv(3→64, stride=2) → BN → ReLU          │
│  3×3 MaxPool(stride=2)                          │
│  Output: 64 × 56 × 56                          │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 1: 3 Bottleneck blocks │
        │     64 → 256 channels            │
        │     Spatial: 56 × 56             │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 2: 4 Bottleneck blocks │
        │     256 → 512 channels           │
        │     Spatial: 28 × 28 (stride=2)  │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 3: 6 Bottleneck blocks │
        │     512 → 1024 channels          │
        │     Spatial: 14 × 14 (stride=2)  │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 4: 3 Bottleneck blocks │
        │     1024 → 2048 channels         │
        │     Spatial: 7 × 7 (stride=2)    │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │  AdaptiveAvgPool2d(1, 1)         │
        │  Flatten: 2048                   │
        │  Linear(2048 → 10)               │
        │  Output: 10 class scores         │
        └──────────────────────────────────┘
```

### Spatial Resolution Flow

```
224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7 → 1×1
```

### Channel Flow

```
3 → 64 (stem) → 256 (layer1) → 512 (layer2) → 1024 (layer3) → 2048 (layer4) → 10 (FC)
```

### Layer Count: Why "50"?

```
Stem conv:        1
Layer1: 3×3 =    9
Layer2: 4×3 =   12
Layer3: 6×3 =   18
Layer4: 3×3 =    9
FC:              1
                ────
Total:          50 layers with learnable weights
```

### Weight Initialization

```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

- **Kaiming/He init for conv layers**: Sets weights so the variance of outputs matches the variance of inputs. Without this, deep networks would have activations that explode or vanish.
- **Constant 1/0 for BatchNorm**: weight=1, bias=0 means "initially, just pass through the normalized values unchanged."

23,528,522 parameters — all trainable, no freezing.

---

## Cells 17-18: Optimizer, Scheduler, and Loss

### Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

Measures how wrong the model's predictions are. For each image:
1. Model outputs 10 raw scores (logits)
2. Softmax converts to probabilities
3. Cross-entropy compares predicted probability vs true label

### Optimizer

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

**AdamW** combines:
- **Adam**: Adapts learning rate per parameter (fast convergence)
- **Weight decay (1e-4)**: Penalizes large weights to prevent overfitting (decoupled from Adam's adaptive LR)

### Scheduler

```python
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

Gradually reduces the learning rate following a cosine curve:

```
Learning Rate
3e-4 │╲
     │ ╲
     │  ╲
     │   ╲
     │    ╲
     │     ╲
     │      ╲
     │       ╲
     │        ╲
1e-6 │         ╲___________________
     └────────────────────────────────
     Epoch 0                       100
```

**Why?** Start fast (big steps to explore), end slow (small steps to converge precisely).

### V5 Additions

V5 adds:
- **Label Smoothing (0.1):** `CrossEntropyLoss(label_smoothing=0.1)` — distributes 0.1 probability across all classes, prevents overconfidence
- **CutMix augmentation:** Cuts a patch from one image and pastes it onto another, mixing labels proportionally

---

## Cells 19-20: Training Loop

Each epoch:

```
┌─────────────────────────── EPOCH ──────────────────────────────┐
│                                                                │
│  TRAINING PHASE:                                               │
│  for each batch (64 images):                                   │
│    optimizer.zero_grad()       ← clear old gradients           │
│    predictions = model(images) ← forward pass (with AMP)       │
│    loss = criterion(predictions, labels)                        │
│    scaler.scale(loss).backward()  ← compute gradients          │
│    scaler.step(optimizer)         ← update weights             │
│    scaler.update()                ← adjust AMP scale           │
│                                                                │
│  VALIDATION PHASE:                                             │
│    model.eval()                                                │
│    with torch.no_grad():                                       │
│      compute val_loss, val_acc                                  │
│                                                                │
│  BOOKKEEPING:                                                  │
│    If val_acc > best: save model, reset patience               │
│    Else: patience_counter += 1                                 │
│    If patience >= 8: STOP (early stopping)                     │
│                                                                │
│  scheduler.step()  ← reduce learning rate                      │
└────────────────────────────────────────────────────────────────┘
```

### Early Stopping

```
Epoch 40: val_acc = 97.0% → New best! Save model. Patience reset to 0
Epoch 41: val_acc = 96.8% → Not best.    Patience = 1
...
Epoch 48: val_acc = 96.3% → Not best.    Patience = 8 → STOP!
```

---

## Cells 21-22: Training Curves

Three subplots:
1. **Loss curve**: Train loss vs Val loss over epochs — both should decrease
2. **Accuracy curve**: Train acc vs Val acc over epochs — both should increase
3. **Learning rate schedule**: Shows the cosine decay from 3e-4 to 1e-6

If train loss keeps dropping but val loss rises → overfitting.

---

## Cells 23-24: Test Evaluation

Loads the best model (saved during training) and runs it on the **test set** — images the model has never seen.

**V5 Results (with TTA):**
```
Overall:  98.32% accuracy, 98.29% macro F1

Per class:
  SeaLake:             99.58%  ← best
  Forest:              98.88%
  Residential:         98.14%
  Highway:             99.20%
  Industrial:          98.95%
  Pasture:             98.51%
  River:               97.86%
  HerbaceousVegetation:97.94%
  AnnualCrop:          98.63%
  PermanentCrop:       95.29%  ← hardest class
```

### V5 Test-Time Augmentation (TTA)

Creates 5 augmented views per test image and averages predictions:
- Original
- Horizontal flip
- Vertical flip
- +5° rotation
- -5° rotation

---

## Cells 25-26: Confusion Matrix

Two heatmaps:
- **Raw counts**: How many images from class X were predicted as class Y
- **Normalized (percentages)**: Same but as row percentages

---

## Cells 27-28: Per-Class Accuracy

Color-coded bars:
- Green (>= 97%): Strong performance
- Orange (93-97%): Decent but room to improve
- Red (< 93%): Needs attention

---

## Cells 29-30: Sample Predictions

Shows 32 test images (4×8 grid) with:
- **T:** = True label
- **P:** = Predicted label
- Green title = correct prediction
- Red title = wrong prediction

---

## Quick Reference: Image Size Pipeline

```
Original EuroSAT image:  64 × 64 pixels (RGB)
         │
         ▼ Resize(256)
       256 × 256
         │
         ▼ RandomResizedCrop(224) [train] or CenterCrop(224) [val/test]
       224 × 224
         │
         ▼ ToTensor()  →  float tensor [3, 224, 224], values in [0, 1]
         │
         ▼ Normalize(mean, std)  →  centered tensor, values ~[-2, 2]
         │
         ▼
    Model Input: [batch_size, 3, 224, 224]
```

Why 224? ResNet50 needs spatial reduction through 5 stride-2 operations:
```
224 → 112 → 56 → 28 → 14 → 7 → 1 (global avg pool)
```

---

---

# Part 2: Code Review (Review 1)

### Critical Issues (from initial review)

1. **Misleading "transfer learning" description** — Both notebooks claimed "transfer learning with ResNet50" but the model is built from scratch. Fixed in V4+ to say "ResNet50 implemented from scratch."

2. **ImageNet normalization on satellite data** — Initially flagged as an issue (compute EuroSAT-specific stats). However, after testing in V3, ImageNet normalization actually performed better (see V3 vs V1/V5 results). Wider dynamic range helps class separation.

3. **Training/test data overlap risk** — Ensure the 70/15/15 split has no overlap between test and train. Fixed with proper random seed + subset sampling.

### Improvements Made (V1 → V5)

- V3: Added dataset-specific normalization (later reverted)
- V4: Better code structure, augmentation visualization section
- V5: Added CutMix augmentation, Label Smoothing, Test-Time Augmentation (TTA)

---

---

# Part 3: Presentation & Demo Design Spec

## Context
The user needs to present their EuroSAT ResNet50 satellite image classification project to their professor. They want an all-in-one, browser-based solution that doesn't depend on PowerPoint or specific software.

## Deliverables

### 1. HTML Slideshow (`presentation/presentation_slides.html`)
- **Format:** reveal.js HTML slideshow generated from the notebook
- **Navigation:** Arrow keys, fullscreen with F key

### 2. Scrollable HTML (`presentation/presentation_scroll.html`)
- **Format:** Standard nbconvert HTML export — one long scrollable page
- **CSS extracted** to `presentation/scroll.css` for easy editing

### 3. Streamlit Demo App (`demo.py`)
- **Framework:** Streamlit
- **Model loading:** Loads `models/best_resnet50_eurosat.pth`
- **Features:** Image upload, test image gallery, dataset samples, confidence table
- **Launch:** `streamlit run demo.py`

## Presentation Workflow
1. Open `presentation/presentation_scroll.html` in browser
2. Scroll through the presentation
3. When ready for demo, switch to Streamlit (already running): `streamlit run demo.py`
4. Upload satellite images or use provided samples
5. Show live predictions

## Files Structure
- `notebooks/eurosat_resnet50_presentation.ipynb` — generated presentation notebook
- `presentation/presentation_scroll.html` + `scroll.css` — scrollable presentation
- `presentation/presentation_slides.html` + `slides.css` — slideshow version
- `scripts/build_presentation.py` — generates the presentation notebook from V5
- `scripts/extract_css.py` — extracts inline CSS from HTML exports
- `demo.py` — Streamlit demo app
- `test_images/` — test images for the demo (rename to test_1, test_2, etc.)
- `photos/` — architecture diagrams and comparison images
