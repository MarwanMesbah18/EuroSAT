# EuroSAT ResNet50 — Cell-by-Cell Explanation (V3)

This document explains every cell in `eurosat_rgb_v3.ipynb` with diagrams and visual walkthroughs.

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

This cell does two big things.

### Part 1: Compute Dataset-Specific Mean and Std

Instead of using ImageNet's mean/std, this code **computes them from the actual EuroSAT data**:

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

vs ImageNet values (used in V2):
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

Here's what each does visually:

```
Original (64×64)         ① Resize to 256×256
┌───────────────┐        ┌─────────────────────────────────────┐
│               │   →    │                                     │
│   satellite   │        │           satellite                 │
│    image      │        │            image                    │
│               │        │                                     │
└───────────────┘        └─────────────────────────────────────┘

② RandomResizedCrop (224)                ③ HorizontalFlip (p=0.5)
┌───────────────────────┐               ┌───────────────────┐
│        ┌──────────┐   │               │                   │
│  ┌─────┤  crop    ├───┤───→           │    mirror ←→      │
│  │     └──────────┘   │               │                   │
└───────────────────────┘               └───────────────────┘
 Random position + scale (80-100%)       50% chance each batch

④ VerticalFlip (p=0.5)                  ⑤ Rotation (±15°)
┌───────────────────┐                   ╱───────────────────╲
│                   │                  ╱                     ╲
│    mirror ↕       │───→             │    rotated image     │
│                   │                  ╲                     ╱
└───────────────────┘                   ╲───────────────────╱
  Why OK? Satellites look down —         Random angle within ±15°
  there's no "up" or "down"

⑥ ColorJitter
┌───────────────────┐    ┌───────────────────┐
│                   │    │                   │
│    original       │───→│  shifted colors   │
│    colors         │    │  +brightness      │
│                   │    │  +contrast         │
└───────────────────┘    └───────────────────┘
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
│   ├── AnnualCrop_2.jpg
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

`ImageFolder` reads this structure and creates:
- `dataset[i]` → returns (image, label_index)
- `dataset.classes` → `['AnnualCrop', 'Forest', ..., 'SeaLake']`
- `dataset.targets` → list of all label indices

### Train/Val/Test Split

```
Total: 27,000 images
├── Train: 18,900 (70%)  ← model learns from these
├── Val:    4,050 (15%)  ← model is evaluated during training (for early stopping)
└── Test:   4,050 (15%)  ← final evaluation (never seen during training)
```

The split is done by shuffling all indices and slicing:

```
All 27,000 indices → shuffle → [idx, idx, idx, ...]
                         ├── first 18,900 → train
                         ├── next 4,050   → val
                         └── last 4,050   → test
```

### DataLoader

```
Dataset → SubsetRandomSampler (picks only the indices for this split)
        → DataLoader (batches of 64, loads in parallel with 2 workers)
        → GPU
```

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Image   │    │  Image   │    │  Image   │    │  Image   │
│  Folder  │ →  │ Sampler  │ →  │  Batch   │ →  │   GPU    │
│ (27,000) │    │ (subset) │    │  (64)    │    │ (train)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

Key DataLoader settings:
- `batch_size=64`: Process 64 images at once
- `num_workers=2`: Use 2 CPU threads to load images in parallel
- `pin_memory=True`: Faster CPU→GPU transfer

---

## Cell 8: Section Header — "Dataset Visualization" (Markdown)

---

## Cell 9: Class Distribution Bar Chart

Counts images per class and plots a bar chart. This shows whether the dataset is balanced.

```
Class counts:
  SeaLake:       3000  ████████████████████████████████
  Forest:        3000  ████████████████████████████████
  Residential:   3000  ████████████████████████████████
  AnnualCrop:    3000  ████████████████████████████████
  HerbaceousVegetation: 2949  ███████████████████████████████
  PermanentCrop:  2500  ██████████████████████████
  Highway:        2500  ██████████████████████████
  Industrial:     2500  ██████████████████████████
  Pasture:        2000  ████████████████████
  River:          2500  ██████████████████████████
```

The dataset is **roughly balanced** but not perfectly — Pasture has fewer images. This is why we might want imbalance handling.

---

## Cell 10: Sample Images Per Class

Shows 5 example images from each class in a grid (10 rows × 5 columns). This gives you a feel for what the satellite images look like and how similar/different the classes are.

Also defines `inv_normalize` — the inverse of normalization, so images can be displayed in normal colors:

```
Normalize does:    x_norm = (x - mean) / std
Inverse does:      x_orig = x_norm * std + mean
```

---

## Cell 11: Section Header — "Model Definition" (Markdown)

---

## Cell 12: ResNet50 Architecture (The Big One)

This is the core of the notebook. Let's break it down piece by piece.

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
                   ┌───┐ ┌───┐
                   │ + │ │ + │    ← ADD skip connection to output
                   └─┬─┘ └─┬─┘
                     ▼     ▼
                   ┌─────────┐
                   │  ReLU   │
                   └────┬────┘
                        ▼
                   Output (256 channels)
```

**Why this design?**
- 1×1 convs are **cheap** (few parameters) — they change channel count without spatial processing
- The 3×3 conv does the **heavy lifting** — but on only 64 channels instead of 256
- This saves ~4× computation vs using 3×3 convs on all 256 channels
- The **skip connection** (`out += identity`) lets gradients flow backward easily, solving the vanishing gradient problem in deep networks

**Channel flow through one Bottleneck:**
```
in: 256 → conv1: 64 → conv2: 64 → conv3: 256 → + skip(256) → out: 256
          ↓ squeeze      ↓ learn      ↓ expand
```

### The `downsample` Shortcut

When dimensions change (stride or channel count), the skip connection can't just copy the input — it needs to match the output dimensions. That's what `downsample` does:

```
Normal block (same dims):         Dimension-changing block (first block of each layer):
  Input 256ch → Block → 256ch      Input 64ch → Block → 256ch (stride=2)
       ↓                           ↓
  Skip: just copy input            Skip: 1×1 conv + BN to match 256ch + stride=2
       ↓                           ↓
       └── ADD ──┘                  └── ADD ──┘
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
        │     9 conv layers                │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 2: 4 Bottleneck blocks │
        │     256 → 512 channels           │
        │     Spatial: 28 × 28 (stride=2)  │
        │     12 conv layers               │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 3: 6 Bottleneck blocks │
        │     512 → 1024 channels          │
        │     Spatial: 14 × 14 (stride=2)  │
        │     18 conv layers               │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     Layer 4: 3 Bottleneck blocks │
        │     1024 → 2048 channels         │
        │     Spatial: 7 × 7 (stride=2)    │
        │     9 conv layers                │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │  AdaptiveAvgPool2d(1, 1)         │
        │  2048 × 7 × 7 → 2048 × 1 × 1   │
        │  (average each 7×7 map to 1×1)   │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │  Flatten: 2048                   │
        │  Linear(2048 → 10)               │
        │  Output: 10 class scores         │
        └──────────────────────────────────┘
```

### Spatial Resolution Flow

```
224×224  ──stem (stride 2)──→  112×112  ──maxpool──→  56×56
    ──layer1──→  56×56
    ──layer2──→  28×28
    ──layer3──→  14×14
    ──layer4──→   7×7
    ──avgpool──→  1×1
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

(BatchNorm, ReLU, and pooling layers don't count — they have no learned weights or are trivially initialized.)

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

### Model Creation

```python
model = ResNet50(num_classes=10).to(DEVICE)
```

23,528,522 parameters — all trainable, no freezing.

---

## Cell 13: Section Header — "Optimizer, Scheduler, and Loss" (Markdown)

---

## Cell 14: Loss, Optimizer, Scheduler, Scaler

### Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

Measures how wrong the model's predictions are. For each image:
1. Model outputs 10 raw scores (logits)
2. Softmax converts to probabilities
3. Cross-entropy compares predicted probability vs true label

```
True label: Forest (index 1)
Model output: [0.1, 0.8, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
                                   ↑ highest = correct!
Loss = -log(0.8) = 0.22 (low = good)

If model output: [0.4, 0.1, 0.3, ...]
                              ↑ wrong prediction
Loss = -log(0.1) = 2.30 (high = bad)
```

### Optimizer

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

**AdamW** updates the model's weights to minimize the loss. It combines:
- **Adam**: Adapts learning rate per parameter (fast convergence)
- **Weight decay (1e-4)**: Penalizes large weights to prevent overfitting

```
Weight update rule (simplified):
  weight = weight - lr × gradient + weight_decay × weight
           └── Adam adapts this ──┘  └── shrink weights slightly ──┘
```

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

### Gradient Scaler (AMP)

```python
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
```

Manages the float16 → float32 scaling during mixed precision training. Prevents gradient underflow (values too small for float16).

---

## Cell 15: Section Header — "Training Loop" (Markdown)

---

## Cell 16: The Training Loop

The heart of the notebook. Each epoch does:

```
┌─────────────────────────── EPOCH ──────────────────────────────┐
│                                                                │
│  TRAINING PHASE:                                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ for each batch (64 images):                            │    │
│  │                                                        │    │
│  │   images, labels ──→ GPU                               │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   optimizer.zero_grad()  ← clear old gradients        │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   Forward pass (with AMP autocast):                    │    │
│  │   images → model → predictions (logits)                │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   loss = criterion(predictions, true_labels)           │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   scaler.scale(loss).backward()  ← compute gradients   │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   scaler.step(optimizer)  ← update weights             │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   scaler.update()  ← adjust AMP scale                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
│  VALIDATION PHASE:                                             │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ model.eval()  ← switch to evaluation mode              │    │
│  │ with torch.no_grad():  ← don't compute gradients       │    │
│  │   for each batch:                                      │    │
│  │     images → model → predictions → loss + accuracy     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
│  BOOKKEEPING:                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Record: train_loss, train_acc, val_loss, val_acc, lr   │    │
│  │                                                        │    │
│  │ If val_acc > best ever:                                │    │
│  │   Save model checkpoint  ← "best model so far"        │    │
│  │   Reset patience counter                               │    │
│  │ Else:                                                  │    │
│  │   patience_counter += 1                                │    │
│  │   If patience_counter >= 8: STOP (early stopping)      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
│  scheduler.step()  ← reduce learning rate (cosine schedule)    │
└────────────────────────────────────────────────────────────────┘
```

### Early Stopping

```
Epoch 40: val_acc = 97.0% → New best! Save model. Patience reset to 0
Epoch 41: val_acc = 96.8% → Not best.    Patience = 1
Epoch 42: val_acc = 96.5% → Not best.    Patience = 2
...
Epoch 48: val_acc = 96.3% → Not best.    Patience = 8 → STOP!
```

We load the saved best model (from epoch 40) for testing. This prevents overfitting — if the model keeps getting worse on validation, we stop.

### AMP Training Flow

```
Standard (float32):                          Mixed Precision (AMP):
images (float32) → model → loss              images (float32)
      ↓                                           ↓ (autocast: float16 for some ops)
gradients (float32)                            model forward in float16
      ↓                                           ↓
weight update                                  loss (float32 — kept precise)
                                                     ↓
                                                gradients (scaled to avoid underflow)
                                                     ↓
                                                weight update (float32)
```

---

## Cell 17: Section Header — "Training Curves" (Markdown)

---

## Cell 18: Plot Training Curves

Three subplots:
1. **Loss curve**: Train loss vs Val loss over epochs — both should decrease
2. **Accuracy curve**: Train acc vs Val acc over epochs — both should increase
3. **Learning rate schedule**: Shows the cosine decay from 3e-4 to 1e-6

```
Loss                              Accuracy                      Learning Rate
│\ Val                            │       _┈┈┈┈┈ Val            │╲
│ \                               │    _/                     │ ╲
│  \ Train                        │ _/  Train                 │  ╲
│   \_                            │/                          │   ╲
│    \__                          │                            │    ╲__...___
└──────────                       └──────────                  └──────────────
 Epochs                            Epochs                       Epochs
```

If train loss keeps dropping but val loss rises → overfitting.

---

## Cell 19: Section Header — "Test Evaluation" (Markdown)

---

## Cell 20: Test Set Evaluation

Loads the best model (saved during training) and runs it on the **test set** — images the model has never seen.

```
Best model weights → Load into model → model.eval()

For each test batch:
  images → model → predictions

Compare predictions vs true labels → accuracy, F1, classification report
```

**V3 Results:**
```
Overall:  97.58% accuracy, 97.48% macro F1

Per class:
  SeaLake:             99.47%  ████████████████████████████████████████  ← best
  Forest:              99.44%  ████████████████████████████████████████
  Residential:         99.30%  ████████████████████████████████████████
  Highway:             97.60%  █████████████████████████████████████
  River:               97.61%  █████████████████████████████████████
  Industrial:          98.16%  ███████████████████████████████████████
  HerbaceousVegetation:96.35%  ████████████████████████████████████
  Pasture:             96.34%  ████████████████████████████████████
  AnnualCrop:          96.08%  ████████████████████████████████████
  PermanentCrop:       94.45%  ███████████████████████████████████    ← hardest class
```

**PermanentCrop** is the hardest — likely confused with HerbaceousVegetation and AnnualCrop (they look similar from above).

---

## Cell 21: Section Header — "Confusion Matrix" (Markdown)

---

## Cell 22: Plot Confusion Matrix

Two heatmaps:
- **Raw counts**: How many images from class X were predicted as class Y
- **Normalized (percentages)**: Same but as row percentages

```
                    Predicted
                 A    F    H    Hi   I    P    PC   R    Ri   S
              ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  AnnualCrop  │ 96│    │  2│    │    │    │  1│    │    │    │
  Forest      │    │ 99│    │    │    │    │    │    │    │    │
  Herb.       │  1│    │ 96│    │    │    │  2│    │    │    │
  Highway     │    │    │    │ 97│    │    │    │  1│  1│    │
T Industrial  │    │    │    │    │ 98│    │    │    │    │    │
r Pasture     │    │    │  1│    │    │ 96│  2│    │    │    │
u PermanentC  │  1│    │  3│    │    │  2│ 94│    │    │    │  ← most confused
e Residential │    │    │    │    │    │    │    │ 99│    │    │
  River       │    │    │    │  1│    │    │    │  1│ 97│    │
  SeaLake     │    │    │    │    │    │    │    │    │    │ 99│
              └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

Diagonal = correct predictions (should be bright)
Off-diagonal = mistakes (should be dark)
```

---

## Cell 23: Section Header — "Per-Class Accuracy" (Markdown)

---

## Cell 24: Per-Class Accuracy Bar Chart

Color-coded bars:
- Green (>= 97%): Strong performance
- Orange (93-97%): Decent but room to improve
- Red (< 93%): Needs attention

---

## Cell 25: Section Header — "Sample Predictions" (Markdown)

---

## Cell 26: Sample Prediction Grid

Shows 32 test images (4×8 grid) with:
- **T:** = True label
- **P:** = Predicted label
- Green title = correct prediction
- Red title = wrong prediction

This helps you visually understand what the model gets right and wrong.

---

## Cell 27: Section Header — "Save Final Artifacts" (Markdown)

---

## Cell 28: Save Everything

Saves three files:
1. **`resnet50_eurosat_best.pth`** — model weights only (smaller file)
2. **`classification_report.csv`** — per-class precision/recall/F1 as a table
3. **`training_history.csv`** — loss/accuracy/lr per epoch (for plotting later)

Plus the earlier checkpoint saved during training:
4. **`best_resnet50_eurosat.pth`** — full checkpoint (model + optimizer + epoch + accuracy) for resuming training

---

## Quick Reference: Image Size Pipeline

```
Original EuroSAT image:  64 × 64 pixels (RGB)
         │
         ▼ Resize(256)
       256 × 256
         │
         ▼ RandomResizedCrop(224)  [train]  or  CenterCrop(224)  [val/test]
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
With 64×64 input: `64 → 32 → 16 → 8 → 4 → 2 → 1` — too little spatial information (2×2 → 1×1).

---

## Quick Reference: Training Pipeline Summary

```
┌───────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────────┐
│  Dataset   │───→│  DataLoader  │───→│  Model   │───→│   Loss      │
│ 27K images │    │ batches of   │    │ ResNet50 │    │CrossEntropy │
│ 10 classes │    │ 64 images    │    │ 23.5M    │    │             │
└───────────┘    └──────────────┘    └──────────┘    └──────┬──────┘
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │     Backward Pass          │
                                              │  compute gradients         │
                                              └─────────────┬──────────────┘
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │     Optimizer (AdamW)      │
                                              │  update 23.5M weights      │
                                              └─────────────┬──────────────┘
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │   Scheduler (CosineAnneal) │
                                              │  reduce learning rate      │
                                              └────────────────────────────┘
```

Repeat for 100 epochs (or until early stopping at patience=8).
