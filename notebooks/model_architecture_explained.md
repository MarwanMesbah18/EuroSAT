# How the ResNet50 Model Runs — Code Walkthrough

This document traces through the **exact code** in V4, cell by cell, showing what happens when you create and run the model. Designed for someone who needs to explain this in a discussion.

---

## 1. Why ResNet50 for Satellite Imagery?

Before diving into code, the natural question: **why this architecture?**

EuroSAT has 10 land-use classes. Some look very similar from above:

```
AnnualCrop vs PermanentCrop  →  both are green farmland
HerbaceousVegetation vs Pasture  →  both are grassy areas
Highway vs River  →  both are long narrow features
```

The model needs to learn **subtle texture and pattern differences**, not just colors. This requires:
- **Depth**: enough layers to build complex features (edges → textures → shapes → objects)
- **Skip connections**: so deep layers can still learn (gradients don't vanish)
- **Reasonable size**: 23.5M params — powerful but trainable on a single GPU in ~2 hours

ResNet50 hits all three. It was designed to solve exactly this problem: "how to train a very deep network without losing gradient information."

Why not a simpler model (like a small CNN)?
- A shallow CNN would struggle with classes that differ in fine texture
- We tried it — accuracy was ~85-90%. ResNet50 pushes it to 97%+.

Why not a bigger model (ResNet101, ResNet152)?
- Diminishing returns: +0.5% accuracy but 2× training time
- EuroSAT images are 64×64 (upscaled to 224×224) — there's a limit to how much complexity the input contains

---

## 2. What Does "Bottleneck" Mean?

The name comes from the **shape** of the data flowing through the block:

```
Input: 256 channels (wide)
        │
        ▼
   1×1 Conv → 64 channels (narrow)     ← the "neck" of the bottle
        │
        ▼
   3×3 Conv → 64 channels (narrow)     ← still narrow
        │
        ▼
   1×1 Conv → 256 channels (wide)      ← back to wide
        │
        ▼
Output: 256 channels (wide)

Visual:
    256 channels  ████████████████████████████████
                   \                              /
                    \                            /
           64 ch     \                          /
                      |========================|    ← bottleneck (narrow)
           64 ch     /                          \
                    /                            \
                   /                              \
    256 channels  ████████████████████████████████
```

**Why is this efficient?** Compare the two approaches for a block that takes 256 channels in and outputs 256 channels:

```
WITHOUT bottleneck (3×3 convs on all channels):
  Conv 3×3 (256 → 256) → Conv 3×3 (256 → 256) → Conv 3×3 (256 → 256)
  Parameters: 3 × (256 × 256 × 3 × 3) = 1,769,472

WITH bottleneck (squeeze → process → expand):
  Conv 1×1 (256 → 64) → Conv 3×3 (64 → 64) → Conv 1×1 (64 → 256)
  Parameters: (256×64×1×1) + (64×64×3×3) + (64×256×1×1)
             = 16,384 + 36,864 + 16,384 = 69,632

Same output, 25× fewer parameters.
```

---

## 3. What Is `expansion = 4`?

This is the ratio between the **narrow** part and the **wide** part of the bottleneck:

```python
class Bottleneck(nn.Module):
    expansion = 4  # ← THIS
```

It means: **the output of each block has 4× the number of base channels**.

```
If out_channels = 64:
  → Block output = 64 × 4 = 256 channels

If out_channels = 128:
  → Block output = 128 × 4 = 512 channels

If out_channels = 256:
  → Block output = 256 × 4 = 1024 channels

If out_channels = 512:
  → Block output = 512 × 4 = 2048 channels
```

This single number controls the entire channel progression through the network. Without it, we'd need to hardcode `256`, `512`, `1024`, `2048` everywhere.

---

## 4. How `__init__` Flows — The Birth of the Model

When you write:
```python
model = ResNet50(num_classes=10).to(DEVICE)
```

Here's exactly what happens, step by step:

### Step 1: `ResNet50.__init__()` runs

```
ResNet50.__init__(num_classes=10)
│
├── 1. CREATE STEM
│   self.conv1   = Conv2d(3 → 64, kernel=7×7, stride=2)
│   self.bn1     = BatchNorm2d(64)
│   self.relu    = ReLU()
│   self.maxpool = MaxPool2d(3×3, stride=2)
│
├── 2. CREATE LAYER 1  → calls _make_layer(64, 64, blocks=3, stride=1)
│   │
│   ├── Block 0: Bottleneck(64 → 64, stride=1, downsample=...)
│   │             downsample: Conv2d(64→256, 1×1) + BN  ← needed because 64≠256
│   │             conv1: Conv2d(64→64, 1×1)  bn1
│   │             conv2: Conv2d(64→64, 3×3)  bn2
│   │             conv3: Conv2d(64→256, 1×1) bn3       ← 64 * expansion(4) = 256
│   │
│   ├── Block 1: Bottleneck(256 → 64, stride=1, downsample=None)
│   │             conv1: Conv2d(256→64, 1×1)  bn1       ← input is now 256
│   │             conv2: Conv2d(64→64, 3×3)   bn2
│   │             conv3: Conv2d(64→256, 1×1)  bn3       ← output: 256
│   │
│   └── Block 2: Bottleneck(256 → 64, stride=1, downsample=None)
│                 (same as Block 1)
│
│   Result: layer1 = Sequential(Block0, Block1, Block2)
│   Input→Output: 64 channels → 256 channels, spatial: 56×56 → 56×56
│
├── 3. CREATE LAYER 2  → calls _make_layer(256, 128, blocks=4, stride=2)
│   │
│   ├── Block 0: Bottleneck(256 → 128, stride=2, downsample=...)
│   │             downsample: Conv2d(256→512, 1×1, stride=2) + BN
│   │             conv1: Conv2d(256→128, 1×1)  bn1
│   │             conv2: Conv2d(128→128, 3×3, stride=2) bn2  ← halves spatial size
│   │             conv3: Conv2d(128→512, 1×1)  bn3          ← 128 * 4 = 512
│   │
│   ├── Block 1: Bottleneck(512 → 128)
│   ├── Block 2: Bottleneck(512 → 128)
│   └── Block 3: Bottleneck(512 → 128)
│
│   Result: layer2 = Sequential(Block0, Block1, Block2, Block3)
│   Input→Output: 256 → 512 channels, spatial: 56×56 → 28×28
│
├── 4. CREATE LAYER 3  → calls _make_layer(512, 256, blocks=6, stride=2)
│   │ (6 blocks, same pattern)
│   Result: 512 → 1024 channels, spatial: 28×28 → 14×14
│
├── 5. CREATE LAYER 4  → calls _make_layer(1024, 512, blocks=3, stride=2)
│   │ (3 blocks, same pattern)
│   Result: 1024 → 2048 channels, spatial: 14×14 → 7×7
│
├── 6. CREATE CLASSIFIER
│   self.avgpool = AdaptiveAvgPool2d(1, 1)   ← 2048 × 7×7 → 2048 × 1×1
│   self.fc      = Linear(2048 → 10)          ← final 10 class scores
│
└── 7. INITIALIZE ALL WEIGHTS
    for every module:
      if Conv2d:     kaiming_normal_ (He initialization)
      if BatchNorm:  weight=1, bias=0
```

### Step 2: `_make_layer()` — The Factory Function

This function is called 4 times. Each time it builds a stack of Bottleneck blocks:

```python
def _make_layer(self, in_channels, out_channels, blocks, stride):
```

```
_make_layer(in_channels=64, out_channels=64, blocks=3, stride=1)
│
│  First block is special — it may need to:
│  a) Change spatial size (stride=2)
│  b) Match channel count (in_channels ≠ out_channels × 4)
│  → Create a downsample shortcut: 1×1 conv + BN
│
├── Create downsample (if needed):
│   nn.Sequential(
│     Conv2d(64 → 256, 1×1, stride=1, bias=False),
│     BatchNorm2d(256)
│   )
│
├── Block 0: Bottleneck(in=64, out=64, stride=1, downsample=above)
│
└── Blocks 1-2: Bottleneck(in=256, out=64, stride=1, downsample=None)
    (after block 0, input channels = 64×4 = 256, which matches output)
    (no downsample needed because channels already match)
```

Why does only the first block need `downsample`?

```
Block 0:  input=64ch  →  output=256ch  → MISMATCH!  → need downsample to convert 64→256
Block 1:  input=256ch →  output=256ch  → MATCH       → skip connection just copies input
Block 2:  input=256ch →  output=256ch  → MATCH       → same
```

---

## 5. How `forward()` Flows — One Image's Journey

When you call:
```python
outputs = model(images)   # images shape: [64, 3, 224, 224]
```

Here's what happens to **one image** (we follow the tensor shape):

```
Input: [1, 3, 224, 224]  (1 image, 3 color channels, 224×224 pixels)
        │
        │  ┌─────────── STEM ───────────┐
        │  │ conv1: [1, 3, 224, 224] → [1, 64, 112, 112]
        │  │   7×7 kernel, stride=2 cuts spatial size in half
        │  │   3 channels → 64 channels
        │  │
        │  │ bn1: same shape [1, 64, 112, 112]
        │  │ relu: same shape
        │  │
        │  │ maxpool: [1, 64, 112, 112] → [1, 64, 56, 56]
        │  │   3×3 pool, stride=2 halves again
        │  └────────────────────────────┘
        │
        ▼
[1, 64, 56, 56]
        │
        │  ┌──── LAYER 1: 3 Bottleneck blocks ────┐
        │  │                                        │
        │  │  Block 0:                              │
        │  │  [1, 64, 56, 56] → [1, 256, 56, 56]  │
        │  │   1×1: 64→64 → 3×3: 64→64 → 1×1: 64→256
        │  │   + skip (downsample: 64→256)          │
        │  │                                        │
        │  │  Block 1:                              │
        │  │  [1, 256, 56, 56] → [1, 256, 56, 56]  │
        │  │   1×1: 256→64 → 3×3: 64→64 → 1×1: 64→256
        │  │   + skip (just copy 256ch)             │
        │  │                                        │
        │  │  Block 2: same as Block 1              │
        │  └────────────────────────────────────────┘
        │
        ▼
[1, 256, 56, 56]
        │
        │  ┌──── LAYER 2: 4 Bottleneck blocks ────┐
        │  │  stride=2 → spatial halves            │
        │  │  [1, 256, 56, 56] → [1, 512, 28, 28]  │
        │  │  4 blocks, same pattern as Layer 1     │
        │  └────────────────────────────────────────┘
        │
        ▼
[1, 512, 28, 28]
        │
        │  ┌──── LAYER 3: 6 Bottleneck blocks ────┐
        │  │  stride=2 → spatial halves            │
        │  │  [1, 512, 28, 28] → [1, 1024, 14, 14] │
        │  │  6 blocks (the deepest layer)          │
        │  └────────────────────────────────────────┘
        │
        ▼
[1, 1024, 14, 14]
        │
        │  ┌──── LAYER 4: 3 Bottleneck blocks ────┐
        │  │  stride=2 → spatial halves            │
        │  │  [1, 1024, 14, 14] → [1, 2048, 7, 7]  │
        │  │  3 blocks                              │
        │  └────────────────────────────────────────┘
        │
        ▼
[1, 2048, 7, 7]
        │
        │  ┌──── CLASSIFIER ────┐
        │  │ avgpool: [1, 2048, 7, 7] → [1, 2048, 1, 1]
        │  │   averages each 7×7 map into a single number
        │  │   2048 "features" extracted
        │  │
        │  │ flatten: [1, 2048, 1, 1] → [1, 2048]
        │  │
        │  │ fc: [1, 2048] → [1, 10]
        │  │   10 raw scores (logits), one per class
        │  └────────────────────┘
        │
        ▼
Output: [1, 10]
```

---

## 6. What Happens Inside One Bottleneck Block

Let's trace through Block 0 of Layer 2 — the most interesting block because it does **both** a channel change AND spatial reduction:

```
Input tensor: [1, 256, 56, 56]    (256 channels, 56×56 spatial)
                │
                ├─── COPY for skip connection ────┐
                │                                 │
                ▼                                 │
         ┌──────────────┐                         │
         │ conv1 (1×1)  │ 256 → 128 channels      │
         │ BatchNorm    │                          │
         │ ReLU         │ [1, 128, 56, 56]        │
         └──────┬───────┘                          │
                ▼                                  │
         ┌──────────────┐                          │
         │ conv2 (3×3)  │ 128 → 128 channels       │
         │ stride=2     │ ← THIS halves spatial    │
         │ BatchNorm    │                           │
         │ ReLU         │ [1, 128, 28, 28]         │
         └──────┬───────┘                           │
                ▼                                   │
         ┌──────────────┐                           │
         │ conv3 (1×1)  │ 128 → 512 channels        │
         │ BatchNorm    │ [1, 512, 28, 28]          │
         └──────┬───────┘                            │
                │                                    │
                │                     ┌──────────────┘
                │                     │
                │              ┌──────────────────┐
                │              │  DOWNSAMPLE      │
                │              │  1×1 conv        │
                │              │  256→512, stride=2│
                │              │  BatchNorm       │
                │              │  [1, 512, 28, 28]│
                │              └────────┬─────────┘
                │                       │
                ▼                       ▼
                ┌───────────────────────┐
                │    ADD (out += skip)   │
                │  [1, 512, 28, 28]     │
                └───────────┬───────────┘
                            ▼
                     ┌─────────────┐
                     │    ReLU     │
                     └──────┬──────┘
                            ▼
                Output: [1, 512, 28, 28]
```

**The skip connection is critical.** Without it, the block only learns a transformation of the input. With it, the block learns a **residual** — the difference between input and desired output:

```
out = F(x) + x
        │     │
        │     └── "keep whatever was already useful"
        └──────── "learn only what needs to change"
```

This is why it's called **ResNet** — Residual Network. Each block learns a residual correction.

---

## 7. Spatial Size Summary

```
         224          112        56       56       28       14        7        1
          │            │         │        │        │        │         │        │
   Input  │  Stem      │  MaxPool│ Layer1 │ Layer2 │ Layer3 │ Layer4  │ AvgPool│ FC
   3ch    │  conv1     │         │ 3blks  │ 4blks  │ 6blks  │ 3blks   │        │10
          │  stride=2  │strid=2  │strid=1 │strid=2 │strid=2 │ stride=2│        │
          │            │         │        │        │        │         │        │
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ → ▓▓▓▓▓▓▓▓▓ → ▓▓▓▓▓▓ → ▓▓▓▓▓▓ → ▓▓▓▓ → ▓▓▓ → ▓▓ → ▓ → ▓▓▓▓▓▓▓▓▓▓
   224×224      112×112    56×56   56×56   28×28   14×14    7×7    1×1    10 scores

   Channels:    3 → 64 → 64 → 256 → 512 → 1024 → 2048 → 2048 → 10
```

---

## 8. Where Do the 23.5M Parameters Live?

```
Layer         Parameters    Percentage
─────────────────────────────────────
Stem            9,408        0.04%
Layer 1      1,481,216       6.3%
Layer 2      3,580,416      15.2%
Layer 3      8,327,168      35.4%    ← largest (6 blocks)
Layer 4      6,894,592      29.3%
FC head       20,490        0.09%
─────────────────────────────────────
Total       23,528,522      100%
```

Layer 3 is the biggest because it has 6 blocks (the most of any layer). This is where the model learns the most complex features.

---

## 9. Code ↔ Architecture Mapping

Here's the exact mapping between code lines and what they create:

```python
# CELL 14: Bottleneck Block Definition
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        #  ┌───────────── squeeze ─────────────┐
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)   # 1×1: reduce channels
        self.bn1   = nn.BatchNorm2d(out_channels)
        #  └────────────────────────────────────┘
        #  ┌───────────── process ─────────────┐
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)                    # 3×3: learn features
        self.bn2   = nn.BatchNorm2d(out_channels)
        #  └────────────────────────────────────┘
        #  ┌───────────── expand ──────────────┐
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1)                               # 1×1: restore channels
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)
        #  └────────────────────────────────────┘
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample   # ← shortcut for dim matching

    def forward(self, x):
        identity = x                   # ← save input for skip connection

        out = self.conv1(x);  out = self.bn1(out);  out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)  # ← NO ReLU here

        if self.downsample is not None:
            identity = self.downsample(x)  # ← adjust skip to match dims

        out += identity              # ← SKIP CONNECTION: add residual
        out = self.relu(out)
        return out
```

```python
# CELL 15: ResNet50 Network
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # STEM — initial feature extraction
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # FOUR LAYERS of bottleneck blocks
        # _make_layer creates N Bottleneck blocks and wraps them in Sequential
        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1)  # → 256ch,  56×56
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)  # → 512ch,  28×28
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)  # → 1024ch, 14×14
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2) # → 2048ch, 7×7

        # CLASSIFIER — turn features into predictions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # collapse spatial dims
        self.fc      = nn.Linear(2048, num_classes)   # 2048 features → 10 scores

        # INITIALIZATION — set starting weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Factory: create a stack of Bottleneck blocks."""
        # First block handles dimension change
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        # Remaining blocks: channels already match, no downsample needed
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # STEM
        x = self.conv1(x)    # [B, 3, 224, 224] → [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # → [B, 64, 56, 56]

        # RESIDUAL LAYERS
        x = self.layer1(x)   # → [B, 256, 56, 56]
        x = self.layer2(x)   # → [B, 512, 28, 28]
        x = self.layer3(x)   # → [B, 1024, 14, 14]
        x = self.layer4(x)   # → [B, 2048, 7, 7]

        # CLASSIFIER
        x = self.avgpool(x)  # → [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # → [B, 2048]
        x = self.fc(x)       # → [B, 10]
        return x
```

---

## 10. The "Why" Behind Each Design Choice

| Design Choice | Why |
|---|---|
| **7×7 stem conv** (not 3×3) | Large receptive field in the first layer captures more spatial context early |
| **Stride 2 in stem + maxpool** | Quickly reduces spatial size from 224 to 56 (4× reduction) — saves compute |
| **Bottleneck (1×1→3×3→1×1)** | 25× fewer parameters than doing 3×3 on full channels — same representational power |
| **Skip connections (`out += identity`)** | Solves vanishing gradient — allows training 50+ layers deep |
| **BatchNorm after every conv** | Stabilizes training — keeps activations in a reasonable range |
| **Kaiming initialization** | Matches weight variance to activation function (ReLU) — prevents early training collapse |
| **AdaptiveAvgPool (not flatten)** | Makes architecture input-size agnostic — any spatial size becomes 1×1 |
| **Single FC head (2048→10)** | Simple classifier — the conv layers do all the heavy feature extraction |
| **No dropout** | BatchNorm + weight decay + early stopping provide enough regularization |
| **50 layers (not more/less)** | Sweet spot: deep enough for complex features, shallow enough to train efficiently |

---

## 11. Quick Reference: The 3-Second Explanation

```
"I built ResNet50 from scratch — 50 layers, 23.5 million parameters.
 It uses bottleneck blocks that squeeze, process, and expand features
 through skip connections that let gradients flow through 50 layers.
 The model takes 224×224 satellite images and outputs 10 class scores.
 It reached 97.65% accuracy on the EuroSAT dataset in about 2 hours on a T4 GPU."
```
