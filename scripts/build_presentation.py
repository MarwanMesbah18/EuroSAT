import nbformat
import copy

# Read V4 notebook
with open('notebooks/eurosat_rgb_v4.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

pres = copy.deepcopy(nb)
pres.metadata['celltoolbar'] = 'Slideshow'

for cell in pres.cells:
    if 'slideshow' not in cell.metadata:
        cell.metadata['slideshow'] = {}
    cell.metadata['slideshow']['slide_type'] = '-'

slide_map = {
    1: 'slide', 2: 'sub',
    3: 'slide', 4: 'sub',
    5: 'slide', 6: 'sub', 7: 'sub',
    8: 'slide', 9: 'sub', 10: 'sub',
    11: 'slide', 12: 'sub',
    13: 'slide', 14: 'sub', 15: 'sub', 16: 'sub',
    17: 'slide', 18: 'sub',
    19: 'slide', 20: 'sub',
    21: 'slide', 22: 'sub',
    23: 'slide', 24: 'sub',
    25: 'slide', 26: 'sub',
    27: 'slide', 28: 'sub',
    29: 'slide', 30: 'sub',
}

for idx, slide_type in slide_map.items():
    if idx < len(pres.cells):
        pres.cells[idx].metadata['slideshow']['slide_type'] = slide_type

for idx in [0, 31, 32]:
    if idx < len(pres.cells):
        pres.cells[idx].metadata['slideshow']['slide_type'] = 'skip'


# === Helper ===
def md(source):
    cell = nbformat.v4.new_markdown_cell(source=source)
    cell.metadata['slideshow'] = {'slide_type': '-'}
    return cell

# === Title ===
title_cell = md("""# EuroSAT Land Use Classification
## with ResNet50 Built From Scratch

**Satellite Image Classification using Sentinel-2 Data**

EuroSAT Dataset · 10 Classes · 27,000 Images · 97.65% Accuracy

*Built without any pretrained weights — trained from random initialization*""")
title_cell.metadata['slideshow'] = {'slide_type': 'slide'}

# === Sections inserted DURING the notebook ===

about_dataset = md("""---

## About the EuroSAT Dataset

> *"EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"*
> — Helber et al., 2019 — [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)

- **Source:** Sentinel-2 satellite (ESA Copernicus program)
- **Spatial resolution:** 10m per pixel (RGB bands)
- **Image size:** 64×64 pixels
- **Spectral bands:** 13 available (we use RGB: bands 4, 3, 2)
- **Total images:** ~27,000 labeled and geo-referenced images
- **10 classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

**Why this dataset?** Sentinel-2 provides free, global coverage with high revisit frequency — ideal for land use monitoring, urban planning, and environmental analysis.

**Reference benchmark:** The original paper achieved **98.57% accuracy** using a *fine-tuned* (pretrained) ResNet-50. Our goal: match this from scratch.
""")

# --- Architecture Deep Dive ---
arch_resnet = md("""---

## Architecture: ResNet50 — Why and How

> *"Deep Residual Learning for Image Recognition"* — He et al., 2015 (227,000+ citations)
> Winner of ILSVRC 2015 with **3.57% top-5 error** on ImageNet — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

### The Problem: Degradation
Adding more layers to a network should improve accuracy, but in practice, **deeper plain networks get *worse*** — not because of overfitting, but because deeper networks are harder to optimize. A 56-layer network has *higher training error* than a 20-layer network.

### The Solution: Skip Connections
Instead of learning a direct mapping H(x), ResNet learns a **residual**: F(x) = H(x) − x, and the output becomes **F(x) + x**.

```
Input x ────────────────────────────┐
  │                                  │
  ├─ Conv 1×1 (reduce)              │
  │  BatchNorm → ReLU               │
  ├─ Conv 3×3                       │
  │  BatchNorm → ReLU               │
  ├─ Conv 1×1 (expand)              │
  │  BatchNorm                      │
  │                                  │
  └────────────── + ←───────────────┘
                 │
              ReLU → Output
```

**Why it works:** If the optimal transformation is close to identity, the network just needs to push F(x) toward zero — much easier than learning identity from scratch. Skip connections also create "information highways" for gradient flow, preventing vanishing gradients in 50+ layer networks.

![ResNet Architecture](../photos/restnett.png)
""")

arch_bottleneck = md("""---

### Bottleneck Block Design (1×1 → 3×3 → 1×1)

Used in ResNet-50/101/152 (not in ResNet-18/34):

| Layer | Operation | Channels | Purpose |
|-------|-----------|----------|---------|
| 1 | **1×1 Conv** | 64→64 | Reduce dimensions (bottleneck) |
| 2 | **3×3 Conv** | 64→64 | Extract spatial features |
| 3 | **1×1 Conv** | 64→256 | Expand back to full channels |

**Why bottleneck?** A single 3×3 conv from 256→256 channels costs **590K parameters**. The bottleneck path (256→64→64→256) costs only **70K parameters** — **8.4× fewer** while maintaining representational power.

### ResNet50 Layer Structure

| Stage | Blocks | Channels | Output Size |
|-------|--------|----------|-------------|
| Stem | — | 3→64 | 224→112 |
| Layer 1 | 3 | 64→256 | 112→56 |
| Layer 2 | 4 | 256→512 | 56→28 |
| Layer 3 | 6 | 512→1024 | 28→14 |
| Layer 4 | 3 | 1024→2048 | 14→7 |
| FC | — | 2048→10 | 1×1 |

**Total: ~23.5M parameters** | **3.8 billion FLOPs** per image
""")

batch_norm = md("""---

### Batch Normalization

> *"Batch Normalization: Accelerating Deep Network Training"* — Ioffe & Szegedy, 2015
> — [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

**Problem:** As training progresses, each layer's input distribution shifts (internal covariate shift), requiring careful initialization and small learning rates.

**Solution:** Normalize each mini-batch: `x̂ = (x − μ_batch) / √(σ² + ε)`, then learn scale γ and shift β.

Applied after every convolution, before ReLU in our ResNet50:
```
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → ...
```

**Why it matters:**
- Enables **14× faster training** (from the paper)
- Allows **10× higher learning rates** without divergence
- Provides regularization (reduces need for Dropout)
- Critical for training 50+ layers from scratch

**Initialization in our model:** BN weights = 1, BN biases = 0 (standard practice)
""")

kaiming = md("""---

### Weight Initialization: Kaiming (He) Init

> *"Delving Deep into Rectifiers"* — He et al., 2015
> — [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)

**Why not Xavier/Glorot?** Xavier init assumes linear activations, but ReLU zeros out half the values — requiring a factor-of-2 correction:

| Init | Formula | For |
|------|---------|-----|
| Xavier | `Var(W) = 1/n_fan_in` | Linear/tanh |
| **Kaiming** | **`Var(W) = 2/n_fan_in`** | **ReLU** |

The extra factor of 2 compensates for the variance lost when ReLU kills half the activations. Without this, signal vanishes through 50+ layers — making training from scratch impossible.

**In our model:** `nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` for all Conv2d layers.
""")

adamw = md("""---

### Optimizer: Why AdamW, Not Adam or SGD?

> *"Decoupled Weight Decay Regularization"* — Loshchilov & Hutter, 2019
> — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

**The problem with Adam + L2:** In Adam, the L2 penalty gets scaled by adaptive learning rates — making regularization inconsistent and ineffective.

**AdamW fix:** Decouples weight decay from the gradient step:
```
# Standard Adam + L2:
loss = loss_fn(output, target) + λ * ||w||²    # regularization gets scaled by Adam

# AdamW:
w = w - lr * wd * w    # weight decay applied directly, independent of Adam
w = w - lr * Adam(grad) # gradient step separate
```

**Result:** Weight decay and learning rate become independent hyperparameters — easier to tune and better generalization.

**Our choice:** AdamW with lr=3e-4, weight_decay=1e-4
""")

cosine = md("""---

### Learning Rate Schedule: Cosine Annealing

Smoothly decreases the learning rate from 3e-4 → 1e-6 following a cosine curve:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(πt / T_max))
```

**Why not StepLR?** Step drops cause sudden jumps that destabilize training. Cosine provides smooth, continuous decay — the model gradually shifts from exploration (high lr) to refinement (low lr).

**Combined with early stopping (patience=8):** The scheduler handles the full 100-epoch plan, while early stopping prevents overfitting if the model converges early.
""")

augmentation = md("""---

### Why These Augmentations?

Satellite images have unique properties different from natural photos — the camera (Sentinel-2) is always directly overhead, so:

| Augmentation | Why |
|-------------|-----|
| **Horizontal flip** | A forest looks the same mirrored |
| **Vertical flip** | Satellite is overhead — north/south are arbitrary |
| **Random rotation (15°)** | Land patches can appear at any orientation |
| **Color jitter** | Atmospheric conditions change colors between captures |
| **RandomResizedCrop** | Simulates zoom level and position variation |

**Why vertical flip is important:** In natural photo datasets (ImageNet), vertical flips are wrong (an upside-down dog is not a dog). But satellite images are orientation-agnostic — a river or crop field looks valid from any angle.
""")

# === Evolution section — placed at the END ===
evolution = md("""---

## Iterative Development: V1 → V4

We went through 4 versions, testing different approaches to improve accuracy:

| Version | Normalization | Loss | Key Change | Test Accuracy | Epochs |
|---------|--------------|------|------------|---------------|--------|
| **V1 (Baseline)** | ImageNet | Standard CE | Full pipeline from scratch | **97.65%** | 100 |
| **V2** | ImageNet | Standard CE | Increased patience (8→12) | **97.65%** | 100 |
| **V3** | **Dataset-specific** | Standard CE | Computed EuroSAT mean/std | **97.58%** | 66 |
| **V4** | ImageNet | Standard CE | Better code structure + viz | **97.14%** | 39 |

> **Note:** Class-weighted loss was tested in V4 but removed — it hurt per-class accuracy (see below). V5 will address this.

### Key Findings

- **ImageNet normalization worked best** — even though EuroSAT images are satellite (not natural photos), the wider normalization range `[0.485, 0.456, 0.406]` helped the model separate harder classes like PermanentCrop vs AnnualCrop.

- **Dataset-specific normalization slightly hurt** — EuroSAT's computed mean `[0.345, 0.381, 0.409]` is darker with less variation `std [0.201, 0.136, 0.114]`. This compressed the feature space, making it harder for the network to distinguish similar land cover types.

- **Lesson learned:** Sometimes simpler is better. The baseline V1 approach achieved the highest accuracy.
""")

per_class = md("""---

### Per-Class Accuracy Across Versions

| Class | V1 | V2 | V3 | V4 |
|-------|----|----|----|----|
| AnnualCrop | 97.95% | 97.95% | 97.72% | 97.49% |
| Forest | 98.88% | 98.88% | **99.33%** | **99.33%** |
| HerbaceousVegetation | 95.64% | 95.64% | 96.79% | 97.02% |
| Highway | 98.13% | 98.13% | 97.86% | 97.59% |
| Industrial | 97.63% | 97.63% | 98.16% | 97.11% |
| Pasture | 96.64% | 96.64% | 98.13% | 97.39% |
| PermanentCrop | 96.07% | 96.07% | **91.36%** | **91.88%** |
| Residential | 98.60% | 98.60% | **99.53%** | **99.30%** |
| River | 97.62% | 97.62% | 97.38% | 94.76% |
| SeaLake | 98.73% | 98.73% | **98.95%** | 98.73% |

**Observations:**
- **PermanentCrop** is the hardest class across all versions (~91-96%) — visually similar to AnnualCrop and HerbaceousVegetation
- **V3** boosted Forest and Residential to 99%+ but PermanentCrop dropped to 91.36%
- **V4** (with class weights) hurt River (94.76%) and didn't help PermanentCrop — weights removed
- **V1/V2** remain the most balanced overall

> **V5 (upcoming):** Will focus on improving PermanentCrop accuracy through targeted augmentation and focal loss.
""")

# --- References ---
references = md("""---

## References

1. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. **Ioffe, S. & Szegedy, C.** (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

3. **Helber, P., Bischke, B., Dengel, A., & Borth, D.** (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS. [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)

4. **Loshchilov, I. & Hutter, F.** (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

5. **He, K., Zhang, X., Ren, S., & Sun, J.** (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* ICCV 2015. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
""")

# --- Summary ---
summary = md("""---

## Results Summary

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **97.65%** |
| Macro F1 | 97.48% |
| Weighted F1 | 97.61% |
| Best Class | SeaLake (98.73%) |
| Hardest Class | PermanentCrop (96.07%) |

### Key Takeaways
- ResNet50 from scratch (no pretrained weights) — **only 1% below the pretrained benchmark** of 98.57%
- Kaiming init + BatchNorm enabled stable training of 50+ layers from random weights
- ImageNet normalization outperformed dataset-specific normalization
- Satellite-specific augmentations (vertical flips, rotation) were critical

---

## Live Demo → Switch to Streamlit
""")
summary.metadata['slideshow'] = {'slide_type': 'slide'}


# === Assemble ===
final_cells = [title_cell]

for i, cell in enumerate(pres.cells):
    if cell.metadata.get('slideshow', {}).get('slide_type') == 'skip':
        continue

    final_cells.append(cell)

    # Insert extra cells after specific sections:
    if i == 7:    # After data loading code
        final_cells.append(about_dataset)
    if i == 12:   # After augmentation viz code
        final_cells.append(augmentation)
    if i == 16:   # After model creation code
        final_cells.extend([arch_resnet, arch_bottleneck, batch_norm, kaiming])
    if i == 18:   # After loss setup code
        final_cells.extend([adamw, cosine])

# End sections: evolution → per-class → references → summary
final_cells.extend([evolution, per_class, references, summary])

pres.cells = final_cells

with open('notebooks/eurosat_resnet50_presentation.ipynb', 'w') as f:
    nbformat.write(pres, f)

print(f"Presentation notebook created with {len(pres.cells)} cells")
