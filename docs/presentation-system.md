# Presentation System — How to Edit and Rebuild

This guide explains how the presentation system works so you (or another chat session) can edit and rebuild it.

---

## Architecture

```
notebooks/eurosat_rgb_v5.ipynb     ← source notebook (V5)
        │
        ▼ scripts/build_presentation.py
notebooks/eurosat_resnet50_presentation.ipynb  ← generated presentation notebook
        │
        ▼ jupyter nbconvert --to html
presentation/presentation_scroll.html  ← main deliverable (scrollable)
presentation/scroll.css                ← extracted CSS (editable)
        │
        ▼ scripts/extract_css.py
(CSS separated from HTML for easy editing)
```

## Files

| File | Purpose |
|------|---------|
| `scripts/build_presentation.py` | Reads V5 notebook, adds extra markdown sections (architecture, comparisons, paper references), generates presentation notebook |
| `scripts/extract_css.py` | Extracts inline CSS from nbconvert HTML into separate .css files |
| `notebooks/eurosat_resnet50_presentation.ipynb` | Generated presentation notebook (do not edit directly — edit the script) |
| `presentation/presentation_scroll.html` | Main scrollable HTML presentation |
| `presentation/scroll.css` | Styles for scrollable presentation |
| `presentation/presentation_slides.html` | Reveal.js slideshow version |
| `presentation/slides.css` | Styles for slideshow |
| `demo.py` | Streamlit live demo app |

## How to Edit the Presentation

### Edit text/content

1. Open `scripts/build_presentation.py`
2. Find the relevant section (`about_dataset`, `normalization`, `arch_resnet`, `evolution`, `per_class`, `curves_compare`, `summary`, etc.)
3. Edit the markdown text in the `md("""...""")` block
4. Rebuild (see below)

### Add a new section

1. In `scripts/build_presentation.py`, create a new markdown cell:
```python
my_new_section = md("""---
## My New Section Title
Content here...
""")
```

2. In the assembly section at the bottom, insert it where you want:
```python
# After cell 16 (model creation)
if i == 16:
    final_cells.extend([arch_resnet, arch_bottleneck, batch_norm, kaiming])
    final_cells.append(my_new_section)  # ← add here
```

### Add a new image

1. Place the image in `photos/` or `outputs/`
2. Reference it in a markdown cell:
```python
md("![My Image](../photos/my_image.png)")
```

### Switch to a different notebook version

In `scripts/build_presentation.py`, change line 4:
```python
# Read V5 notebook
with open('notebooks/eurosat_rgb_v5.ipynb', 'r') as f:
```

### Update per-class accuracy or evolution data

Edit the `evolution` and `per_class` markdown cells in the build script. The V1-V5 per-class recall values are hardcoded in the markdown table.

## Rebuild Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Step 1: Generate presentation notebook
python scripts/build_presentation.py

# Step 2: Export to scrollable HTML
jupyter nbconvert --to html notebooks/eurosat_resnet50_presentation.ipynb --output presentation_scroll.html
mv notebooks/presentation_scroll.html presentation/

# Step 3: Extract CSS from HTML
python scripts/extract_css.py

# Step 4 (optional): Export slideshow version
jupyter nbconvert --to slides notebooks/eurosat_resnet50_presentation.ipynb --output presentation_slides.html
mv notebooks/presentation_slides.html.slides.html presentation/presentation_slides.html
# or just:
mv notebooks/presentation_slides.html presentation/
```

## Run the Demo

```bash
source venv/bin/activate
streamlit run demo.py
```

Test images go in `test_images/` — name them `test_1.jpg`, `test_2.jpg`, etc.

## Notebook Version Reference

| Version | File | Key Features | Accuracy |
|---------|------|-------------|----------|
| V1 | `eurosat_rgb_v1.ipynb` | Baseline, ImageNet norm | 97.65% |
| V2 | `eurosat_rgb_v2.ipynb` | Patience 8→12 | 97.65% |
| V3 | `eurosat_rgb_v3.ipynb` | Dataset-specific norm | 97.58% |
| V4 | `eurosat_rgb_v4.ipynb` | Class-weighted loss (removed), augmentation viz | 97.14% |
| V5 | `eurosat_rgb_v5.ipynb` | **CutMix, Label Smoothing, TTA** | **98.32%** |

## Important Notes

- Cell 22 in V5 (training curves) has no output — skipped in the presentation, replaced by the V3 vs V5 curves comparison section
- The presentation uses **V5 as the base** — all notebook outputs (confusion matrix, per-class accuracy, sample predictions) are from V5
- Extra markdown sections (architecture, normalization, evolution, per-class comparison, references) are injected by the build script
- The `docs/superpowers/` folder contains the original design spec and is safe to delete
- `docs/project-documentation.md` consolidates all explanation markdown files
