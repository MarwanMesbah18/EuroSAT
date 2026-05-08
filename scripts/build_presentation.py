import nbformat
import copy

# Read V4 notebook
with open('notebooks/eurosat_rgb_v4.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Create presentation notebook (deep copy of V4)
pres = copy.deepcopy(nb)
pres.metadata['celltoolbar'] = 'Slideshow'

# Clear all existing slideshow metadata and set defaults
for cell in pres.cells:
    if 'slideshow' not in cell.metadata:
        cell.metadata['slideshow'] = {}
    cell.metadata['slideshow']['slide_type'] = '-'

# Define slide assignments by cell index (based on V4 structure)
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

# Skip cells 0 (old title), 31-32 (save artifacts)
for idx in [0, 31, 32]:
    if idx < len(pres.cells):
        pres.cells[idx].metadata['slideshow']['slide_type'] = 'skip'

# Insert title slide at the beginning
title_cell = nbformat.v4.new_markdown_cell(source="""# EuroSAT Land Use Classification
## with ResNet50 Built From Scratch

**Satellite Image Classification using Sentinel-2 Data**

EuroSAT Dataset · 10 Classes · 27,000 Images · 97.28% Accuracy""")
title_cell.metadata['slideshow'] = {'slide_type': 'slide'}
pres.cells.insert(0, title_cell)

# Append summary slide at the end
summary_cell = nbformat.v4.new_markdown_cell(source="""## Results Summary

| Metric | Score |
|--------|-------|
| **Best Validation Accuracy** | **97.28%** |
| Macro F1 | 97.48% |
| Weighted F1 | 97.61% |

---

### Key Takeaways
- ResNet50 from scratch — no pretrained weights
- Class-weighted loss to handle data imbalance
- ImageNet normalization + satellite-specific augmentations
- Mixed precision training on Kaggle T4 GPU

---

## Live Demo → Switch to Streamlit""")
summary_cell.metadata['slideshow'] = {'slide_type': 'slide'}
pres.cells.append(summary_cell)

# Write presentation notebook
with open('notebooks/eurosat_resnet50_presentation.ipynb', 'w') as f:
    nbformat.write(pres, f)

print(f"Presentation notebook created with {len(pres.cells)} cells")
