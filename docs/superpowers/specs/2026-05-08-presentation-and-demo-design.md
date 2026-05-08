# Presentation & Live Demo Design

## Context
The user needs to present their EuroSAT ResNet50 satellite image classification project to their professor ("doctor"). They want an all-in-one, browser-based solution that doesn't depend on PowerPoint or specific software. The presentation should showcase the project work, results, and include a live demo where the professor can see the model classify images in real-time.

## Deliverables

### 1. HTML Slideshow (`presentation_slides.html`)
- **Format:** reveal.js HTML slideshow generated from the notebook
- **Navigation:** Arrow keys, fullscreen with F key
- **Content flow (10 slides):**
  1. Title slide — project name, author, date
  2. Dataset overview — EuroSAT description, Sentinel-2 context, 10 classes, class distribution chart
  3. Sample images — 10x5 sample grid showing each class
  4. Preprocessing & augmentation — normalization, satellite-specific augmentations, augmentation examples image
  5. ResNet50 architecture — built from scratch, bottleneck blocks, skip connections, 23.5M params
  6. Training setup — hyperparams, AdamW, CosineAnnealing, AMP, early stopping
  7. Training results — training curves (loss + accuracy)
  8. Evaluation — confusion matrix, per-class accuracy, classification report highlights
  9. Sample predictions — prediction grid with correct/incorrect labels
  10. Summary & demo — final accuracy (97.65%), key achievements, "switch to demo"
- **Images:** All 7 existing output PNGs embedded from `outputs/`
- **Styling:** Dark theme matching notebook style, clean fonts, code snippets where relevant
- **Generation:** Restructure notebook cells with slide metadata, then `nbconvert --to slides`

### 2. Pure HTML Scroll Version (`presentation_scroll.html`)
- **Format:** Standard nbconvert HTML export — one long scrollable page
- **Purpose:** Backup/reference version, easy to scroll through everything
- **Content:** The full notebook with all code cells + outputs as-is
- **Generation:** `nbconvert --to html` on the presentation-ready notebook
- **Styling:** Default nbconvert styling (clean, readable)

### 3. Gradio Demo App (`demo.py`)
- **Framework:** Gradio
- **Model loading:** Loads `models/best_resnet50_eurosat.pth` (weights-only checkpoint)
- **Interface:**
  - Image upload widget (drag & drop or click)
  - Prediction output: top prediction with confidence bar
  - All 10 class confidences shown as horizontal bar chart
  - Option to try sample images from the dataset
- **Preprocessing:** Same transforms as validation (Resize(256) → CenterCrop(224) → Normalize with dataset mean/std)
- **Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Launch:** `python demo.py` → opens at `localhost:7860`

## Presentation Workflow
1. Open `presentation_slides.html` in browser
2. Present with arrow keys (fullscreen with F)
3. At slide 10, switch browser tab to Gradio (already running)
4. Upload satellite images or use provided samples
5. Show live predictions

## Files to Create/Modify
- `notebooks/eurosat_resnet50_presentation.ipynb` — restructured notebook with slide metadata and title markdown cells
- `presentation_slides.html` — exported reveal.js slideshow
- `presentation_scroll.html` — exported scrollable HTML
- `demo.py` — Gradio demo app at project root

## Existing Files Referenced
- `notebooks/eurosat_resnet50.ipynb` — source notebook to restructure
- `outputs/*.png` — all 7 visualization images to embed
- `models/best_resnet50_eurosat.pth` — saved model weights
- `outputs/classification_report.csv` — metrics for summary slide

## Verification
1. Open `presentation_slides.html` in browser — verify all slides render, images load, arrow keys work
2. Open `presentation_scroll.html` — verify all cells and outputs display
3. Run `python demo.py` — verify Gradio launches, upload an image, confirm prediction works and shows confidence bars for all 10 classes
4. Test with an image from each of the 10 dataset classes
