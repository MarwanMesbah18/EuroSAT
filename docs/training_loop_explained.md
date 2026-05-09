# Training Loop — Code Walkthrough

This document explains every line of the training loop cell in the V4 notebook.

---

## The Big Picture

The training loop repeats the same cycle for up to 100 epochs:

```
┌─────────────────────────────────────────────────────────┐
│                     ONE EPOCH                           │
│                                                         │
│  1. TRAIN: Feed all 18,900 images (in batches of 64)    │
│     → Model makes predictions                           │
│     → Calculate loss (how wrong)                        │
│     → Backward pass (compute gradients)                  │
│     → Update 23.5M weights                              │
│                                                         │
│  2. VALIDATE: Feed 4,050 images (no weight updates)     │
│     → Measure accuracy on unseen data                   │
│                                                         │
│  3. CHECKPOINT: Is this the best model so far?           │
│     → If yes: save it                                   │
│     → If no improvement for 8 epochs: STOP              │
│                                                         │
│  4. SCHEDULER: Reduce learning rate (cosine curve)       │
└─────────────────────────────────────────────────────────┘

Repeat until epoch 100 OR early stopping.
```

---

## Line by Line

### Setup (Before the Loop)

```python
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
best_val_acc = -1.0
best_model_wts = None
patience_counter = 0
```

```
history          → stores metrics after each epoch (for plotting curves later)
best_val_acc     → tracks the highest validation accuracy seen so far (starts at -1)
best_model_wts   → saves a copy of the best model's weights
patience_counter → counts how many epochs without improvement (stops at 8)
```

---

### The Outer Loop: Epochs

```python
for epoch in range(NUM_EPOCHS):  # 0, 1, 2, ..., 99
    start_time = time.time()
```

One epoch = one full pass through the entire training dataset (18,900 images).

```
Epoch 1:  see all 18,900 images → update weights
Epoch 2:  see all 18,900 images again → update weights again
...
Epoch 50: model has seen every image 50 times
```

Why see the same images multiple times? Because the model improves gradually. On epoch 1 it might learn "green = Forest." By epoch 20 it learns "this specific texture = Forest." By epoch 50 it catches subtle details.

---

### Phase 1: Training

```python
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
```

- `model.train()` — switches the model to **training mode**. This matters because:
  - **BatchNorm** uses batch statistics (mean/var of current batch) instead of running averages
  - **Dropout** would be active (if we used it — we don't, but the mode switch still matters for BN)

- `running_loss`, `running_corrects`, `total` — accumulators to compute averages after the epoch

```python
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
```

This iterates through the training data **in batches of 64**:

```
train_loader contains 18,900 images:
  Batch 1:  images[0:64]    → inputs shape [64, 3, 224, 224]
  Batch 2:  images[64:128]  → inputs shape [64, 3, 224, 224]
  ...
  Batch 295: images[18816:18880]
  Batch 296: images[18880:18900]  → last batch might be smaller

Total: 296 batches per epoch
```

`tqdm` shows a progress bar for each epoch.

#### Step 1: Move to GPU

```python
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
```

```
Before: inputs on CPU → After: inputs on GPU
         [64, 3, 224, 224]      [64, 3, 224, 224]
```

Everything must be on the same device (GPU) for computation.

#### Step 2: Clear Old Gradients

```python
        optimizer.zero_grad()
```

PyTorch **accumulates** gradients by default. If you don't clear them, gradients from batch 1 would be added to gradients from batch 2, etc. `zero_grad()` resets them to zero before each batch.

```
Before zero_grad:  all 23.5M gradient values contain numbers from the previous batch
After zero_grad:   all 23.5M gradient values = 0
```

#### Step 3: Forward Pass (with Mixed Precision)

```python
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
```

**Forward pass** = feed images through the model:

```
inputs [64, 3, 224, 224]
    │
    ▼
model.forward()
    │
    ▼
outputs [64, 10]  ← 10 raw scores per image (logits)
```

Then calculate loss:

```
For each of the 64 images:
  outputs[i] = [2.1, 5.3, -0.2, 0.8, 1.1, -1.5, 3.0, 0.3, -0.7, 4.2]  ← model's guess
  labels[i] = 1                                                           ← true class (Forest)

  CrossEntropyLoss compares them:
  "You said class 7 had score 3.0, but the answer was class 1 with score 5.3"
  → loss for this image = some number (lower = better)

Average loss across all 64 images → single number (e.g., loss = 0.8423)
```

**AMP autocast**: automatically uses float16 for some operations (faster, less memory) while keeping critical parts in float32 (accurate). This happens automatically — you don't need to think about it.

#### Step 4: Backward Pass (Compute Gradients)

```python
        scaler.scale(loss).backward()
```

**Backward pass** = "how much should each weight change to reduce the loss?"

```
loss (single number: 0.8423)
    │
    ▼  backward() — chain rule through all 50 layers
    │
    ├→ "conv1 weight should change by +0.001"
    ├→ "conv1 bias should change by -0.002"
    ├→ "bn1 weight should change by +0.0005"
    ├→ ... (23.5M gradients computed)
    └→ "fc weight [2048, 10] should change by ..."

This is stored in each parameter's .grad attribute:
  model.conv1.weight.grad = [gradient values]
  model.fc.weight.grad = [gradient values]
```

**scaler.scale()**: multiplies the loss by a large number before backward to prevent float16 underflow (very small gradients becoming zero). The scaler tracks this and undoes it before the weight update.

#### Step 5: Update Weights

```python
        scaler.step(optimizer)
        scaler.update()
```

- `scaler.step(optimizer)` — applies the gradients to update weights:
  ```
  For each of the 23.5M parameters:
    weight = weight - lr × gradient + weight_decay × weight
    (AdamW adds momentum and adaptive learning rate on top of this)
  ```
- `scaler.update()` — adjusts the scale factor for the next batch (if gradients were fine, increase scale for more speed; if they overflowed, decrease scale)

#### Step 6: Track Metrics

```python
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)
```

```
loss.item()          → extract the number from the loss tensor (e.g., 0.8423)
× inputs.size(0)     → multiply by batch size (64) so we can average later

torch.max(outputs, 1) → for each image, find which class had the highest score
  outputs: [64, 10]
  preds:   [64]  ← one predicted class index per image

(preds == labels)     → compare predictions to true labels
  [True, False, True, True, ...]
  .sum()              → count how many were correct
  running_corrects   → accumulate across all batches

total                 → count total images seen
```

After all batches:

```python
    train_loss = running_loss / total
    train_acc = running_corrects / total
```

```
train_loss = total loss across all 18,900 images / 18,900
train_acc  = total correct predictions / 18,900

Example:
  running_loss = 15,923.4  →  train_loss = 15923.4 / 18900 = 0.8423
  running_corrects = 12,054  →  train_acc = 12054 / 18900 = 0.6379 (63.8%)
```

---

### Phase 2: Validation

```python
    model.eval()
    val_loss, val_corrects, val_total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
```

Key differences from training:

```
model.train()                        model.eval()
├─ BatchNorm: uses batch stats       ├─ BatchNorm: uses running averages (stable)
├─ Gradients computed                ├─ No gradients (faster, less memory)
└─ Weights updated                   └─ Weights NOT updated (evaluation only)

torch.no_grad():
  Tells PyTorch "don't build the computation graph"
  → Saves ~50% memory
  → Runs faster
  → We can't call backward() (but we don't need to)
```

The rest is the same as training — forward pass, compute loss, count correct predictions — but **no weight updates**:

```python
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += (preds == labels).sum().item()
            val_total += labels.size(0)
```

After all validation batches:

```python
    val_loss = val_loss / val_total
    val_acc = val_corrects / val_total
```

---

### Phase 3: Scheduler Step

```python
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
```

After each epoch, the learning rate decreases following the cosine curve:

```
Epoch 1:  lr = 0.000300
Epoch 10: lr = 0.000295
Epoch 25: lr = 0.000250
Epoch 50: lr = 0.000150
Epoch 75: lr = 0.000050
Epoch 100: lr = 0.000001

      3e-4 ┤╲
           │ ╲
           │  ╲
           │   ╲
           │    ╲
           │     ╲
      1e-6 ┤      ╲___________________
           └──────────────────────────
           0     25    50    75   100
```

This is recorded in `history["lr"]` for plotting.

---

### Phase 4: Record History + Print

```python
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] {elapsed:.0f}s | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.6f}")
```

Example output:
```
Epoch [50/100] 120s | Train Loss: 0.0234 Acc: 0.9931 | Val Loss: 0.0891 Acc: 0.9765 | LR: 0.000150
```

The `history` dict is what the training curves plot (cell after this) uses.

---

### Phase 5: Checkpoint + Early Stopping

```python
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({...}, os.path.join(OUTPUT_DIR, "best_resnet50_eurosat.pth"))
        patience_counter = 0
        print(f"  -> New best model saved (val_acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
```

This is the decision-making part:

```
Epoch 40: val_acc = 97.0% → better than 96.8% (previous best)
  → SAVE model weights (deep copy so training doesn't modify them)
  → SAVE checkpoint file (model + optimizer state, for resuming later)
  → Reset patience to 0

Epoch 41: val_acc = 96.8% → NOT better than 97.0%
  → patience = 1

Epoch 42: val_acc = 96.5% → NOT better
  → patience = 2
  ...
Epoch 48: val_acc = 96.3%
  → patience = 8 = PATIENCE
  → STOP training

Final model = the one saved at epoch 40 (the best)
```

**Why early stopping?** If val accuracy stops improving but train accuracy keeps climbing, the model is **overfitting** — memorizing training data instead of learning general patterns. Early stopping catches this and uses the best model.

**Why `copy.deepcopy`?** `model.state_dict()` returns a reference to the model's current weights. If we just save the reference, training would modify them. `deepcopy` creates an independent snapshot.

**Why save optimizer state too?** If you want to resume training later, you need the optimizer's momentum buffers (AdamW tracks running averages of gradients). Without it, the optimizer would start "cold" and training would be unstable.

---

## End of Training

```python
print(f"\nBest validation accuracy: {best_val_acc:.4f}")
```

After the loop ends (either by reaching 100 epochs or early stopping), the best model weights are loaded later in the evaluation cell:

```python
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)
```

This loads the **best checkpoint** (not the last epoch) for testing.

---

## Timeline: What a Full Training Run Looks Like

```
Epoch  1: Train Acc: 30% → 50%    Val Acc: 45%
          Model learns basic colors: green=plants, blue=water, gray=urban

Epoch  5: Train Acc: 70% → 80%    Val Acc: 75%
          Model learns textures: forests are dense, crops are striped

Epoch 15: Train Acc: 88% → 92%    Val Acc: 90%
          Model learns shapes: highways are long lines, rivers curve

Epoch 30: Train Acc: 96% → 98%    Val Acc: 96%
          Model fine-tunes: separating similar classes (AnnualCrop vs PermanentCrop)

Epoch 45: Train Acc: 99%+          Val Acc: 97.6%
          Near peak — small improvements

Epoch 50: Train Acc: 99.5%         Val Acc: 97.65%  ← BEST MODEL SAVED

Epoch 51-58: Val Acc fluctuates 96.5-97.0% (no improvement)
          patience counter: 1, 2, 3, 4, 5, 6, 7, 8

Epoch 58: EARLY STOPPING

Final: Best model from epoch 50 loaded for testing → 97.65% test accuracy
```

---

## One Batch Visualized

```
64 images [64, 3, 224, 224] + 64 labels [64]
        │
        ▼  inputs.to(DEVICE), labels.to(DEVICE)
        │
        ▼  optimizer.zero_grad()
        │  Clear all 23.5M gradients to zero
        │
        ▼  model(inputs)  ← FORWARD PASS
        │  Images flow through 50 layers
        │  Output: [64, 10] raw scores
        │
        ▼  criterion(outputs, labels)  ← LOSS
        │  Compare predictions to truth
        │  Output: single number (e.g., 0.8423)
        │
        ▼  loss.backward()  ← BACKWARD PASS
        │  Chain rule through all 50 layers
        │  Computes 23.5M gradients
        │
        ▼  optimizer.step()  ← WEIGHT UPDATE
        │  Adjust all 23.5M weights using gradients
        │  (AdamW adds momentum + adaptive LR + weight decay)
        │
        ▼  Track metrics (loss + accuracy)
        │
        ▼  Next batch of 64 images...
```

This repeats 296 times per epoch (18,900 images / 64 per batch), for up to 100 epochs.
