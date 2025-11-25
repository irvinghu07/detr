# DETR Standalone Demo

This document explains why we run the standalone demo, how it works, and what the output means.

---

## Why This Demo Is Needed

- **Confirm DETR loads correctly** - Pretrained weights, transformer, and backbone are properly initialized
- **Verify environment setup** - CUDA, PyTorch, and dependencies work before downloading COCO
- **Minimal reproducible test** - Teammates can run this instantly without large downloads
- **Inference demonstration** - Shows how DETR performs inference without datasets, annotations, or training
- **Stage-1 verification** - Essential checkpoint before running full evaluation or training

---

## How the Demo Works

### 1. Load pretrained DETR (ResNet-50)

```python
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
```

Torch Hub automatically downloads:
- ResNet-50 ImageNet weights
- DETR COCO-trained weights

### 2. Fetch a sample COCO image (online)

No dataset download required. The demo fetches a single image directly from the COCO validation set.

### 3. Run a forward pass

DETR outputs:
- `pred_logits` → class scores for 100 queries
- `pred_boxes` → bounding boxes (normalized `cx`, `cy`, `w`, `h`)

### 4. Apply a confidence threshold

Keep predictions above **0.7** to filter out low-confidence detections.

### 5. Print detected objects

Shows class name, confidence score, and bounding box for each detection.

---

## What the Output Means

**Example output:**

```
Detected 4 objects (confidence > 0.7):
  cat             | confidence: 0.999 | bbox: (0.345, 0.267, 0.456, 0.789)
  remote          | confidence: 0.905 | bbox: (0.512, 0.123, 0.089, 0.156)
  bed             | confidence: 0.884 | bbox: (0.500, 0.600, 0.800, 0.700)
```

**This proves:**
- DETR is functioning correctly
- Transformer decoder is producing object queries
- Post-processing and softmax are correct
- Environment setup is valid
- No COCO dataset, training, or annotations required

---

## Conclusion

This demo serves as a **lightweight verification step** before:

1. Downloading COCO dataset
2. Running full evaluation (`main.py --eval`)
3. Modifying DETR for experiments
4. Training on subsets or custom datasets

**It confirms the entire DETR inference pipeline works end-to-end.**