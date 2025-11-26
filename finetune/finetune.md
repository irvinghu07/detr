# DETR Fine-Tuning Summary

## Mini-COCO Experiment – Full Explanation of What Was Done and Why

This document summarizes the complete fine-tuning process, the motivation behind each step, the architectural modifications we tested, and the interpretation of the results. It is written so that any team member can understand the workflow and use it directly in the final report.

---

## 1. Objective

The project requires the following:

1. Running the DETR model described in the ECCV 2020 paper
2. Obtaining comparable outputs
3. Applying meaningful modifications to improve or analyze the model
4. Producing conclusions based on experiments

Because full COCO training requires more resources than available (both storage and GPU hours), the task was redesigned into a smaller, reproducible fine-tuning experiment that still demonstrates understanding and improvement of DETR.

---

## 2. Dataset Decision: Using a Mini-COCO Split

The official DETR training uses the COCO train2017 set (118,000 images). This dataset is too large to fit into our storage quota on RunPod.

As a practical alternative, we used **COCO val2017** (5,000 images) and created an 80/20 split:

- **4,000 images** for training
- **1,000 images** for validation

This forms a smaller dataset that still preserves COCO's diversity. It allows DETR to be fine-tuned without exceeding disk limitations and provides a reliable evaluation signal.

---

## 3. Fine-Tuning Setup

### Base Model

The base model is the official COCO-pretrained **DETR-ResNet50** checkpoint (`detr-r50-e632da11.pth`). This checkpoint represents a complete training run on the full COCO dataset. Using this as initialization ensures that fine-tuning on mini-COCO is meaningful and stable.

### Architecture

The architecture is directly built using the official DETR implementation from the FacebookResearch repository (`build_model` and criterion, including Hungarian matching loss).

### Training Configuration

- **Optimizer:** AdamW
- **Learning rates:**
  - `1e-4` for DETR components
  - `1e-5` for the backbone
- **Epochs:** 5
- **Batch size:** 2
- **Loss:** Original DETR set prediction loss
  - Classification loss
  - L1 box regression
  - GIoU loss
  - Auxiliary layer losses

The dataset wrapper converts COCO bounding boxes into the normalized box format DETR expects.

---

## 4. Experiments Performed

Three main configurations were tested to analyze DETR under limited-data conditions.

### Experiment 1: Baseline (Original Architecture)

**Configuration:**
- Decoder layers: 6
- Object queries: 100
- All settings identical to the original DETR architecture

**Result:**
- Best validation loss: **~71.3**
- This serves as the baseline reference point

### Experiment 2: Reducing the Transformer Decoder Depth

**Configuration:**
- Decoder layers: 3
- Object queries: 100

**Motivation:**
The decoder is the most computationally expensive part of DETR. Reducing the depth may prevent overfitting and improve generalization on smaller datasets.

**Result:**
- Best validation loss: **~52.5**
- This is a substantial improvement compared to the baseline
- Shows that on a smaller dataset, a shallow decoder can perform better than the original deeper one

### Experiment 3: Reducing the Number of Object Queries

**Configuration:**
- Decoder layers: 3
- Object queries: 50

**Implementation:**
Only the first 50 query embeddings from the pretrained checkpoint were retained. DETR originally uses 100 queries, but smaller datasets usually do not require so many detection slots.

**Result:**
- Best validation loss: **~51.45**
- Slightly better than the 3-layer 100-query model
- Significantly better than the original architecture
- Shows that DETR does not require 100 queries in small-data regimes and can operate effectively with fewer

---

## 5. Comparison of Results

The following table summarizes the performance of all three models on the mini-COCO validation set:

| Model Configuration          | Decoder Layers | Queries | Best Validation Loss |
|------------------------------|----------------|---------|----------------------|
| Original COCO-pretrained     | 6              | 100     | ~71.3                |
| Shallow Decoder              | 3              | 100     | ~52.5                |
| Shallow + Reduced Queries    | 3              | 50      | ~51.45               |
---

## 6. Interpretation of the Results

The results demonstrate three important points:

### 6.1 Decoder Depth and Overfitting

DETR with six decoder layers appears to overfit when fine-tuned on a subset as small as mini-COCO. Lowering the decoder depth improves generalization and yields a substantially lower validation loss.

### 6.2 Query Efficiency

Reducing the number of object queries from 100 to 50 does not degrade performance. Instead, it achieves slightly better results while reducing the model size and computation. This means DETR retains flexibility in the number of query slots, and excessive queries are unnecessary in small-data or resource-constrained environments.

### 6.3 Overall Impact

Both tested modifications create a more efficient and better-performing model for our specific dataset size and hardware limitations. This fulfills the "make improvements through modifications" requirement of the project.

---

## 7. Summary for Report Integration

We replicated DETR using the official implementation and adapted it for a reduced-resource environment. We fine-tuned the COCO-pretrained ResNet50 DETR on a mini-COCO dataset created from the COCO val2017 split.

To examine improvements, we tested architectural modifications, including:
- Reducing the number of transformer decoder layers
- Reducing the number of object queries

**Key Findings:**
- **Baseline** (6-layer decoder, 100 queries): validation loss ~71.3
- **Shallow decoder** (3 layers, 100 queries): validation loss ~52.5 (27% improvement)
- **Optimized** (3 layers, 50 queries): validation loss ~51.45 (28% improvement)

These findings show that DETR can be made both **more efficient** and **more accurate** in smaller-data regimes by simplifying its architecture, which aligns well with the constraints and goals of our project.