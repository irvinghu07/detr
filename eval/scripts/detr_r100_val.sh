#!/bin/bash

echo ""
echo "=========================================="
echo "Evaluating DETR ResNet-101"
echo "=========================================="
python eval/eval_detr_coco.py \
  --coco-path coco \
  --model detr_resnet101 \
  --conf-threshold 0.0 \
  --batch-size 1 \
  --device cuda \
  --output eval/outputs/resnet101b1/detr_r101_b1_val_results.json \
  --log-csv eval/outputs/resnet101b1/detr_r101_b1_val_latency.csv

echo ""
echo "=========================================="
echo "Result saved to:"
echo "  - eval/outputs/resnet101b1/"
echo "=========================================="
