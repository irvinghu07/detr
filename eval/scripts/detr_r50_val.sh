#!/bin/bash

python eval/eval_detr_coco.py \
  --coco-path coco \
  --model detr_resnet50 \
  --conf-threshold 0.0 \
  --batch-size 1 \
  --device cuda \
  --output eval/outputs/resnet50b1/detr_r50_b1_val_results.json \
  --log-csv eval/outputs/resnet50b1/detr_r50_b1_val_latency.csv
