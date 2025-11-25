"""
DETR COCO Evaluation Script

Evaluates a pretrained DETR model on the COCO val2017 dataset and computes
standard COCO metrics (mAP, AR, etc.) using pycocotools.

Features:
- Robust CLI (dataset path, model choice, device, conf threshold, batch size)
- Correct COCO category mapping via category names
- Batch inference for better GPU utilization
- Timing statistics (ms/img, FPS)
- JSON results for COCOeval + separate metadata JSON
- Optional CSV logging hook for plotting latency / detection stats
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# COCO category names (90 categories, plus background placeholder at index 0)
# This list matches the standard COCO class naming convention used in many repos.
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window",
    "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DETR on COCO val2017")
    parser.add_argument(
        "--coco-path", default="coco", type=str,
        help="Path to COCO dataset root (containing val2017/ and annotations/)"
    )
    parser.add_argument(
        "--output", default="detr_coco_val_results.json", type=str,
        help="Output JSON file for COCO-format predictions"
    )
    parser.add_argument(
        "--conf-threshold", default=0.0, type=float,
        help="Confidence threshold for detections (0.0 for full COCO-style eval)"
    )
    parser.add_argument(
        "--device", default="cuda", type=str,
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size", default=1, type=int,
        help="Batch size for inference (>=1)"
    )
    parser.add_argument(
        "--model", default="detr_resnet50", type=str,
        help="Model to evaluate (e.g., detr_resnet50, detr_resnet101)"
    )
    parser.add_argument(
        "--log-csv", default=None, type=str,
        help="Optional CSV file to log per-image latency and detection counts"
    )
    parser.add_argument(
        "--seed", default=42, type=int,
        help="Random seed for deterministic shuffling (if used)"
    )
    return parser.parse_args()


def build_classname_to_catid_map(coco: COCO):
    """
    Build a mapping from COCO category name -> category_id using the dataset
    annotation file, so we can convert DETR's class indices to COCO catIds.
    """
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c["name"]: c["id"] for c in cats}
    return name_to_id


def rescale_bboxes(out_bbox: torch.Tensor, size):
    """
    Convert normalized DETR [cx, cy, w, h] (0-1) to absolute [x, y, w, h] in pixels.
    """
    img_w, img_h = size
    cx, cy, w, h = out_bbox.unbind(-1)
    x0 = (cx - 0.5 * w) * img_w
    y0 = (cy - 0.5 * h) * img_h
    w = w * img_w
    h = h * img_h
    return torch.stack([x0, y0, w, h], dim=-1)


def main():
    args = parse_args()
    random.seed(args.seed)

    start_time = time.time()

    # -------------------------------------------------
    # Device setup
    # -------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("DETR COCO Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Batch size: {args.batch_size}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # -------------------------------------------------
    # Load COCO validation dataset
    # -------------------------------------------------
    try:
        coco_root = Path(args.coco_path)
        ann_file = coco_root / "annotations" / "instances_val2017.json"

        if not ann_file.exists():
            print(f"Error: Annotation file not found at {ann_file}")
            sys.exit(1)

        print(f"\nLoading COCO annotations from {ann_file}...")
        coco = COCO(str(ann_file))

        img_dir = coco_root / "val2017"
        if not img_dir.exists():
            print(f"Error: Image directory not found at {img_dir}")
            sys.exit(1)

        img_ids = coco.getImgIds()
        print(f"Found {len(img_ids)} validation images")

        classname_to_catid = build_classname_to_catid_map(coco)

    except Exception as e:
        print(f"Error loading COCO dataset: {e}")
        sys.exit(1)

    # -------------------------------------------------
    # Load the released DETR model
    # -------------------------------------------------
    try:
        print(f"\nLoading {args.model} from torch.hub...")
        model = torch.hub.load("facebookresearch/detr", args.model, pretrained=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # -------------------------------------------------
    # Image transform (standard DETR preprocessing)
    # -------------------------------------------------
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    # -------------------------------------------------
    # Optional CSV logger setup
    # -------------------------------------------------
    csv_writer = None
    csv_file = None
    if args.log_csv is not None:
        csv_path = Path(args.log_csv)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_id", "time_ms", "num_detections"])

    # -------------------------------------------------
    # Run inference on all images in val2017 (batched)
    # -------------------------------------------------
    print("\nStarting inference...")
    print("-" * 70)

    results = []
    inference_times = []
    detection_counts = []
    images_with_detections = 0
    failed_images = []

    total_images = len(img_ids)
    inference_start = time.time()

    # Simple deterministic ordering (or shuffle if you want)
    # random.shuffle(img_ids)

    batch_size = max(1, args.batch_size)

    for batch_start in range(0, total_images, batch_size):
        batch_ids = img_ids[batch_start: batch_start + batch_size]

        images = []
        sizes = []
        valid_img_ids = []

        # Load and preprocess batch
        for img_id in batch_ids:
            try:
                img_info = coco.loadImgs(img_id)[0]
                img_path = img_dir / img_info["file_name"]

                if not img_path.exists():
                    failed_images.append((img_id, "File not found"))
                    continue

                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                sizes.append((w, h))
                images.append(transform(img))
                valid_img_ids.append(img_id)

            except Exception as e:
                failed_images.append((img_id, str(e)))
                continue

        if not images:
            continue  # nothing valid in this batch

        inputs = torch.stack(images, dim=0).to(device)

        # Inference timing
        batch_start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        batch_time = time.time() - batch_start_time

        # Record average per-image time for this batch
        per_image_time = batch_time / len(valid_img_ids)
        for _ in valid_img_ids:
            inference_times.append(per_image_time)

        # Process outputs
        prob = outputs["pred_logits"].softmax(-1)[..., :-1]  # drop "no-object"
        boxes = outputs["pred_boxes"]

        for i, img_id in enumerate(valid_img_ids):
            img_probs = prob[i]
            img_boxes = boxes[i]
            img_size = sizes[i]

            max_scores, labels = img_probs.max(-1)
            keep = max_scores > args.conf_threshold if args.conf_threshold > 0.0 else torch.ones_like(max_scores, dtype=torch.bool)

            kept_scores = max_scores[keep]
            kept_labels = labels[keep]
            kept_boxes = rescale_bboxes(img_boxes[keep], img_size)

            num_dets = len(kept_boxes)
            detection_counts.append(num_dets)
            if num_dets > 0:
                images_with_detections += 1

            # COCO-format results
            for box, score, label_idx in zip(kept_boxes, kept_scores, kept_labels):
                x, y, bw, bh = box.tolist()

                # Map class index -> class name -> COCO category_id
                # Note: label_idx is 0-based index into our category list WITHOUT background.
                # We offset by +1 to skip "__background__".
                class_name_idx = int(label_idx) + 1
                if class_name_idx >= len(COCO_INSTANCE_CATEGORY_NAMES):
                    continue

                class_name = COCO_INSTANCE_CATEGORY_NAMES[class_name_idx]
                if class_name not in classname_to_catid:
                    # Some classes in the name list might not appear in this COCO subset
                    continue

                category_id = classname_to_catid[class_name]

                results.append({
                    "image_id": img_id,
                    "category_id": int(category_id),
                    "bbox": [x, y, bw, bh],
                    "score": float(score.item())
                })

            # CSV logging per image (using avg time for this batch)
            if csv_writer is not None:
                csv_writer.writerow([img_id, per_image_time * 1000.0, num_dets])

        # Progress + ETA
        processed = batch_start + len(valid_img_ids)
        elapsed = time.time() - inference_start
        avg_time = elapsed / max(1, processed)
        eta = avg_time * (total_images - processed)

        print(
            f"Processed {processed:>5}/{total_images} images | "
            f"Avg: {avg_time*1000:.1f} ms/img | "
            f"ETA: {eta/60:.1f} min"
        )

    if csv_file is not None:
        csv_file.close()

    inference_time = time.time() - inference_start

    # -------------------------------------------------
    # Inference Summary
    # -------------------------------------------------
    print("-" * 70)
    print("\nInference Summary:")
    num_processed = total_images - len(failed_images)
    print(f"  Total images processed: {num_processed}/{total_images}")
    if inference_times:
        avg_ms = sum(inference_times) / len(inference_times) * 1000.0
        print(f"  Total inference time: {inference_time:.2f} s")
        print(f"  Average time per image: {avg_ms:.1f} ms")
        print(f"  Images/second: {num_processed / inference_time:.2f}")
    print(f"  Total detections: {len(results)}")
    if detection_counts:
        print(f"  Avg detections per image: {sum(detection_counts)/len(detection_counts):.2f}")
    print(
        f"  Images with detections: {images_with_detections}/{num_processed} "
        f"({100.0 * images_with_detections / max(1, num_processed):.1f}%)"
    )

    if failed_images:
        print(f"\n  Failed images: {len(failed_images)}")
        for img_id, error in failed_images[:5]:
            print(f"    - Image {img_id}: {error}")
        if len(failed_images) > 5:
            print(f"    ... and {len(failed_images) - 5} more")

    # -------------------------------------------------
    # Save predictions + metadata
    # -------------------------------------------------
    print(f"\nSaving predictions to {args.output}...")

    metadata = {
        "model": args.model,
        "confidence_threshold": args.conf_threshold,
        "num_images": total_images,
        "num_processed": num_processed,
        "num_detections": len(results),
        "timestamp": datetime.now().isoformat(),
        "inference_time_seconds": inference_time,
        "device": str(device),
        "batch_size": batch_size,
    }

    # Save results only (COCO eval format)
    with open(args.output, "w") as f:
        json.dump(results, f)

    # Save detailed metadata
    metadata_file = args.output.replace(".json", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Results saved to {args.output}")
    print(f"✓ Metadata saved to {metadata_file}")

    # -------------------------------------------------
    # COCO Evaluation
    # -------------------------------------------------
    print("\n" + "=" * 70)
    print("Running COCO Evaluation")
    print("=" * 70)

    try:
        coco_dt = coco.loadRes(args.output)
        coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]

        print("\n" + "=" * 70)
        print("Key Metrics Summary:")
        print("=" * 70)
        print(f"  mAP (IoU=0.50:0.95): {stats[0]:.3f}")
        print(f"  mAP (IoU=0.50):      {stats[1]:.3f}")
        print(f"  mAP (IoU=0.75):      {stats[2]:.3f}")
        print(f"  AR  (maxDets=100):   {stats[8]:.3f}")
        print("=" * 70)

    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        sys.exit(1)

    # -------------------------------------------------
    # Final Summary
    # -------------------------------------------------
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {args.output}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
