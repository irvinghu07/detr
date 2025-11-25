"""
DETR Demo Script
Simple demo showing DETR object detection without COCO dependencies.
"""
import torch
from PIL import Image
import requests
from torchvision.transforms import ToTensor

# COCO class names
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Constants
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
CONFIDENCE_THRESHOLD = 0.7

def main():
    try:
        # Detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load DETR from torch.hub
        print("Loading DETR model...")
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        model.to(device)
        model.eval()
        
        # Load sample image
        print(f"Fetching image from {IMAGE_URL}...")
        response = requests.get(IMAGE_URL, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(response.raw)
        print(f"Image size: {img.size}")
        
        # Preprocess
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        
        # Inference
        print("\nRunning inference...")
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Post-process results
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > CONFIDENCE_THRESHOLD
        
        # Extract detected objects
        boxes = outputs['pred_boxes'][0, keep].cpu()
        scores = probas[keep].max(-1).values.cpu()
        labels = probas[keep].max(-1).indices.cpu()
        
        print(f"\nDetected {len(boxes)} objects (confidence > {CONFIDENCE_THRESHOLD}):")
        print("-" * 60)
        for score, label, box in zip(scores, labels, boxes):
            class_name = COCO_CLASSES[label]
            cx, cy, w, h = box.tolist()
            print(f"  {class_name:15s} | confidence: {score:.3f} | bbox: ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
        
        print("\n" + "="*60)
        print("Success! DETR is running without COCO dependencies.")
        print("="*60)
        
    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

if __name__ == "__main__":
    main()
