import random
from pathlib import Path
import sys

# ----------------------------------------------------------------------
# Make repo root importable (so "models", "util", "main" can be imported)
# ----------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.append(str(REPO_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
import torchvision.transforms as T

from models.detr import build as build_model
from util.misc import nested_tensor_from_tensor_list
from main import get_args_parser


class CocoDetrDataset(CocoDetection):
    """
    Minimal COCO dataset wrapper for DETR.

    - Uses val2017 images + instances_val2017.json
    - Converts COCO bbox [x, y, w, h] -> [x_min, y_min, x_max, y_max]
    - Normalizes boxes to [0, 1] for DETR
    """

    def __init__(self, img_folder, ann_file, img_transform=None):
        # NOTE: do NOT pass transforms to super().__init__ and
        # do NOT override self.transforms, because CocoDetection
        # expects transforms(image, target).
        super().__init__(img_folder, ann_file, transforms=None)
        self.img_transform = img_transform or T.ToTensor()

    def __getitem__(self, idx):
        # CocoDetection.__getitem__ returns (image, target_anns)
        img, anns = super().__getitem__(idx)

        # Original image size (width, height)
        w, h = img.size

        # Apply our own image-only transform
        if self.img_transform is not None:
            img = self.img_transform(img)

        boxes = []
        labels = []

        for ann in anns:
            # COCO bbox: [x, y, w, h] in absolute pixels
            x, y, bw, bh = ann["bbox"]
            x_min = x
            y_min = y
            x_max = x + bw
            y_max = y + bh

            # Normalize to [0,1] as expected by DETR
            x_min /= w
            x_max /= w
            y_min /= h
            y_max /= h

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,                  # normalized [x_min, y_min, x_max, y_max]
            "labels": labels,                # COCO category_id
            "image_id": torch.tensor(self.ids[idx]),
        }
        return img, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def main():
    # ------------------------------------------------------------------
    # 1) Reproducibility
    # ------------------------------------------------------------------
    random.seed(7)
    torch.manual_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 2) Dataset: use val2017 with a train/val split
    # ------------------------------------------------------------------
    coco_root = REPO_ROOT / "coco"
    img_dir = coco_root / "val2017"
    ann_file = coco_root / "annotations" / "instances_val2017.json"

    dataset = CocoDetrDataset(img_dir, ann_file)
    n_total = len(dataset)
    train_ratio = 0.8
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(7),
    )

    print(f"Total images: {n_total}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # 3) Build DETR model & criterion from this repo
    # ------------------------------------------------------------------
    parser = get_args_parser()
    # parse empty list -> use defaults, then override what we need
    args = parser.parse_args([])

    args.dataset_file = "coco"
    args.device = device.type
    args.dec_layers = 3
    args.num_queries = 50
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # ------------------------------------------------------------------
    # 4) Load official COCO-pretrained ResNet-50 DETR weights
    # ------------------------------------------------------------------
    ckpt_path = REPO_ROOT / "detr-r50-e632da11.pth"
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]

    # --- handle query_embed size mismatch (100 -> 50) ---
    qe_key = "query_embed.weight"
    if qe_key in state_dict:
        qe = state_dict[qe_key]
        if qe.shape[0] >= args.num_queries:
            # keep first 50 queries
            state_dict[qe_key] = qe[:args.num_queries, :]
            print(f"Sliced query_embed from {qe.shape[0]} to {state_dict[qe_key].shape[0]}")
        else:
            # (not your case, but for completeness)
            print(f"Warning: checkpoint has fewer queries ({qe.shape[0]}) than requested ({args.num_queries}).")

    # for dec_layers=3 we also keep strict=False to ignore extra decoder layers
    load_result = model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded with strict=False.")
    print("  Missing keys:", len(load_result.missing_keys))
    print("  Unexpected keys:", len(load_result.unexpected_keys))



    # ------------------------------------------------------------------
    # 5) Optimizer with smaller LR for backbone
    # ------------------------------------------------------------------
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and "backbone" not in n
            ],
            "lr": 1e-4,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and "backbone" in n
            ],
            "lr": 1e-5,
        },
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)

    num_epochs = 5  # keep this small for mini-COCO fine-tuning

    # ------------------------------------------------------------------
    # 6) Training + validation loop
    # ------------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # DETR uses NestedTensor internally
            samples = nested_tensor_from_tensor_list(images).to(device)

            # Forward pass
            outputs = model(samples)

            # Loss via DETR's original criterion (Hungarian matching)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                avg = running_loss / 50
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Iter [{i+1}/{len(train_loader)}] "
                    f"Loss: {avg:.4f}"
                )
                running_loss = 0.0

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                samples = nested_tensor_from_tensor_list(images).to(device)
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                val_loss += sum(loss_dict.values()).item()

        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}: val loss = {val_loss:.4f}")

        # Save checkpoint each epoch
        ckpt_out = REPO_ROOT / f"detr_r50_mini_finetune_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")


if __name__ == "__main__":
    main()
