import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import detr_resnet50
import torch.optim as optim
from pathlib import Path


class CocoDetrTrain(CocoDetection):
    def __init__(self, img_folder, ann_file):
        self._transform = T.ToTensor()  # basic, model will handle resizing etc.
        super().__init__(img_folder, ann_file)

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)   # img: PIL, anns: list of dicts
        img = self._transform(img)

        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            # DETR doesn’t like empty targets; you can skip or give dummy
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(self.ids[idx])
        }
        return img, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_root = Path("coco")  # adjust if needed

    train_dataset = CocoDetrTrain(
        img_folder=coco_root / "train2017",
        ann_file=coco_root / "annotations" / "instances_train2017.json"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,   # adjust for your GPU
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # --- Load pretrained DETR-ResNet50 (COCO weights) ---
    model = detr_resnet50(weights="DETR_ResNet50_Weights.COCO_V1")
    model.to(device)

    # Fine-tune entire model, but typically we use smaller LR for backbone
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" not in n],
            "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n],
            "lr": 1e-5,
        },
    ]

    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)

    num_epochs = 5  # start small, you can increase later

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward: returns dict of losses in training mode
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            if (i + 1) % 100 == 0:
                avg = running_loss / 100
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Iter [{i+1}/{len(train_loader)}] "
                    f"Loss: {avg:.4f}"
                )
                running_loss = 0.0

        # Save checkpoint per epoch
        ckpt_path = f"detr_r50_finetune_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
