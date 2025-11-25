# DETR Environment Setup (RunPod)

## 1. Pod Template
- GPU: A40 (48GB)
- Template: PyTorch 2.4.0 + Python 3.11 + CUDA 12.4 + Ubuntu 22
- No virtualenv / conda used.

## 2. System Packages
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev git
```

## 3. Clone Repo
```bash
git clone https://github.com/irvinghu07/detr.git
cd detr
```

## 4. Install Python Dependencies
Do NOT use cocoapi/panopticapi from requirements (they fail to build).
Comment them out in requirements.txt.

Install working dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install cython numpy
pip install "pycocotools>=2.0.6"
pip install -r requirements.txt
```
(Install panopticapi only if needed for panoptic segmentation.)
