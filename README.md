# Shatter Tag Project

Project Overview:
This project is a fully reworked and generalized watermarking framework derived from StegaStamp, designed for embedding imperceptible binary fingerprints into images. Unlike traditional watermarking tools focused solely on robustness, our system is being adapted for controlled fragility — enabling it to act as a tamper seal in addition to traditional ownership verification.

We began by stripping out unnecessary diffusion model dependencies and hardcoded CIFAR-specific logic. What emerged is a clean, modular, Colab-compatible pipeline for:

Embedding binary fingerprints (watermarks) into image datasets

Decoding them reliably under benign conditions

Logging visual and scalar results across training epochs

Saving both fingerprinted images and associated bitstrings for verification

# StegaStamp Robust Watermarking Toolkit

A flexible, production-ready implementation of a deep learning-based image watermarking system—adapted from prior work and enhanced for robustness, modularity, and usability in both **Google Colab** and **CLI environments**.

Supports:
- Robust watermark embedding using spatial binary fingerprints
- Fingerprint recovery with bitwise accuracy tracking
- Easy training on custom datasets
- Tamper detection research (fragility tuning supported)
- Compatibility with Colab, local scripts, and CLI usage

---

## Installation (Colab or Local)

```bash
# Clone the repo
git clone https://github.com/yourusername/stegastamp-robust.git

# Install dependencies
pip install -r requirements.txt
```

---

## File Overview

| File/Folder | Purpose |
|-------------|---------|
| `models.py` | Encoder/Decoder models for StegaStamp |
| `train_cifar10.py` | Main training loop (works on any image dataset, not just CIFAR) |
| `embed_watermark.py` | Embed fingerprints into a dataset |
| `detect_watermark.py` | Detect fingerprints and report accuracy |
| `README.md` | You're here. |
| `saved_images/` | Sample fingerprinted images (output) |
| `checkpoints/` | Model checkpoints |

---

## Google Colab Usage

### 1. Upload or mount your dataset

```python
!mkdir -p data/myimages
# Upload images to this folder or use Google Drive
```

### 2. Train the watermarking model

```python
from train_cifar10 import main as train_main
from argparse import Namespace

args = Namespace(
    data_dir="data/myimages",
    output_dir="./output",
    image_resolution=64,
    bit_length=64,
    batch_size=32,
    num_epochs=10,
    lr=1e-4,
    cuda="cuda",
    l2_loss_weight=1.0,
    BCE_loss_weight=1.0,
    l2_loss_ramp=1000,
    l2_loss_await=0,
    plot_logging_steps=500  # smaller for colab debug
)

train_main(args)
```

### 3. Embed watermarks into your images

```python
from embed_watermark import embed_watermarks

embed_watermarks(
    data_dir="data/myimages",
    encoder_path="./output/checkpoints/your_encoder.pth",
    output_dir="./output/watermarked_images",
    output_dir_note="./output/fingerprint_notes",
    image_resolution=64,
    bit_length=64,
    batch_size=32,
    identical_fingerprints=False,
    check=True,
    decoder_path="./output/checkpoints/your_decoder.pth",
    seed=42,
    device="cuda"
)
```

### 4. Detect fingerprints

```python
from detect_watermark import detect_watermarks

detect_watermarks(
    data_dir="./output/watermarked_images",
    decoder_path="./output/checkpoints/your_decoder.pth",
    output_dir="./output/detection_results",
    image_resolution=64,
    batch_size=32,
    seed=42,
    device="cuda",
    check=True
)
```

---

## CLI Usage

### Train a model

```bash
python train_cifar10.py \
  --data_dir data/myimages \
  --output_dir ./output \
  --image_resolution 64 \
  --bit_length 64 \
  --batch_size 32 \
  --num_epochs 20 \
  --lr 1e-4 \
  --cuda cuda
```

### Embed watermarks

```bash
python embed_watermark.py \
  --data_dir data/myimages \
  --encoder_path ./output/checkpoints/your_encoder.pth \
  --output_dir ./output/watermarked_images \
  --output_dir_note ./output/fingerprint_notes \
  --image_resolution 64 \
  --bit_length 64 \
  --batch_size 32 \
  --identical_fingerprints \
  --check \
  --decoder_path ./output/checkpoints/your_decoder.pth \
  --cuda 0
```

### Detect watermarks

```bash
python detect_watermark.py \
  --data_dir ./output/watermarked_images \
  --decoder_path ./output/checkpoints/your_decoder.pth \
  --output_dir ./output/detection_results \
  --image_resolution 64 \
  --batch_size 32 \
  --check \
  --cuda 0
```

---

## Testing Robustness (Next Steps)

The current architecture is designed for **robust watermarking**. That means:

- Fingerprints should **survive JPEG compression**, light cropping, resizing, and noise.
- You can later adapt it for **fragile watermarking** by modifying training with augmentation attacks or targeted loss functions.

Coming soon:
- Attack test suite (blurring, JPEG, cropping, noise)
- Tamper detection toggles
- Adaptive fragility for integrity validation (ShatterTag-style)

---

## Citation

This repo is inspired by StegaStamp and extended toward tamper-evident watermarking, building toward the research goals of:

> **ShatterTag**: https://arxiv.org/abs/2303.10137  
> Deep learning-based visual integrity authentication via fragile watermarking

---

## Contact

Built and refactored by Chansen, with upgrades for practicality, clarity, and fragility/robustness exploration.
