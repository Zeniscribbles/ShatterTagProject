"""
Detect watermarks with a pretrained StegaStamp decoder.

Minimal changes from original:
- Flat-folder loader for Colab CIFAR-10 export (no class subdirs).
- Optional --ground_truth_fp for fixed-bitstring evaluation (Method A).
- Match train/embed preprocessing (Resize -> CenterCrop -> ToTensor).

Usage:
python detect_watermark_cifar10.py \
  --data_dir "/content/drive/MyDrive/ShatterTagProject/output/cifar10_run1/embedded" \
  --image_resolution 32 \
  --decoder_path "/content/drive/.../checkpoints/*_decoder_last.pth" \
  --batch_size 64 \
  --cuda 0 \
  --ground_truth_fp "0110...<64 bits>..." \
  --output_dir "/content/drive/.../output/cifar10_run1/detect"

Author: Amanda + Chansen
Citation: https://github.com/yunqing-me/WatermarkDM.git
"""
import argparse
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution",
    type=int,
    help="Height and width of square images.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--encoder_path", type=str, default=None,
    help="(Optional) Matching StegaStamp encoder .pth used for training; used to infer bit length."
)

# NEW: fixed watermark for detection: If provided, the same bitstring is used for accuracy.
parser.add_argument("--ground_truth_fp", type=str, default=None,
                    help="Fixed binary string (e.g., '0101...') used as GT for all images.")
args = parser.parse_args()

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms


# -----------------------------
# Device
# -----------------------------
if int(args.cuda) == -1 or not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{int(args.cuda)}" if isinstance(args.cuda, int) else "cuda")


# -----------------------------
# Dataset: flat Colab folder
# -----------------------------
class CustomImageFolder(Dataset):
    """
    CHANGE (Colab): read images from a single flat directory (e.g., 00000.png..).
    Force RGB to match decoder’s 3-channel expectation.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB") # 3 channel
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

# -----------------------------
# Data loading (match train/embed preprocessing)
# -----------------------------
def load_data():
    global dataset, dataloader
    transform = transforms.Compose([
        transforms.Resize(args.image_resolution),
        transforms.CenterCrop(args.image_resolution),
        transforms.ToTensor(),
    ])
    print(f"Loading image folder {args.data_dir} ...")
    t0 = time()
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - t0:.2f}s")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)


# -----------------------------
# Model loading 
# -----------------------------
def load_decoder():
    global RevealNet, FINGERPRINT_SIZE
    from models import StegaStampDecoder

    # Always load on CPU first; we’ll move the model to device afterward.
    dec_sd = torch.load(args.decoder_path, map_location="cpu")

    # --- Option B: infer bit-length from encoder checkpoint when provided ---
    FINGERPRINT_SIZE = None
    if args.encoder_path is not None and os.path.exists(args.encoder_path):
        enc_sd = torch.load(args.encoder_path, map_location="cpu")
        # Original StegaStamp encoders keep the bit size on the secret/projection layer
        if "secret_dense.weight" in enc_sd:
            FINGERPRINT_SIZE = enc_sd["secret_dense.weight"].shape[-1]
        else:
            # Fallback: try to find a key that ends with 'secret_dense.weight'
            cand = [k for k in enc_sd.keys() if k.endswith("secret_dense.weight")]
            if cand:
                FINGERPRINT_SIZE = enc_sd[cand[0]].shape[-1]

    # --- If no encoder provided (or fallback failed), infer from the decoder checkpoint ---
    if FINGERPRINT_SIZE is None:
        # Pick the linear weight whose out_features equals the smallest 2D .weight[0]
        # In typical StegaStamp, MLP is 512->512-><bits>; min out_features = <bits>.
        candidates = [
            (k, v.shape) for k, v in dec_sd.items()
            if k.endswith(".weight") and v.ndim == 2
        ]
        if not candidates:
            raise RuntimeError("Could not find any 2D linear weights in decoder checkpoint.")
        FINGERPRINT_SIZE = min(shp[0] for _, shp in candidates)

    # Build and load the decoder with the inferred bit-length
    RevealNet = StegaStampDecoder(
        args.image_resolution, 3, FINGERPRINT_SIZE
    ).to(device)
    RevealNet.load_state_dict(dec_sd)
    RevealNet.eval()


# -----------------------------
# Detection (fixed GT optional)
# -----------------------------
def extract_fingerprints():
    total_correct = 0.0
    total_seen = 0

    # If a fixed GT bitstring is provided, parse it once to a 1×F tensor
    gt_bits_1x = None
    if args.ground_truth_fp is not None:
        bits = [int(b) for b in args.ground_truth_fp.strip()]
        if len(bits) != FINGERPRINT_SIZE:
            raise ValueError(
                f"--ground_truth_fp length ({len(bits)}) "
                f"does not match decoder fingerprint size ({FINGERPRINT_SIZE})."
            )
        gt_bits_1x = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)

    # Optional: collect predictions for saving
    detected_lines = []

    with torch.no_grad():
        for b, (images, _) in enumerate(tqdm(dataloader, desc="Detecting")):
            images = images.to(device)
            logits = RevealNet(images)
            preds = (logits > 0).float()           # B×F, values in {0,1}

            # Save predictions (filename + bits) if requested
            for i in range(images.size(0)):
                idx = total_seen + i
                fname = os.path.basename(dataset.filenames[idx])
                bits = "".join(str(int(x)) for x in preds[i].cpu().tolist())
                detected_lines.append(f"{fname} {bits}")

            # If GT was provided, compute batch accuracy vs the same fixed GT
            if gt_bits_1x is not None:
                gt_batch = gt_bits_1x.expand(images.size(0), -1)  # B×F
                batch_acc = 1.0 - torch.mean(torch.abs(preds - gt_batch)).item()
                total_correct += batch_acc * images.size(0)

            total_seen += images.size(0)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "detected_fingerprints.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(detected_lines))
        print(f"Saved predictions to: {out_path}")

    if gt_bits_1x is not None and total_seen > 0:
        print(f"\nBitwise accuracy (vs fixed GT): {total_correct / total_seen:.4f}")


def main():
    load_decoder()
    load_data()
    extract_fingerprints()

if __name__ == "__main__":
    main()