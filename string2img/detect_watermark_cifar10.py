"""
Detect Watermarks from Images Using a Pretrained StegaStamp Decoder.

This script loads a batch of fingerprinted images and attempts to decode
the embedded fingerprint from each using a trained decoder model. Outputs
bitwise accuracy and optionally prints or stores the detected fingerprints.

Usage (Colab-friendly):
    Modify the argparse default values or use CLI for terminal runs.

Author: Your Project Team
"""

import argparse
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with fingerprinted images.")
parser.add_argument("--output_dir", type=str, help="Directory to save detected fingerprints.")
parser.add_argument("--image_resolution", type=int, default=32, help="Input image resolution (square).")
parser.add_argument("--decoder_path", type=str, help="Path to trained StegaStamp decoder.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
parser.add_argument("--cuda", type=int, default=0, help="Use CUDA device index or -1 for CPU.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--ground_truth_fp", type=str, default=None, help="Optional ground truth binary string for bitwise accuracy.")
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda != -1 and torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset Definition
# -----------------------------
class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.filenames = sorted(
            glob.glob(os.path.join(data_dir, "*.png"))
            + glob.glob(os.path.join(data_dir, "*.jpg"))
            + glob.glob(os.path.join(data_dir, "*.jpeg"))
        )
        if not self.filenames:
            raise RuntimeError(f"No images found in {data_dir}")
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert("RGB")
        return self.transform(image), 0

    def __len__(self):
        return len(self.filenames)

# -----------------------------
# Model Loading
# -----------------------------
def load_decoder():
    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path, map_location=device)

    fingerprint_size = state_dict["dense.2.weight"].shape[0]
    decoder = StegaStampDecoder(
        resolution=args.image_resolution,
        IMAGE_CHANNELS=3,
        fingerprint_size=fingerprint_size
    )
    decoder.load_state_dict(state_dict)
    decoder.to(device).eval()
    return decoder, fingerprint_size

# -----------------------------
# Fingerprint Detection
# -----------------------------
def extract_fingerprints(decoder, fingerprint_size):
    transform = transforms.Compose([
        transforms.Resize(args.image_resolution),
        transforms.CenterCrop(args.image_resolution),
        transforms.ToTensor(),
    ])

    dataset = CustomImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_fingerprints = []
    accuracy_sum = 0

    # Prepare GT if provided
    if args.ground_truth_fp:
        assert len(args.ground_truth_fp) == fingerprint_size, "GT fingerprint size mismatch"
        gt_tensor = torch.tensor([int(b) for b in args.ground_truth_fp], dtype=torch.float).unsqueeze(0)
        gt_tensor = gt_tensor.expand(args.batch_size, -1).to(device)

    print(f"Running detection on {len(dataset)} images...")

    for images, _ in tqdm(dataloader, desc="Detecting"):
        images = images.to(device)
        with torch.no_grad():
            output = decoder(images)
            predicted = (output > 0).float()

        if args.ground_truth_fp:
            batch_acc = 1.0 - torch.mean(torch.abs(predicted - gt_tensor[:images.size(0)]))
            accuracy_sum += batch_acc.item() * images.size(0)

        all_fingerprints.append(predicted.cpu())

    all_fingerprints = torch.cat(all_fingerprints, dim=0)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "detected_fingerprints.txt"), "w") as f:
            for idx, fp in enumerate(all_fingerprints):
                filename = os.path.basename(dataset.filenames[idx])
                bits = "".join(str(int(b)) for b in fp.tolist())
                f.write(f"{filename} {bits}\n")

    if args.ground_truth_fp:
        final_accuracy = accuracy_sum / len(dataset)
        print(f"\nBitwise accuracy: {final_accuracy:.4f}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    decoder, fingerprint_size = load_decoder()
    extract_fingerprints(decoder, fingerprint_size)
