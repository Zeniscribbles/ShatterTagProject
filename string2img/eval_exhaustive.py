
"""
eval_exhaustive.py

Exhaustive evaluation harness for ShatterTag/StegaStamp models.
- Deterministically embed a fingerprint per image, then decode.
- Log per-image metrics to CSV (and optional NPZ tensors).
- Save images for failures only (or all/none) to control disk.
- Optionally tar-compress saved images for space and portability.

This lets you keep exhaustive *metrics* while saving far fewer images.
You can always regenerate the exact images later using the saved seeds.

==========================================================================
TYPICAL USAGE:

python eval_exhaustive.py \
  --data_dir /path/to/images \
  --encoder_path /path/to/checkpoints/encoder_last.pth \
  --decoder_path /path/to/checkpoints/decoder_last.pth \
  --output_dir ./eval_out/run1 \
  --image_resolution 32 \
  --bit_length 64 \
  --batch_size 128 \
  --cuda cuda \
  --seed 123 \
  --save_images failures \        # none | failures | all
  --save_npz \                    # store tensors for failures (optional)
  --tar_images \                  # compress saved images (optional)
  --threshold 0.99                # what counts as a "failure"

EXPECTED OUTPUTS:
eval_out/run1/metrics.csv      — one row per image (exhaustive ground truth for your thesis).

eval_out/run1/images/          — only failures by default (or everything if you insist).

eval_out/run1/saved_images.tar — optional tar with all saved images.

eval_out/run1/eval_config.json — reproducibility snapshot.
============================================================================

Author: Chansen + Amanda
"""

import argparse, os, glob, json, tarfile, io, random
from os.path import join
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import PIL
import csv

import models


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_device(arg):
    if isinstance(arg, int):
        return torch.device(f"cuda:{arg}") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(arg, str):
        s = arg.strip().lower()
        if s == "cpu":
            return torch.device("cpu")
        if s == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if s.startswith("cuda:"):
            idx = int(s.split(":")[1])
            return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
        if s.isdigit():
            idx = int(s)
            return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageFolder(Dataset):
    def __init__(self, root, image_resolution):
        pats = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
        files = sum((glob.glob(os.path.join(root, p), recursive=True) for p in pats), [])
        self.files = sorted(files)
        if not self.files:
            raise RuntimeError(f"No images found under {root}")
        self.tf = transforms.Compose([
            transforms.Resize((image_resolution, image_resolution)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = PIL.Image.open(path).convert("RGB")
        except Exception:
            # Skip corrupted; in practice, you'd log it
            return self.__getitem__((idx + 1) % len(self))
        return self.tf(img), path


def bits_to_str(bits: torch.Tensor) -> str:
    # Save as compact hex for CSV
    b = (bits.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()
    # pack 8 bits per byte
    out = 0
    arr = []
    for i, bit in enumerate(b):
        out = (out << 1) | bit
        if (i % 8) == 7:
            arr.append(out)
            out = 0
    if len(b) % 8 != 0:
        # flush remaining
        out = out << (8 - (len(b) % 8))
        arr.append(out)
    return "".join(f"{x:02x}" for x in arr)


def bit_accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - torch.mean(torch.abs(a - b)).item()


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    # Simple AUC via trapezoidal rule on ROC; labels in {0,1}
    order = np.argsort(-scores)
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / (tps[-1] if tps[-1] > 0 else 1.0)
    fpr = fps / (fps[-1] if fps[-1] > 0 else 1.0)
    # integrate
    return float(np.trapz(tpr, fpr))


def maybe_open_tar(output_dir, enable_tar):
    if not enable_tar:
        return None
    tar_path = os.path.join(output_dir, "saved_images.tar")
    return tarfile.open(tar_path, mode="w")


def add_to_tar(tar, folder, filename):
    if tar is None:
        return
    tar.add(os.path.join(folder, filename), arcname=filename)


def main():
    ap = argparse.ArgumentParser(description="Exhaustive evaluation for ShatterTag/StegaStamp")
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--encoder_path", required=True, type=str)
    ap.add_argument("--decoder_path", required=True, type=str)
    ap.add_argument("--output_dir", default="./eval_out", type=str)
    ap.add_argument("--image_resolution", default=32, type=int)
    ap.add_argument("--bit_length", default=64, type=int)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--cuda", default="cuda", type=str)
    ap.add_argument("--seed", default=123, type=int)
    ap.add_argument("--limit_images", default=None, type=int, help="Optional cap for quick runs")
    ap.add_argument("--save_images", choices=["none","failures","all"], default="failures")
    ap.add_argument("--save_npz", action="store_true", help="Save raw tensors per failure")
    ap.add_argument("--tar_images", action="store_true", help="Tar-compress saved images")
    ap.add_argument("--threshold", default=0.99, type=float, help="Bitwise accuracy threshold for 'failure'")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    set_seed(args.seed)
    device = parse_device(args.cuda)
    use_cuda = device.type == "cuda"

    # Models
    IMAGE_CHANNELS = 3
    encoder = models.StegaStampEncoder(args.image_resolution, IMAGE_CHANNELS, args.bit_length, return_residual=False)
    decoder = models.StegaStampDecoder(args.image_resolution, IMAGE_CHANNELS, args.bit_length)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location="cpu"))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location="cpu"))
    encoder.to(device).eval()
    decoder.to(device).eval()

    # Data
    ds = ImageFolder(args.data_dir, args.image_resolution)
    if args.limit_images is not None:
        ds.files = ds.files[:args.limit_images]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)

    # Outputs
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    tar = maybe_open_tar(args.output_dir, args.tar_images)

    # CSV for per-image metrics
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["index","path","seed","bit_length","bit_acc","l2","bce","fp_hex","pred_hex"])

        idx_base = 0
        with torch.no_grad():
            for batch, paths in dl:
                batch = batch.to(device, non_blocking=True)
                bs = batch.size(0)

                # Deterministic fingerprint per image index
                # (reproducible across machines/runs)
                torch.manual_seed(args.seed + idx_base)
                fingerprints = torch.randint(0, 2, (bs, args.bit_length), dtype=torch.float, device=device)

                with torch.cuda.amp.autocast(enabled=use_cuda):
                    watermarked = encoder(fingerprints, batch)
                    logits = decoder(watermarked)

                    l2 = torch.mean((watermarked - batch) ** 2, dim=(1,2,3))
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.view(bs, -1), fingerprints.view(bs, -1), reduction="none"
                    ).mean(dim=1)

                preds = (logits > 0).float()
                accs = 1.0 - torch.mean(torch.abs(fingerprints - preds), dim=1)

                # Save rows
                for i in range(bs):
                    fp_hex = bits_to_str(fingerprints[i])
                    pr_hex = bits_to_str(preds[i])
                    writer.writerow([idx_base + i, paths[i], args.seed, args.bit_length,
                                     float(accs[i].item()), float(l2[i].item()), float(bce[i].item()),
                                     fp_hex, pr_hex])

                # Save images according to policy
                if args.save_images != "none":
                    # failures if acc < threshold
                    mask = accs < args.threshold if args.save_images == "failures" else torch.ones_like(accs).bool()
                    chosen = torch.nonzero(mask).view(-1).tolist()
                    for j in chosen:
                        # Save individual image and residual
                        base = f"{idx_base + j}"
                        save_image(watermarked[j].detach().cpu(), os.path.join(images_dir, base + "_wm.png"), normalize=True)
                        res = (watermarked[j] - batch[j]).detach().cpu()
                        save_image(res, os.path.join(images_dir, base + "_residual.png"), normalize=True)
                        if args.save_npz:
                            np.savez_compressed(os.path.join(images_dir, base + ".npz"),
                                                watermarked=watermarked[j].detach().cpu().numpy(),
                                                clean=batch[j].detach().cpu().numpy(),
                                                logits=logits[j].detach().cpu().numpy(),
                                                fp=fingerprints[j].detach().cpu().numpy())

                idx_base += bs

    if tar is not None:
        # Add all saved images to tar then (optionally) you can delete the folder outside this script
        for fn in os.listdir(images_dir):
            add_to_tar(tar, images_dir, fn)
        tar.close()

    print("Evaluation complete.")
    print(f"Per-image metrics: {csv_path}")
    print(f"Saved images dir:  {images_dir}")
    if tar is not None:
        print(f"TAR archive:      {os.path.join(args.output_dir, 'saved_images.tar')}")
    print("You can regenerate exact fingerprints from (seed, index, bit_length).")


if __name__ == "__main__":
    main()
