"""
Embed fingerprints into images using a trained StegaStamp encoder.

- Flat-folder OR CIFAR-10 loading (toggle with --use_cifar10)
- Optional CelebA preprocessing
- Identical vs. random per-image fingerprints
- Optional on-the-fly check via decoder

Typical use:
------------
!python \path\to\embed_watermark_cifar10.py \
  --use_cifar10 \
  --cifar10_root ./_data \
  --cifar10_limit 10000 \
  --encoder_path path/to/cifar10_10k_encoder.pth \
  --decoder_path path/to/cifar10_10k_decoder.pth \
  --check \
  --output_dir outputs/cifar10_10k_watermarked \
  --output_dir_note outputs/cifar10_10k_watermarked_notes \
  --image_resolution 32 \
  --batch_size 64

!python \path\to\embed_watermark_cifar10.py \
  --use_cifar10 \
  --cifar10_root ./_data \
  --encoder_path path/to/encoder.pth \
  --decoder_path path/to/decoder.pth \
  --check \
  --output_dir outputs/cifar10_watermarked \
  --output_dir_note outputs/cifar10_watermarked_notes \
  --image_resolution 32 \
  --batch_size 64

Author: Amanda + Chansen
Citation: https://github.com/yunqing-me/WatermarkDM.git
"""

import argparse
import os
import glob
from time import time
from tqdm import tqdm
import PIL

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--use_celeba_preprocessing", action="store_true",
    help="Use CelebA-specific preprocessing (requires --image_resolution 128).")

parser.add_argument("--encoder_path", type=str, required=True,
    help="Path (or glob) to StegaStamp encoder .pth.")

parser.add_argument("--data_dir", type=str, default=None,
    help="Directory with images (non-recursive by default).")

parser.add_argument("--output_dir", type=str, required=True,
    help="Directory to save watermarked images.")

parser.add_argument("--output_dir_note", type=str, required=True,
    help="Directory to save fingerprint metadata (embedded_fingerprints.txt).")

parser.add_argument("--image_resolution", type=int, required=True,
    help="Height and width of square images.")

parser.add_argument("--identical_fingerprints", action="store_true",
    help="If set, use identical fingerprints per batch; else random per image.")

parser.add_argument("--check", action="store_true",
    help="Validate fingerprint detection accuracy with a decoder.")

parser.add_argument("--decoder_path", type=str, default=None,
    help="Path (or glob) to StegaStamp decoder .pth (required if --check).")

parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

parser.add_argument("--seed", type=int, default=42, help="Random seed for fingerprints.")

parser.add_argument("--cuda", type=str, default="cuda",
    help="Device: -1 or 'cpu', 'cuda', 'cuda:N', or an int index.")

parser.add_argument("--num_workers", type=int, default=None,
    help="DataLoader workers (auto: 2 on CUDA, 0 on CPU).")

# Cifar10 Dataset Arguments
parser.add_argument("--use_cifar10", action="store_true",
                    help="Load images from torchvision.datasets.CIFAR10 instead of --data_dir.")

parser.add_argument("--cifar10_root", type=str, default="./_data",
                    help="Root to the CIFAR-10 cache (used with --use_cifar10).")

# Cifar10 subset (mirror train_cifar10.py)
parser.add_argument("--subset_size", type=int, default=0,
                    help="If >0, sample this many training images from CIFAR-10.")
parser.add_argument("--subset_seed", type=int, default=1337,
                    help="RNG seed for CIFAR-10 subset sampling.")


# Opt-in: keep default non-recursive, lowercase-only to match your original logic.
parser.add_argument("--recursive", action="store_true",
    help="If set, recurse into subdirectories and accept upper-case extensions.")

args = parser.parse_args()

# -----------------------------
# Device
# -----------------------------
def parse_device(spec: str) -> torch.device:
    s = str(spec).strip().lower()
    if s in ("-1", "cpu"):
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{int(s)}")
    if s.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = parse_device(args.cuda)
print(f"Using device: {device}")

# -----------------------------
# Utils
# -----------------------------
def resolve_ckpt(path_or_glob: str) -> str:
    """Return a single checkpoint path. If glob, pick newest by mtime."""
    if os.path.exists(path_or_glob):
        return path_or_glob
    candidates = sorted(glob.glob(path_or_glob), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matches: {path_or_glob}")
    return candidates[0]

def generate_random_fingerprints(fingerprint_size: int, batch_size: int) -> torch.Tensor:
    """(batch_size, fingerprint_size) binary {0,1} tensor."""
    return torch.zeros((batch_size, fingerprint_size), dtype=torch.float32).random_(0, 2)


class FlatFolder(Dataset):
    def __init__(self, root, recursive=False, transform=None):
        exts = ["*.png", "*.jpg", "*.jpeg"]
        pats = [f"**/{e}" if recursive else e for e in exts]
        files = []
        for pat in pats:
            files.extend(glob.glob(os.path.join(root, pat), recursive=recursive))
        self.files = sorted(files)
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img) if self.transform else img
        return img, 0

def build_transform():
    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, "CelebA preprocessing requires image_resolution=128."
        return transforms.Compose([transforms.CenterCrop(148), transforms.Resize(128), transforms.ToTensor()])
    return transforms.Compose([
        transforms.Resize(args.image_resolution),
        transforms.CenterCrop(args.image_resolution),
        transforms.ToTensor()
    ])


def build_loader_and_namer():
    # CIFAR-10 at native 32x32: just ToTensor; otherwise use generic pipeline.
    transform_pipeline = (
        transforms.ToTensor()
        if (args.use_cifar10 and args.image_resolution == 32)
        else build_transform()
    )

    if args.use_cifar10:
        print(f"[Data] Loading CIFAR-10 from {args.cifar10_root}")
        base_dataset = CIFAR10(
            root=args.cifar10_root,
            train=True,
            download=True,
            transform=transform_pipeline,
        )

        if getattr(args, "cifar10_limit", None) is not None:
            n = min(args.cifar10_limit, len(base_dataset))
            indices = list(range(n))
            dataset = Subset(base_dataset, indices)
            print(
                f"[Data] Using CIFAR-10 subset: first {n} images "
                f"(of {len(base_dataset)} total)."
            )
        else:
            dataset = base_dataset
            print(
                f"[Data] Using full CIFAR-10 train split: "
                f"{len(base_dataset)} images."
            )

        def name_fn(index: int) -> str:
            # Logical index in [0, len(dataset) - 1]
            return f"{index:05d}.png"

    else:
        if not args.data_dir:
            raise ValueError("--data_dir is required when --use_cifar10 is not set.")
        print(
            f"[Data] Loading flat images from {args.data_dir} "
            f"(recursive={args.recursive})"
        )
        dataset = FlatFolder(
            args.data_dir,
            recursive=args.recursive,
            transform=transform_pipeline,
        )

        def name_fn(index: int) -> str:
            return (
                os.path.splitext(os.path.basename(dataset.files[index]))[0]
                + ".png"
            )

    if args.num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0
    else:
        num_workers = args.num_workers


    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return dataset, data_loader, name_fn
# -----------------------------
# Models
# -----------------------------
def _split_encoder_ckpt(obj):
    """
    Handle both:
    - raw state_dict
    - dict with {'encoder_state': ..., 'config': ...} style
    """
    if isinstance(obj, dict):
        # new style
        if "encoder_state" in obj:
            return obj["encoder_state"], obj.get("config", None)
        # old style: state_dict with weight tensors
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, None
    raise ValueError("Unrecognized encoder checkpoint format.")


def _split_decoder_ckpt(obj):
    if isinstance(obj, dict):
        if "decoder_state" in obj:
            return obj["decoder_state"]
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized decoder checkpoint format.")


def load_models():
    from models import StegaStampEncoder, StegaStampDecoder

    # ----- encoder -----
    encoder_path = resolve_ckpt(args.encoder_path)
    enc_obj = torch.load(encoder_path, map_location="cpu")
    encoder_state_dict, enc_cfg = _split_encoder_ckpt(enc_obj)

    fingerprint_size = encoder_state_dict["secret_dense.weight"].shape[-1]
    print(f"[Model] Inferred fingerprint_size={fingerprint_size}")

    if enc_cfg is not None and "image_resolution" in enc_cfg:
        cfg_res = enc_cfg["image_resolution"]
        if cfg_res != args.image_resolution:
            print(
                f"[Warning] Checkpoint image_resolution={cfg_res} "
                f"but CLI image_resolution={args.image_resolution}."
            )

    hide_net = StegaStampEncoder(
        args.image_resolution, 3, fingerprint_size, return_residual=False
    ).to(device)
    hide_net.load_state_dict(encoder_state_dict)
    hide_net.eval()

    # ----- decoder (optional check) -----
    reveal_net = None
    if args.check:
        if not args.decoder_path:
            raise ValueError("--check requires --decoder_path")
        decoder_path = resolve_ckpt(args.decoder_path)
        dec_obj = torch.load(decoder_path, map_location="cpu")
        decoder_state_dict = _split_decoder_ckpt(dec_obj)

        reveal_net = StegaStampDecoder(
            args.image_resolution, 3, fingerprint_size
        ).to(device)
        reveal_net.load_state_dict(decoder_state_dict)
        reveal_net.eval()

    return hide_net, reveal_net, fingerprint_size


# -----------------------------
# Main embed loop
# -----------------------------
def embed_fingerprints():
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_note, exist_ok=True)

    dataset, data_loader, name_fn = build_loader_and_namer()
    hide_net, reveal_net, fingerprint_size = load_models()

    torch.manual_seed(args.seed)

    fixed_fingerprints = None
    if args.identical_fingerprints:
        fixed_fingerprints = generate_random_fingerprints(fingerprint_size, 1)
        fixed_fingerprints = fixed_fingerprints.view(1, fingerprint_size).expand(
            args.batch_size, fingerprint_size
        ).to(device)

    fingerprint_records = []
    bitwise_accuracy_sum = 0.0
    num_images = 0

    total_image_count = len(dataset)
    num_batches = len(data_loader)
    tqdm.write(
        f"[Embed] Total images to fingerprint: {total_image_count} "
        f"in {num_batches} batches (epoch 1/1)"
    )

    print("Fingerprinting the images...")
    with torch.no_grad():
        for batch_index, (images, _) in tqdm(enumerate(data_loader), total=num_batches):
            images = images.to(device)
            batch_size = images.size(0)

            if args.identical_fingerprints:
                fingerprints = fixed_fingerprints[:batch_size]
            else:
                fingerprints = generate_random_fingerprints(fingerprint_size, batch_size).to(device)

            fingerprinted_images = hide_net(fingerprints, images)

            start_index = batch_index * args.batch_size
            for i in range(batch_size):
                filename = name_fn(start_index + i)
                save_image(
                    fingerprinted_images[i].detach().cpu(),
                    os.path.join(args.output_dir, filename),
                    padding=0,
                )
                fingerprint_records.append(
                    (filename, fingerprints[i].detach().cpu().long().tolist())
                )

            if args.check and reveal_net is not None:
                logits = reveal_net(fingerprinted_images)
                detected = (logits > 0).long()  # logit threshold at 0
                bitwise_accuracy_sum += (
                    (detected[:batch_size].cpu() == fingerprints.cpu().long())
                    .float()
                    .mean(dim=1)
                    .sum()
                    .item()
                )
            num_images += batch_size
            images_embedded_so_far = min(num_images, total_image_count)

            tqdm.write(
                f"[Embed] Epoch 1/1 - batch {batch_index + 1}/{num_batches} "
                f"- images {images_embedded_so_far}/{total_image_count}"
            )

    with open(os.path.join(args.output_dir_note, "embedded_fingerprints.txt"), "w") as f:
        for filename, bits in fingerprint_records:
            f.write(f"{filename} {''.join(map(str, bits))}\n")

    if args.check and num_images:
        print(
            f"Bitwise accuracy on fingerprinted images: "
            f"{bitwise_accuracy_sum / num_images:.4f}"
        )


def main():
    start_time = time()
    embed_fingerprints()
    print(f"Done in {time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
