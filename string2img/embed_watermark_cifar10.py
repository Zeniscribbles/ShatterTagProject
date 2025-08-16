"""
Embed fingerprints into images using a trained StegaStamp encoder.

Keeps original logic:
- optional CelebA preprocessing
- identical vs. random per-image fingerprints
- optional on-the-fly check via decoder
- same function structure: load_data -> load_models -> embed_fingerprints

Colab hardening:
- creates output dirs up front; streams saves (no giant RAM lists)
- robust device parsing; eval()+inference_mode() to cut memory
- safe encoder/decoder checkpoint resolution (supports glob patterns)
- optional case-insensitive recursive loader (flag off by default)

Typical use
-----------
python embed_watermark_cifar10.py \
  --encoder_path "/path/to/*_encoder_last.pth" \
  --data_dir "/path/to/images" \
  --output_dir "/path/to/embedded" \
  --output_dir_note "/path/to/notes" \
  --image_resolution 32 \
  --batch_size 64 \
  --seed 42 \
  --identical_fingerprints \
  --check --decoder_path "/path/to/*_decoder_last.pth"

Author: Amanda + Chansen
"""

import argparse
import os
import glob
from time import time
from tqdm import tqdm
import PIL

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true",
    help="Use CelebA-specific preprocessing (requires --image_resolution 128).")
parser.add_argument("--encoder_path", type=str, required=True,
    help="Path (or glob) to StegaStamp encoder .pth.")
parser.add_argument("--data_dir", type=str, required=True,
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
# Opt-in: keep default non-recursive, lowercase-only to match your original logic.
parser.add_argument("--recursive", action="store_true",
    help="If set, recurse into subdirectories and accept upper-case extensions.")
args = parser.parse_args()
BATCH_SIZE = args.batch_size

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
use_cuda = (device.type == "cuda")
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

class CustomImageFolder(Dataset):
    """
    Image loader matching your original logic by default:
    - non-recursive
    - lowercase extensions only
    Enable --recursive to recurse and accept any case.
    """
    def __init__(self, data_dir, transform=None, recursive=False):
        self.data_dir = data_dir
        if not recursive:
            files = (glob.glob(os.path.join(data_dir, "*.png")) +
                     glob.glob(os.path.join(data_dir, "*.jpg")) +
                     glob.glob(os.path.join(data_dir, "*.jpeg")))
        else:
            files = []
            exts = {".png", ".jpg", ".jpeg"}
            for root, _, fns in os.walk(data_dir):
                for fn in fns:
                    if os.path.splitext(fn)[1].lower() in exts:
                        files.append(os.path.join(root, fn))
        self.filenames = sorted(files)
        if not self.filenames:
            raise RuntimeError(f"No images found in {data_dir} (recursive={recursive})")
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

# -----------------------------
# Data
# -----------------------------
def load_data():
    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, \
            f"CelebA preprocessing requires image_resolution=128, got {args.image_resolution}"
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(args.image_resolution),
            transforms.CenterCrop(args.image_resolution),
            transforms.ToTensor(),
        ])

    print(f"Loading image folder {args.data_dir} ...")
    s = time()
    dataset = CustomImageFolder(args.data_dir, transform=transform, recursive=args.recursive)
    print(f"Finished. Loading took {time() - s:.2f}s")
    return dataset

# -----------------------------
# Models
# -----------------------------
def infer_fingerprint_size(state_dict: dict) -> int:
    # Prefer your original key; fall back to a common alternative.
    if "secret_dense.weight" in state_dict:
        return state_dict["secret_dense.weight"].shape[-1]
    # Some forks use a final dense named like 'dense.2.weight'
    dense_keys = [k for k in state_dict.keys() if k.endswith("weight") and (".2." in k or "dense" in k)]
    for k in dense_keys:
        if len(state_dict[k].shape) == 2:
            return state_dict[k].shape[0]
    raise KeyError("Could not infer fingerprint size from checkpoint keys.")

def load_models():
    """Load encoder; optionally load decoder when --check is set. Set both to eval()."""
    from models import StegaStampEncoder, StegaStampDecoder

    enc_path = resolve_ckpt(args.encoder_path)
    enc_sd = torch.load(enc_path, map_location="cpu")
    fp_size = infer_fingerprint_size(enc_sd)

    encoder = StegaStampEncoder(args.image_resolution, 3, fingerprint_size=fp_size, return_residual=False)
    encoder.load_state_dict(enc_sd)
    encoder = encoder.to(device).eval()

    decoder = None
    if args.check:
        if not args.decoder_path:
            raise ValueError("--check was set but --decoder_path is missing.")
        dec_path = resolve_ckpt(args.decoder_path)
        dec_sd = torch.load(dec_path, map_location="cpu")
        decoder = StegaStampDecoder(args.image_resolution, 3, fingerprint_size=fp_size)
        decoder.load_state_dict(dec_sd)
        decoder = decoder.to(device).eval()

    return encoder, decoder, fp_size

# -----------------------------
# Main embed loop (streaming saves; no RAM hoarding)
# -----------------------------
def embed_fingerprints():
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_note, exist_ok=True)

    dataset = load_data()
    encoder, decoder, FINGERPRINT_SIZE = load_models()

    if args.num_workers is None:
        num_workers = 2 if use_cuda else 0
    else:
        num_workers = args.num_workers

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )

    torch.manual_seed(args.seed)

    # Base fingerprint for the identical mode; refreshed per-batch otherwise
    base_fp = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE).to(device)

    total_correct = 0.0
    total_seen = 0

    meta_path = os.path.join(args.output_dir_note, "embedded_fingerprints.txt")
    with open(meta_path, "w") as meta_f, torch.inference_mode():
        for images, _ in tqdm(loader, desc="Embedding"):
            images = images.to(device, non_blocking=True)
            bs = images.size(0)

            if args.identical_fingerprints:
                fp = base_fp[0:1].expand(bs, -1).contiguous()
            else:
                fp = generate_random_fingerprints(FINGERPRINT_SIZE, bs).to(device)

            wm = encoder(fp, images)

            if decoder is not None:
                logits = decoder(wm)
                preds = (logits > 0).float()
                batch_acc = 1.0 - torch.mean(torch.abs(preds - fp)).item()
                total_correct += batch_acc * bs

            # Save per-sample immediately; write metadata line-by-line
            for i in range(bs):
                idx = total_seen + i
                in_base = os.path.basename(dataset.filenames[idx])
                out_name = os.path.splitext(in_base)[0] + ".png"
                out_path = os.path.join(args.output_dir, out_name)
                save_image(wm[i].detach().cpu(), out_path, normalize=True)

                bits = "".join(map(str, fp[i].detach().cpu().long().tolist()))
                meta_f.write(f"{out_name} {bits}\n")

            total_seen += bs

    if decoder is not None and total_seen > 0:
        print(f"Bitwise accuracy on fingerprinted images: {total_correct / total_seen:.4f}")

# -----------------------------
# Entry
# -----------------------------
def main():
    embed_fingerprints()

if __name__ == "__main__":
    main()
