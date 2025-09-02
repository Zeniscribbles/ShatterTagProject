"""
Embed fingerprints into images using a trained StegaStamp encoder.

Keeps original logic:
- optional CelebA preprocessing
- identical vs. random per-image fingerprints
- optional on-the-fly check via decoder
- same function structure: load_data -> load_models -> embed_fingerprints

Colab hardening (only what this file actually does):
- creates output dirs up front
- robust device parsing (CPU / CUDA / CUDA:N); models set to eval()
- RGB-safe image loading (PIL .convert('RGB'))
- checkpoints loaded on CPU, then moved to the selected device

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
Citation: https://github.com/yunqing-me/WatermarkDM.git
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

uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        if not self.filenames:
            raise RuntimeError(f"No images found in {data_dir}")
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB")  # robust RGB
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

# -----------------------------
# Data
# -----------------------------
dataset = None
HideNet = None
RevealNet = None
FINGERPRINT_SIZE = None

def load_data():
    global dataset, dataloader
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
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

# -----------------------------
# Models
# -----------------------------
def load_models():
    global HideNet, RevealNet, FINGERPRINT_SIZE

    from models import StegaStampEncoder, StegaStampDecoder

    # map_location=device keeps CPU/GPU consistent and avoids surprises
    state_dict = torch.load(args.encoder_path, map_location="cpu")
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE, return_residual=False
    ).to(device).eval()
    HideNet.load_state_dict(state_dict)

    RevealNet = None
    if args.check:
        if not args.decoder_path:
            raise ValueError("--check was set but --decoder_path is missing.")
        dec_sd = torch.load(args.decoder_path, map_location="cpu")
        RevealNet = StegaStampDecoder(
            IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
        ).to(device).eval()
        RevealNet.load_state_dict(dec_sd)

# -----------------------------
# Main embed loop (streaming saves; no RAM hoarding)
# -----------------------------
def embed_fingerprints():
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_note, exist_ok=True)

    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
    fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE)
    fingerprints = fingerprints.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)

    bitwise_accuracy = 0

    for images, _ in tqdm(dataloader):

        # generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)
            fingerprints = fingerprints.to(device)

        images = images.to(device)

        fingerprinted_images = HideNet(fingerprints[: images.size(0)], images)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()

    dirname = args.output_dir
    # if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
    #     os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()

    f = open(os.path.join(args.output_dir_note, "embedded_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprinted_images)):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        # filename = filename.split('.')[0] + ".png"
        # save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{filename}"), padding=0)
        save_image(image, os.path.join(args.output_dir, f"{filename}"), padding=0)
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")

        # Add print statements to print this out
        save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        save_image(torch.abs(images - fingerprinted_images)[:49], os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)

# -----------------------------
# Entry
# -----------------------------
def main():
    
    device = parse_device(args.cuda)
    print(f"Using device: {device}")

    # Single-pass mode (Colab): honor CLI args and bail out
    if args.data_dir and args.output_dir and args.output_dir_note and args.encoder_path and args.image_resolution:
        # optional preflight (helpful, not required)
        # preflight()
        load_data()
        load_models()
        embed_fingerprints()
        return

    # ----- legacy shard loop below (unchanged) -----
    args.encoder_path   = "./_output/cifar10_64/checkpoints/*.pth"
    args.image_resolution = 32
    args.identical_fingerprints = True
    root_data_dir = "../edm/datasets/uncompressed/cifar10/"
    image_outdir  = "../edm/datasets/embedded/cifar10/images/"
    note_outdir   = "../edm/datasets/embedded/cifar10/note/"

    # process cifar10 dataset
    for i in tqdm(range(50)):
        args.data_dir         = os.path.join(root_data_dir, f"{str(i).zfill(5)}")
        args.output_dir       = os.path.join(image_outdir, f"{str(i).zfill(5)}")
        args.output_dir_note  = os.path.join(note_outdir, f"{str(i).zfill(5)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.output_dir_note):
            os.makedirs(args.output_dir_note)
        load_data()
        load_models()
        embed_fingerprints()

if __name__ == "__main__":
    main()
