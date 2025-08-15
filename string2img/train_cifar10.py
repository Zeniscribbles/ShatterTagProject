
"""
train_cifar10.py (patched)

- Robust device parsing: accepts int, "cpu", "cuda", "cuda:N" and falls back gracefully.
- Determinism switch via --seed (freezes CuDNN where appropriate).
- Faster DataLoader defaults: num_workers/pin_memory chosen based on device; CLI override.
- Automatic Mixed Precision (AMP) on CUDA for speed.
- Clear run header and path validations.
- Keeps your losses/scheduling, logging, and checkpointing semantics.

Author: Amanda + Chansen (+ patch pass)
"""

import argparse
import glob
import os
from os.path import join
from datetime import datetime

import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.optim import Adam
from tqdm import tqdm
import random
import numpy as np

import models


# -------------------- Utilities --------------------

def parse_device(arg_cuda):
    """
    Accepts: int (0,1,...), 'cpu', 'cuda', or 'cuda:N'.
    Falls back to best available device.
    """
    if isinstance(arg_cuda, int):
        return torch.device(f"cuda:{arg_cuda}") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(arg_cuda, str):
        s = arg_cuda.strip().lower()
        if s == "cpu":
            return torch.device("cpu")
        if s == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if s.startswith("cuda:"):
            try:
                idx = int(s.split(":")[1])
                return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
            except Exception:
                pass
        # plain int as string?
        if s.isdigit():
            idx = int(s)
            return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    # default
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------- Dataset --------------------

class CustomImageFolder(Dataset):
    """
    Dataset that loads all .png, .jpg, and .jpeg images from a directory recursively.
    Applies specified torchvision transforms.
    """
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.filenames = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.[pj][pn]g"), recursive=True)
        )
        if not self.filenames:
            raise RuntimeError(f"No image files found in {data_dir}")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def build_transforms(image_resolution: int):
    return transforms.Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
    ])


def build_dataloader(data_dir: str, image_resolution: int, batch_size: int, device: torch.device,
                     num_workers: int = None):
    transform = build_transforms(image_resolution)
    dataset = CustomImageFolder(data_dir, transform=transform)

    use_cuda = device.type == "cuda"
    if num_workers is None:
        # Good defaults for Colab/local
        num_workers = 2 if use_cuda else 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=use_cuda,
    )

    # Preview image shape
    sample_image, _ = dataset[0]
    print("Sample image loaded size:", tuple(sample_image.shape))

    return dataset, dataloader


# -------------------- Main training --------------------

def main(args):
    # Validate paths
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"--data_dir not found: {args.data_dir}")

    # Device + determinism
    device = parse_device(args.cuda)
    use_cuda = device.type == "cuda"
    if args.seed is not None:
        make_deterministic(args.seed)

    # Output structure
    LOGS_PATH = os.path.join(args.output_dir, "logs")
    CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
    SAVED_IMAGES = os.path.join(args.output_dir, "saved_images")
    ensure_dir(LOGS_PATH); ensure_dir(CHECKPOINTS_PATH); ensure_dir(SAVED_IMAGES)

    writer = SummaryWriter(LOGS_PATH)
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"

    # Data
    dataset, dataloader = build_dataloader(
        args.data_dir, args.image_resolution, args.batch_size, device, args.num_workers
    )
    IMAGE_CHANNELS = 3

    # Clear run header
    print("=" * 70)
    print("ShatterTag CIFAR10 training")
    print(f"device={device} (cuda_available={torch.cuda.is_available()})")
    print(f"seed={args.seed}")
    print(f"data_dir={args.data_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"image_resolution={args.image_resolution}  batch_size={args.batch_size}  epochs={args.num_epochs}")
    print(f"bit_length={args.bit_length}  lr={args.lr}")
    print(f"losses: BCE={args.BCE_loss_weight}  L2={args.l2_loss_weight} "
          f"(ramp={args.l2_loss_ramp}, await={args.l2_loss_await})")
    print(f"num_workers={args.num_workers if args.num_workers is not None else ('auto(' + str(2 if use_cuda else 0) + ')')}  pin_memory={use_cuda}")
    print("=" * 70)

    # Models
    encoder = models.StegaStampEncoder(
        args.image_resolution, IMAGE_CHANNELS, args.bit_length, return_residual=False
    ).to(device)
    decoder = models.StegaStampDecoder(
        args.image_resolution, IMAGE_CHANNELS, args.bit_length
    ).to(device)
    encoder.train(); decoder.train()

    # Optimizer
    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )

    global_step = 0
    steps_since_l2_loss_activated = -1

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # Precompute logging steps
    # If log_interval is in "steps", just use modulo. Keep your plot_points approach but safer.
    next_log = args.log_interval

    for i_epoch in range(args.num_epochs):
        print(f"\n🌀 Epoch [{i_epoch + 1}/{args.num_epochs}]")

        for images, _ in tqdm(dataloader, desc=f"Training (Epoch {i_epoch + 1})", leave=False):
            global_step += 1
            batch_size = images.size(0)

            fingerprints = torch.randint(
                0, 2, (batch_size, args.bit_length), dtype=torch.float, device=device
            )

            # Weight schedule for L2 loss
            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / max(1, args.l2_loss_ramp),
                ),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                fingerprinted_images = encoder(fingerprints, clean_images)
                residual = fingerprinted_images - clean_images

                decoder_output = decoder(fingerprinted_images)

                # Compute loss
                l2_loss = nn.MSELoss()(fingerprinted_images, clean_images)
                BCE_loss = nn.BCEWithLogitsLoss()(decoder_output.view(-1), fingerprints.view(-1))
                loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            decoder_encoder_optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(decoder_encoder_optim)
            scaler.update()

            # Accuracy
            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(torch.abs(fingerprints - fingerprints_predicted))

            if steps_since_l2_loss_activated == -1 and bitwise_accuracy.item() > 0.9:
                steps_since_l2_loss_activated = 0
            elif steps_since_l2_loss_activated != -1:
                steps_since_l2_loss_activated += 1

            # Logging
            if global_step >= next_log:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy.item(), global_step)
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/bce", BCE_loss.item(), global_step)
                writer.add_scalar("loss/l2", l2_loss.item(), global_step)
                writer.add_scalars("clean_statistics", {
                    "min": clean_images.min().item(),
                    "max": clean_images.max().item()
                }, global_step)
                writer.add_scalars("with_fingerprint_statistics", {
                    "min": fingerprinted_images.min().item(),
                    "max": fingerprinted_images.max().item()
                }, global_step)
                writer.add_scalars("residual_statistics", {
                    "min": residual.min().item(),
                    "max": residual.max().item(),
                    "mean_abs": residual.abs().mean().item()
                }, global_step)

                writer.add_image("clean_image", make_grid(clean_images, normalize=True), global_step)
                writer.add_image("residual", make_grid(residual, normalize=True, scale_each=True), global_step)
                writer.add_image("image_with_fingerprint", make_grid(fingerprinted_images, normalize=True), global_step)

                # Save each fingerprinted image individually instead of as a grid
                for i in range(fingerprinted_images.size(0)):
                    save_path = os.path.join(SAVED_IMAGES, f"{global_step}_{i}.png")
                    save_image(fingerprinted_images[i], save_path, normalize=True)


                writer.add_scalar("loss_weights/l2_loss_weight", l2_loss_weight, global_step)
                writer.add_scalar("loss_weights/BCE_loss_weight", BCE_loss_weight, global_step)

                next_log += args.log_interval

            # Periodic checkpoint
            if global_step % args.ckpt_interval == 0:
                torch.save(decoder_encoder_optim.state_dict(), join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"))
                torch.save(encoder.state_dict(), join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"))
                torch.save(decoder.state_dict(), join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"))
                with open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w") as f:
                    f.write(str(global_step))

    print(f"\nTraining complete. Final step: {global_step}")
    print(f"Checkpoints saved to: {CHECKPOINTS_PATH}")
    print(f"Fingerprinted images saved to: {SAVED_IMAGES}")
    print(f"Logs available for TensorBoard in: {LOGS_PATH}")

    writer.close()


# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StegaStamp model for image watermarking.")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to image dataset.")
    parser.add_argument("--image_resolution", type=int, default=32, help="Input image resolution (e.g., 32 for CIFAR).")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for logs/checkpoints/images.")
    parser.add_argument("--bit_length", type=int, default=64, help="Length of the binary fingerprint vector.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default bumped to 1e-3 for speed).")
    parser.add_argument("--cuda", type=str, default="cuda", help="Device to use: int index, 'cpu', 'cuda', or 'cuda:N'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism (set None to disable).")

    parser.add_argument("--l2_loss_weight", type=float, default=1.0, help="Max weight for image reconstruction loss (MSE).")
    parser.add_argument("--l2_loss_await", type=int, default=0, help="Step at which to begin applying L2 loss.")
    parser.add_argument("--l2_loss_ramp", type=int, default=1000, help="Steps to ramp up L2 loss to full strength.")
    parser.add_argument("--BCE_loss_weight", type=float, default=1.0, help="Weight for binary cross-entropy loss.")

    parser.add_argument("--log_interval", type=int, default=1000, help="Steps between logging diagnostics/images.")
    parser.add_argument("--ckpt_interval", type=int, default=5000, help="Steps between checkpoints.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (None = auto).")

    args = parser.parse_args()
    main(args)
