"""
train_cifar10.py

Main training script for the StegaFrag watermarking framework.

This script trains a convolutional encoder-decoder pair to embed and recover binary fingerprints
(aka watermarks) from RGB images. The system is designed to support both robust watermarking 
(for ownership verification) and future extensions toward fragile watermarking 
(for tamper detection and image integrity).

Key Features:
-------------
- Dataset-agnostic training (supports arbitrary image folders)
- Configurable loss weights, logging intervals, and batch sizes via argparse
- Modular encoder/decoder architecture (see models.py)
- TensorBoard logging and periodic checkpointing
- Compatible with Google Colab or local execution

Typical Use Case:
-----------------
Use this script to train an encoder and decoder pair on your dataset.
The encoder embeds binary fingerprints into images, while the decoder learns to recover them.

Future Direction:
-----------------
This framework will be extended to support fragile watermarking where 
minor image tampering breaks fingerprint recoverability — useful for tamper detection.

Usage:
------
python train_cifar10.py --data_dir ./images --output_dir ./output --image_resolution 128 --bit_length 64

Author: Amanda + Chansen
"""

import argparse
import glob
import os
from os.path import join
from datetime import datetime
from time import time

import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.optim import Adam
from tqdm import tqdm

import models


def generate_random_fingerprints(batch_size, bit_length):
    """
    Generate a binary fingerprint vector for each image in a batch.

    Args:
        batch_size (int): Number of samples in the batch.
        bit_length (int): Number of bits per fingerprint.

    Returns:
        Tensor of shape (batch_size, bit_length) with binary values (0 or 1).
    """
    return torch.randint(0, 2, (batch_size, bit_length), dtype=torch.float)


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


def load_data(args):
    """
    Load image dataset and return a PyTorch DataLoader.

    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        tuple: (dataset, dataloader)
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
    ])

    dataset = CustomImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2, # 2 workers for google colab
        drop_last=True,
    )

    # Preview image shape
    sample_image, _ = dataset[0]
    print("📸 Sample image loaded size:", sample_image.shape)

    return dataset, dataloader


def main(args):
    """
    Main training loop for StegaStamp encoder/decoder.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    LOGS_PATH = os.path.join(args.output_dir, "logs")
    CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
    SAVED_IMAGES = os.path.join(args.output_dir, "saved_images")

    os.makedirs(LOGS_PATH, exist_ok=True)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    os.makedirs(SAVED_IMAGES, exist_ok=True)

    writer = SummaryWriter(LOGS_PATH)

    dt_string = datetime.now().strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    dataset, dataloader = load_data(args)
    IMAGE_CHANNELS = 3

    # Set up plot points for logging frequency
    plot_points = list(range(0, args.num_epochs * 1000, args.log_interval))

    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
        return_residual=False,
    ).to(device)

    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
    ).to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )

    global_step = 0
    steps_since_l2_loss_activated = -1

    for i_epoch in range(args.num_epochs):
        print(f"\n🌀 Epoch [{i_epoch + 1}/{args.num_epochs}]")

        for images, _ in tqdm(dataloader, desc=f"Training Step (Epoch {i_epoch + 1})", leave=False):
            global_step += 1
            batch_size = images.size(0)

            fingerprints = generate_random_fingerprints(
                batch_size,
                args.bit_length
            ).to(device)

            # Debug fingerprint/image shape before encoder
            print("\n📦 DEBUG: Sending batch into encoder")
            print(f"  ↳ fingerprints: {fingerprints.shape}")
            print(f"  ↳ clean_images: {images.shape}")

            # Weight schedule for L2 loss
            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device)
            fingerprinted_images = encoder(fingerprints, clean_images)
            residual = fingerprinted_images - clean_images

            decoder_output = decoder(fingerprinted_images)

            # Compute loss
            l2_loss = nn.MSELoss()(fingerprinted_images, clean_images)
            BCE_loss = nn.BCEWithLogitsLoss()(decoder_output.view(-1), fingerprints.view(-1))
            loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            decoder_encoder_optim.step()

            # Accuracy
            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )

            if steps_since_l2_loss_activated == -1 and bitwise_accuracy.item() > 0.9:
                steps_since_l2_loss_activated = 0
            elif steps_since_l2_loss_activated != -1:
                steps_since_l2_loss_activated += 1

            # Logging
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step)
                writer.add_scalar("loss", loss, global_step)
                writer.add_scalar("BCE_loss", BCE_loss, global_step)
                writer.add_scalars("clean_statistics", {
                    "min": clean_images.min(),
                    "max": clean_images.max()
                }, global_step)
                writer.add_scalars("with_fingerprint_statistics", {
                    "min": fingerprinted_images.min(),
                    "max": fingerprinted_images.max()
                }, global_step)
                writer.add_scalars("residual_statistics", {
                    "min": residual.min(),
                    "max": residual.max(),
                    "mean_abs": residual.abs().mean()
                }, global_step)

                writer.add_image("clean_image", make_grid(clean_images, normalize=True), global_step)
                writer.add_image("residual", make_grid(residual, normalize=True, scale_each=True), global_step)
                writer.add_image("image_with_fingerprint", make_grid(fingerprinted_images, normalize=True), global_step)

                save_image(fingerprinted_images, os.path.join(SAVED_IMAGES, f"{global_step}.png"), normalize=True)

                writer.add_scalar("loss_weights/l2_loss_weight", l2_loss_weight, global_step)
                writer.add_scalar("loss_weights/BCE_loss_weight", BCE_loss_weight, global_step)

            # Checkpoint
            if global_step % 5000 == 0:
                torch.save(decoder_encoder_optim.state_dict(), join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"))
                torch.save(encoder.state_dict(), join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"))
                torch.save(decoder.state_dict(), join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"))
                with open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w") as f:
                    f.write(str(global_step))

    print(f"\n✅ Training complete. Final step: {global_step}")
    print(f"📁 Checkpoints saved to: {CHECKPOINTS_PATH}")
    print(f"🖼️  Fingerprinted images saved to: {SAVED_IMAGES}")
    print(f"📊 Logs available for TensorBoard in: {LOGS_PATH}")

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StegaStamp model for image watermarking.")

    parser.add_argument("--data_dir", type=str, required=True, help="Pat
