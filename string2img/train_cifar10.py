"""
train_cifar10.py — StegaStamp training

Overview
--------
Trains a StegaStamp-style encoder/decoder to embed and recover binary fingerprints from RGB images.
This version stays faithful to the original training flow (same losses, scheduling, logging,
and checkpointing). The only change is how the dataset is loaded so it works cleanly with
a Google Colab–exported CIFAR-10 folder.

What changed for Colab CIFAR-10 import
--------------------------------------
- Flat-folder loader: instead of class subdirs or shard loops, the script reads a single
  flat directory of images produced by your Colab export:
    /.../data/cifar10/00000.png … 49999.png
- No recursion, no labels required; we just glob .png/.jpg/.jpeg in that folder.
- Everything else (model defs, optimizer, losses, bitwise accuracy metric, TensorBoard
  logging, and checkpoint cadence) is unchanged.

Typical use
-----------
python train_cifar10.py \
  --data_dir "/content/drive/MyDrive/ShatterTagProject/data/cifar10" \
  --output_dir "/content/drive/MyDrive/ShatterTagProject/output/cifar10_run1" \
  --image_resolution 32 \
  --bit_length 64 \
  --batch_size 64 \
  --num_epochs 10 \
  --lr 1e-4 \
  --cuda cuda \
  --l2_loss_weight 1.0 \
  --l2_loss_await 0 \
  --l2_loss_ramp 1000 \
  --BCE_loss_weight 1.0 

Inputs
------
--data_dir: path to the flat CIFAR-10 export folder (images only, no subfolders)
--output_dir: base directory for logs / checkpoints / saved images
--image_resolution: square size to which images are resized/cropped
--bit_length: number of bits in the embedded fingerprint
(plus the usual optimizer, loss-weight, and logging args)

Outputs
-------
- TensorBoard logs: <output_dir>/logs
- Checkpoints:      <output_dir>/checkpoints
- Sample images:    <output_dir>/saved_images

Assumptions
-----------
- The Colab export produced RGB images with lowercase extensions (.png/.jpg/.jpeg).
- The folder is flat (no class subdirectories).
- image_resolution matches the training/eval pipeline (e.g., 32 for CIFAR-10).

Author: Amanda + Chansen
Citation: https://github.com/yunqing-me/WatermarkDM.git
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory with image dataset."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save results to."
)
parser.add_argument(
    "--bit_length",
    type=int,
    default=64,
    # required=True, // Must be supplied in CLI. Interferes with default.
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=32,
    # required=True, // Must be supplied in CLI. Interferes with default.
    help="Height and width of square images.",
)
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default="cuda")

parser.add_argument(
    "--l2_loss_await",
    help="Train without L2 loss for the first x iterations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--l2_loss_weight",
    type=float,
    default=10,
    help="L2 loss weight for image fidelity.",
)
parser.add_argument(
    "--l2_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase L2 loss weight over x iterations.",
)

parser.add_argument(
    "--BCE_loss_weight",
    type=float,
    default=1,
    help="BCE loss weight for fingerprint reconstruction.",
)

# --- For CIFAR10 dataset ---
parser.add_argument("--use_cifar10", action="store_true",
                    help="Use torchvision.datasets.CIFAR10 instead of a flat image folder.")
parser.add_argument("--cifar10_root", type=str, default="./_data",
                    help="Where to download/store CIFAR-10 when --use_cifar10 is set.")


args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
# from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
#from tensorboardX import SummaryWriter // Or import in google colab
from torch.utils.tensorboard import SummaryWriter # For google colab


from torch.optim import Adam

import models


LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)


def generate_random_fingerprints(bit_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, bit_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)

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


# -------------------- NEW: flat-folder dataset for Colab export --------------------
# CHANGE: replace the original dataset/sharded loader with a simple flat-folder loader.
class CustomImageFolder(Dataset):
    """
    Minimal change: read a flat folder of images produced by the Colab CIFAR-10 export.
    (e.g., /.../cifar10/00000.png ... 49999.png; no class subdirs)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # CHANGE: load all images directly from a single directory (no recursion / no shards)
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB") # Converts anything to 3-channel 8-bit RGB:
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3
    SECRET_SIZE = args.bit_length

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_RESOLUTION),
                transforms.CenterCrop(IMAGE_RESOLUTION),
                transforms.ToTensor(),
            ]
        )
    
    if args.use_cifar10:
        print(f"Loading CIFAR-10 (root={args.cifar10_root}) ...")
        dataset = CIFAR10(root=args.cifar10_root, train=True, download=True, transform=transform)
    else:
        print(f"Loading image folder {args.data_dir} ...")
        dataset = CustomImageFolder(args.data_dir, transform=transform)

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    # CHANGE: use the flat-folder loader instead of the original (which expected shards/class dirs)
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"

    device = parse_device(args.cuda)
    print(f"Using device: {device}")

    load_data()
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )

    global_step = 0
    steps_since_l2_loss_activated = -1

    for i_epoch in range(args.num_epochs):
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        for images, _ in tqdm(dataloader):
            global_step += 1

            batch_size = min(args.batch_size, images.size(0))
            fingerprints = generate_random_fingerprints(
                args.bit_length, batch_size, (args.image_resolution, args.image_resolution)
            )

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
            fingerprints = fingerprints.to(device)

            fingerprinted_images = encoder(fingerprints, clean_images)
            residual = fingerprinted_images - clean_images

            decoder_output = decoder(fingerprinted_images)

            criterion = nn.MSELoss()
            l2_loss = criterion(fingerprinted_images, clean_images)

            criterion = nn.BCEWithLogitsLoss()
            BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))

            loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1

            # Logging
            if global_step in plot_points:
                # writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step)
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy.item(), global_step) # For TensorBoard
                print("Bitwise accuracy {}".format(bitwise_accuracy))

                # writer.add_scalar("loss", loss, global_step) # For TensorBoard
                writer.add_scalar("loss", loss.item(), global_step)

                # writer.add_scalar("BCE_loss", BCE_loss, global_step) # For TensorBoard
                writer.add_scalar("BCE_loss", BCE_loss.item(), global_step)
                writer.add_scalars(
                    "clean_statistics",
                    {"min": clean_images.min(), "max": clean_images.max()},
                    global_step,
                ),
                writer.add_scalars(
                    "with_fingerprint_statistics",
                    {
                        "min": fingerprinted_images.min(),
                        "max": fingerprinted_images.max(),
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "residual_statistics",
                    {
                        "min": residual.min(),
                        "max": residual.max(),
                        "mean_abs": residual.abs().mean(),
                    },
                    global_step,
                ),
                print(
                    "residual_statistics: {}".format(
                        {
                            "min": residual.min(),
                            "max": residual.max(),
                            "mean_abs": residual.abs().mean(),
                        }
                    )
                )
                writer.add_image(
                    "clean_image", make_grid(clean_images, normalize=True), global_step
                )
                writer.add_image(
                    "residual",
                    make_grid(residual, normalize=True, scale_each=True),
                    global_step,
                )
                writer.add_image(
                    "image_with_fingerprint",
                    make_grid(fingerprinted_images, normalize=True),
                    global_step,
                )
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )

                writer.add_scalar(
                    "loss_weights/l2_loss_weight", l2_loss_weight, global_step
                )
                writer.add_scalar(
                    "loss_weights/BCE_loss_weight",
                    BCE_loss_weight,
                    global_step,
                )

            # checkpointing
            if global_step % 5000 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                
                f = open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w")
                f.write(str(global_step))
                f.close()

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()