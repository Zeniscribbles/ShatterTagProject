"""
train_cifar10.py — StegaStamp training

Typical Use:
------------
!python /content/drive/MyDrive/ShatterTagProject/string2img/train_cifar10.py \
  --use_cifar10 \
  --cifar10_root "/content/drive/MyDrive/ShatterTagProject/_datasets/cifar10_torch" \
  --output_dir "/content/drive/MyDrive/ShatterTagProject/output/cifar10_10k" \
  --subset_size 10000 \
  --subset_seed 1337 \
  --image_resolution 32 \
  --bit_length 64 \
  --batch_size 64 \
  --num_epochs 40 \
  --lr 1e-4 \
  --cuda cuda \
  --beta 0.0 \
  --l2_loss_weight 1.0 \
  --l2_loss_ramp 1000 \
  --BCE_loss_weight 1.0

Author: Amanda + Chansen
Citation: https://github.com/yunqing-me/WatermarkDM.git
"""
import argparse
import os
from os.path import join
from datetime import datetime
import time  # For time.time()
import random

import glob
import PIL
from tqdm import tqdm


import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from perturbations.cifar_perturbations import cifar10_perturbation_bank as perturbation_bank


# from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter # For google colab

import models


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
#parser.add_argument(
#    "--data_dir", type=str, required=True, help="Directory with image dataset."
#)
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
parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")

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

parser.add_argument("--data_dir", type=str, default=None,
                    help="Flat image folder (only used when --use_cifar10 is NOT set).")


# --- Cifar10 Subset argparse additions ---
parser.add_argument("--subset_size", type=int, default=0,
                    help="If >0, sample this many training images from CIFAR-10.")
parser.add_argument("--subset_seed", type=int, default=1337,
                    help="RNG seed for subset sampling.")


# ----------- For Fragility --------------------
parser.add_argument("--beta", type=float, default=1.0,
                    help="Weight for anti-robustness (fragility) term")

parser.add_argument("--aug_strength", type=float, default=1.0,
                    help="Scales all perturbation magnitudes in bank 𝒜")

parser.add_argument(
    "--frag_bitacc_threshold",
    type=float,
    default=0.9,
    help="Clean Bitwise accuracy threshold to achieve before applying the fragility term.",
)


args = parser.parse_args()
# -----------------------------
# Paths (create BEFORE SummaryWriter)
# -----------------------------
LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "saved_images")  # <- no './'

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(SAVED_IMAGES, exist_ok=True)

writer = SummaryWriter(LOGS_PATH)

'''
# -----------------------------
# Perturbation bank
# -----------------------------
"""
    x: (N, C, H, W), values in [0,1]
    strength in [0, 1]-ish
    Returns tampered image, no clipping of grad assumed here (caller can no_grad).
"""
def perturbation_bank(x, strength: float = 1.0):
    # build a few small, CIFAR-friendly ops and chain 1–3 of them.
    def jpeg_like(img):
        # fake JPEG via down-up-sample
        N, C, H, W = img.shape
        scale = 1.0 - 0.02 * strength  # shrink by up to ~2%
        scale = max(0.5, scale)
        newH = max(2, int(H * scale))
        newW = max(2, int(W * scale))
        img2 = F.interpolate(img, size=(newH, newW), mode="bilinear", align_corners=False)
        img3 = F.interpolate(img2, size=(H, W), mode="bilinear", align_corners=False)
        return img3.clamp(0, 1)

    def gauss_noise(img):
        sigma = (0.5 / 255.0) * strength
        return (img + sigma * torch.randn_like(img)).clamp(0, 1)

    def blur(img):
        k = 3 if strength < 0.6 else 5
        return TF.gaussian_blur(img, kernel_size=k, sigma=0.5 + 0.5 * strength)

    def brightness(img):
        delta = 0.01 * strength
        factor = 1.0 + random.choice([-delta, delta])
        return TF.adjust_brightness(img, factor)

    def tiny_crop(img):
        N, C, H, W = img.shape
        pad = max(1, int(0.02 * H * strength))  # tiny crop
        top = random.randint(0, pad)
        left = random.randint(0, pad)
        newH = H - random.randint(0, pad)
        newW = W - random.randint(0, pad)
        newH = max(2, newH)
        newW = max(2, newW)
        cropped = img[..., top:newH, left:newW]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)

    def subpixel_shift(img):
        N, C, H, W = img.shape
        dx = 0.5 * strength * random.choice([-1, 1])
        dy = 0.5 * strength * random.choice([-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing="ij",
        )
        grid = torch.stack((grid_x + dx / W, grid_y + dy / H), dim=-1)
        grid = grid.unsqueeze(0).expand(N, H, W, 2)
        return F.grid_sample(img, grid, align_corners=True)

    ops = [jpeg_like, gauss_noise, blur, brightness, tiny_crop, subpixel_shift]

    k = random.randint(1, 3)
    out = x
    for f in random.sample(ops, k=k):
        out = f(out)
    return out
'''

# -----------------------------
# Utils
# -----------------------------
def generate_random_fingerprints(bit_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, bit_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)

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


# -----------------------------
# NEW: flat-folder dataset for Colab export (fallback)
# CHANGE: replace the original dataset/sharded 
# loader with a simple flat-folder loader.
# -----------------------------
class CustomImageFolder(Dataset):
    """
    (Optional, unused) original-style loader kept for reference. Not called.
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


# -----------------------------
# Main
# -----------------------------
def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"

    device = parse_device(args.cuda)
    print(f"Using device: {device}")

    IMAGE_CHANNELS = 3  # <-- used by models: CIFAR-10: shape [3, 32, 32] → 3 color channels → IMAGE_CHANNELS = 3

    # Transforms
    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires 128x128, got {args.image_resolution}."
        transform_train = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(args.image_resolution),
            transforms.CenterCrop(args.image_resolution),
            transforms.ToTensor(),
        ])

    # Dataset (CIFAR-10 recommended)
    if args.use_cifar10:
        print(f"Loading CIFAR-10 (root={args.cifar10_root}) ...")
        # download=True won't re-download if already cached
        train_full = CIFAR10(root=args.cifar10_root, train=True, download=True,
                            transform=(transforms.ToTensor() if args.image_resolution == 32 else transform_train))
        if args.subset_size and args.subset_size > 0:
            print(f"Sampling {args.subset_size} images from CIFAR-10 (seed={args.subset_seed})")
            g = torch.Generator().manual_seed(args.subset_seed)
            idx = torch.randperm(len(train_full), generator=g)[:args.subset_size]
            train_set = Subset(train_full, idx.tolist())
            print(f"[Data] Using CIFAR-10 subset: {len(train_set)} / {len(train_full)}")
        else:
            train_set = train_full
            print(f"[Data] Using full CIFAR-10 train set: {len(train_set)}")
    else:
        if not args.data_dir:
            raise ValueError("--data_dir must be set when --use_cifar10 is not provided.")
        print(f"Loading image folder {args.data_dir} ...")
        train_set = CustomImageFolder(args.data_dir, transform=transform_train)
        print(f"[Data] Using flat-folder dataset: {len(train_set)} images")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    # Models
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

    #-------------IMPORTANT: PyTorch Shit------------
    #In PyTorch, every module (nn.Module) has two internal modes:
    # train mode → model.train()
    # eval mode → model.eval()
    #They control layers that behave differently depending on mode
    #Turn on when running real training and out of debugging modes.
    #encoder.train()
    #decoder.train()

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

    # Helper to freeze/unfreeze decoder params
    def set_decoder_trainable(dec, trainable: bool):
        for p in dec.parameters():
            p.requires_grad = trainable

    # Initially, decoder is trainable during robust phase
    set_decoder_trainable(decoder, True)

    decoder_encoder_optim = Adam(params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr)
    torch.backends.cudnn.benchmark = True

    global_step = 0
    steps_since_l2_loss_activated = -1
    log_every = 1 # 1 for debugging, O.W. 100
    bce_logits = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    fragility_active = False  # will flip to True once bitwise_acc crosses threshold
    decoder_frozen_for_frag = False  # Has decoder been frozen already?
    
    
    for i_epoch in range(args.num_epochs):
        print(f"\n[Train] Starting epoch {i_epoch + 1}/{args.num_epochs}")
        
        epoch_loss_sum = 0.0
        epoch_bit_acc_sum = 0.0
        epoch_batches = 0

        # minimum warmup before fragility can activate
        MIN_WARMUP_STEPS = 1000  # You can tune this (500–2000 for CIFAR10)

        # Starting fragility training
        for images, _ in tqdm(train_loader):
            
            global_step += 1

            bsz = images.size(0)
            fingerprints = generate_random_fingerprints(
                args.bit_length,
                bsz,
                (args.image_resolution, args.image_resolution),
            )

            # l2 schedule stays exactly as before
            l2_loss_weight = min(
                max(0, args.l2_loss_weight * (steps_since_l2_loss_activated - args.l2_loss_await) / args.l2_loss_ramp),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)

            # ----- encoder: clean path -----
            fingerprinted_images = encoder(fingerprints, clean_images)            
            # residual = fingerprinted_images - clean_images logging/visualization

            # clean decode
            decoder_output_clean = decoder(fingerprinted_images)

            # clean losses
            l2_loss = mse_loss(fingerprinted_images, clean_images)
            BCE_loss_clean = bce_logits(
                decoder_output_clean.view(-1),
                fingerprints.view(-1),
            )

            # ----- tamper path: attack only the encoded image -----
            tampered_images = perturbation_bank(
                fingerprinted_images,
                strength=args.aug_strength,
            )
            decoder_output_tam = decoder(tampered_images)

            BCE_loss_tam = bce_logits(
                decoder_output_tam.view(-1),
                fingerprints.view(-1),   # target = TRUE watermark bits
            )

             # ----- clean bitwise accuracy (used for schedules + fragility switch) -----
            fingerprints_predicted = (decoder_output_clean > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(torch.abs(fingerprints - fingerprints_predicted))

            # ----- fragility activation & decoder freezing -----
            if (not fragility_active) \
                and (global_step > MIN_WARMUP_STEPS) \
                and (bitwise_accuracy.item() > args.frag_bitacc_threshold):
                
                fragility_active = True
                print(f"[Train] Activating fragility at step {global_step}, "
                      f"Clean bitwise_acc={bitwise_accuracy.item():.4f}")

                # Freeze decoder immediately once fragility starts
                set_decoder_trainable(decoder, False)
                decoder_frozen_for_frag = True

            # effective beta: 0 before activation, args.beta after
            beta_eff = args.beta if fragility_active else 0.0 

            # ----- total loss: clean + fidelity - beta * tamper -----
            loss = (
                l2_loss_weight * l2_loss
                + BCE_loss_weight * BCE_loss_clean
                - beta_eff * BCE_loss_tam
            )

            encoder.zero_grad(set_to_none=True)
            decoder.zero_grad(set_to_none=True)
            loss.backward()
            decoder_encoder_optim.step()

            # optional: tamper metrics (if you want to log them later)
            tamper_pred = (decoder_output_tam > 0).float()
            tamper_bit_acc_vs_true = 1.0 - torch.mean(torch.abs(fingerprints - tamper_pred))
           
            # -----------------------------------------
            # TensorBoard logging for fragility stuff
            # -----------------------------------------
            writer.add_scalar("train/clean_bit_acc", bitwise_accuracy.item(), global_step)
            writer.add_scalar("train/tamper_bit_acc", tamper_bit_acc_vs_true.item(), global_step)
            writer.add_scalar("train/fragility_active", float(fragility_active), global_step)
            writer.add_scalar("train/decoder_frozen", float(decoder_frozen_for_frag), global_step)
            writer.add_scalar("train/BCE_loss_clean", BCE_loss_clean.item(), global_step)
            writer.add_scalar("train/BCE_loss_tam", BCE_loss_tam.item(), global_step)
            
            # Legacy Code: Stay as-is
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1

            # ---- epoch stats accumulation ----
            epoch_loss_sum += loss.item()
            epoch_bit_acc_sum += bitwise_accuracy.item()
            epoch_batches += 1

            # ---- occasional console logging ----
            if global_step % log_every == 0:
                print(
                    f"[Train] step {global_step} | "
                    f"loss={loss.item():.4f} | "
                    f"clean_BCE={BCE_loss_clean.item():.4f} | "
                    f"tam_BCE={BCE_loss_tam.item():.4f} | "
                    f"bitwise_acc={bitwise_accuracy.item():.4f} | "
                    f"tamper_vs_true={tamper_bit_acc_vs_true.item():.4f} | "
                    f"l2_w={l2_loss_weight:.3f} | "
                    f"frag_active={fragility_active} | "
                    f"decoder_frozen={decoder_frozen_for_frag}"
                )
                
            '''
            # -------------------------------------------------
            # Legacy image/stat logging & checkpoints 
            #   - kept here but COMMENTED OUT
            #   - uncomment if you want full TensorBoard visuals
            # -------------------------------------------------
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy.item(), global_step)
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("BCE_loss", BCE_loss_clean.item(), global_step)

                writer.add_scalars("clean_statistics", {"min": clean_images.min(), "max": clean_images.max()}, global_step)
                writer.add_scalars("with_fingerprint_statistics", {"min": fingerprinted_images.min(), "max": fingerprinted_images.max()}, global_step)
                writer.add_scalars("residual_statistics", {"min": residual.min(), "max": residual.max(), "mean_abs": residual.abs().mean()}, global_step)

                writer.add_image("clean_image", make_grid(clean_images, normalize=True), global_step)
                writer.add_image("residual", make_grid(residual, normalize=True, scale_each=True), global_step)
                writer.add_image("image_with_fingerprint", make_grid(fingerprinted_images, normalize=True), global_step)

                save_image(fingerprinted_images, join(SAVED_IMAGES, f"{global_step}.png"), normalize=True)

                writer.add_scalar("loss_weights/l2_loss_weight", l2_loss_weight, global_step)
                writer.add_scalar("loss_weights/BCE_loss_weight", BCE_loss_weight, global_step)

            if global_step % 5000 == 0:
                torch.save(decoder_encoder_optim.state_dict(), join(CHECKPOINTS_PATH, f"{EXP_NAME}_optim.pth"))
                torch.save(encoder.state_dict(), join(CHECKPOINTS_PATH, f"{EXP_NAME}_encoder.pth"))
                torch.save(decoder.state_dict(), join(CHECKPOINTS_PATH, f"{EXP_NAME}_decoder.pth"))
                with open(join(CHECKPOINTS_PATH, f"{EXP_NAME}_variables.txt"), "w") as f:
                    f.write(str(global_step))
            '''

            # ---- end-of-epoch summary ----
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_batches)
            avg_epoch_bit_acc = epoch_bit_acc_sum / max(1, epoch_batches)
            print(
                f"[Train] Finished epoch {i_epoch + 1}/{args.num_epochs} | "
                f"avg_loss={avg_epoch_loss:.4f} | "
                f"avg_bitwise_acc={avg_epoch_bit_acc:.4f}")

    # ----- final save so _last always updates -----
    torch.save(decoder_encoder_optim.state_dict(), join(CHECKPOINTS_PATH, f"{EXP_NAME}_optim_last.pth"))
    torch.save(encoder.state_dict(), join(CHECKPOINTS_PATH, f"{EXP_NAME}_encoder_last.pth"))
    torch.save(decoder.state_dict(), join(CHECKPOINTS_PATH, f"{EXP_NAME}_decoder_last.pth"))
    with open(join(CHECKPOINTS_PATH, f"{EXP_NAME}_variables.txt"), "w") as f:
        f.write(str(global_step))

    writer.close()

if __name__ == "__main__":
    main()