import argparse
import os
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--encoder_path", type=str, help="Path to trained StegaStamp encoder.")
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument("--output_dir", type=str, help="Path to save watermarked images to.")
parser.add_argument("--output_dir_note", type=str, help="Directory to save fingerprint metadata (embedded_fingerprints.txt).")
parser.add_argument("--image_resolution", type=int, help="Height and width of square images.")
parser.add_argument("--identical_fingerprints", action="store_true", help="Use identical fingerprints instead of random ones.")
parser.add_argument("--check", action="store_true", help="Validate fingerprint detection accuracy.")
parser.add_argument("--decoder_path", type=str, help="Trained StegaStamp decoder to verify fingerprint detection accuracy.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0)

args = parser.parse_args()
BATCH_SIZE = args.batch_size

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# Device setup
if int(args.cuda) == -1 or not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.cuda}")
print(f"Using device: {device}")

def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z

uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = sorted(
            glob.glob(os.path.join(data_dir, "*.png"))
            + glob.glob(os.path.join(data_dir, "*.jpeg"))
            + glob.glob(os.path.join(data_dir, "*.jpg"))
        )
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
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

def load_models():
    global HideNet, RevealNet, FINGERPRINT_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path, map_location=device)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    ).to(device)

    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    ).to(device)

    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path, map_location=device))
    HideNet.load_state_dict(torch.load(args.encoder_path, map_location=device))

def embed_fingerprints():
    """
    Embeds fingerprints into images using the trained StegaStamp encoder.
    Optionally evaluates fingerprint recovery accuracy using the decoder.
    Saves fingerprinted images and their associated fingerprints.
    """
    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_note, exist_ok=True)
    fingerprint_log = open(os.path.join(args.output_dir_note, "embedded_fingerprints.txt"), "w")

    bitwise_accuracy = 0
    total_images = 0

    for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)

        if args.identical_fingerprints:
            if batch_idx == 0:
                fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
                fingerprints = fingerprints.expand(images.size(0), -1)
        else:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, images.size(0))

        fingerprints = fingerprints.to(device)

        fingerprinted_images = HideNet(fingerprints, images)

        for i in range(images.size(0)):
            global_idx = batch_idx * BATCH_SIZE + i
            if global_idx >= len(dataset.filenames):
                continue  # avoid overflow if dataset length not divisible by batch size

            original_filename = os.path.basename(dataset.filenames[global_idx])
            save_path = os.path.join(args.output_dir, original_filename)

            save_image(fingerprinted_images[i].detach().cpu(), save_path, padding=0)

            fingerprint_str = "".join(map(str, fingerprints[i].long().tolist()))
            fingerprint_log.write(f"{original_filename} {fingerprint_str}\n")

        if args.check:
            detected = RevealNet(fingerprinted_images)
            detected = (detected > 0).long()
            bitwise_accuracy += (
                (detected == fingerprints.long()).float().mean(dim=1).sum().item()
            )
            total_images += images.size(0)

    fingerprint_log.close()

    if args.check and total_images > 0:
        accuracy = bitwise_accuracy / total_images
        print(f"Bitwise accuracy on fingerprinted images: {accuracy:.4f}")


        # Optional debugging output: uncomment to generate visualization grids.
        # These show that the fingerprints are imperceptible to the human eye.

        """
        #save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        #save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        #save_image(
        #    torch.abs(images - fingerprinted_images)[:49],
        #    os.path.join(args.output_dir, "test_samples_residual.png"),
        #    normalize=True,
        #   nrow=7
        #)
        """
