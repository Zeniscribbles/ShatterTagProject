import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import relu, sigmoid

def crop_to_match(tensor, target):
    """
    Center-crop a tensor to match the spatial dimensions (height and width) of a target tensor.

    This is critical when using upsample + conv layers, which can sometimes introduce off-by-one 
    mismatches due to padding or rounding. Ensures tensors can be concatenated cleanly.

    Args:
        tensor (Tensor): Tensor to crop (N, C, H, W).
        target (Tensor): Reference tensor to match shape (N, C, H_target, W_target).

    Returns:
        Tensor: Cropped tensor with same height and width as `target`.
    """
    print("crop_to_match() called")
    _, _, h, w = tensor.shape
    _, _, th, tw = target.shape
    print(f"  ↳ cropping from ({h}, {w}) to ({th}, {tw})")
    dh, dw = max(h - th, 0), max(w - tw, 0)
    return tensor[..., dh // 2 : h - (dh - dh // 2), dw // 2 : w - (dw - dw // 2)]

class StegaStampEncoder(nn.Module):
    """
    Encoder network that embeds a binary fingerprint into an image.

    Args:
        resolution (int): Input image resolution (must be power of 2).
        IMAGE_CHANNELS (int): Number of image channels (typically 3 for RGB).
        fingerprint_size (int): Length of the binary fingerprint vector.
        return_residual (bool): If True, returns residual instead of full fingerprinted image.
    """
    def __init__(self, resolution=32, IMAGE_CHANNELS=3, fingerprint_size=100, return_residual=False):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.encoded_size = resolution // (2 ** (log_resolution - 4))
        self.base_shape = (IMAGE_CHANNELS, self.encoded_size, self.encoded_size)

        self.secret_dense = nn.Linear(
            fingerprint_size, self.encoded_size * self.encoded_size * IMAGE_CHANNELS
        )

        self.fingerprint_upsample = nn.Upsample(
            scale_factor=(2 ** (log_resolution - 4), 2 ** (log_resolution - 4)),
            mode='bilinear', align_corners=False
        )

        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

        self.up6 = nn.Conv2d(256, 128, 2, 1, padding=1)
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)

        self.up7 = nn.Conv2d(128, 64, 2, 1, padding=1)
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)

        self.up8 = nn.Conv2d(64, 32, 2, 1, padding=1)
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)

        self.up9 = nn.Conv2d(32, 32, 2, 1, padding=1)
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)

        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, fingerprint, image):
        print("StegaStampEncoder forward() called")
        print(f"  ↳ fingerprint shape: {fingerprint.shape}")
        print(f"  ↳ image shape: {image.shape}")

        fingerprint = relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view(-1, *self.base_shape)
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)

        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))

        up6 = relu(self.up6(F.interpolate(conv5, size=conv4.shape[-2:], mode='bilinear', align_corners=False)))
        up6 = crop_to_match(up6, conv4)
        print("merge6 shapes:", conv4.shape, up6.shape)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))

        up7 = relu(self.up7(F.interpolate(conv6, size=conv3.shape[-2:], mode='bilinear', align_corners=False)))
        up7 = crop_to_match(up7, conv3)
        print("merge7 shapes:", conv3.shape, up7.shape)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))

        up8 = relu(self.up8(F.interpolate(conv7, size=conv2.shape[-2:], mode='bilinear', align_corners=False)))
        up8 = crop_to_match(up8, conv2)
        print("merge8 shapes:", conv2.shape, up8.shape)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))

        up9 = relu(self.up9(F.interpolate(conv8, size=conv1.shape[-2:], mode='bilinear', align_corners=False)))
        up9 = crop_to_match(up9, conv1)
        print("merge9 shapes:", conv1.shape, up9.shape, inputs.shape)
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))

        residual = self.residual(conv10)
        if not self.return_residual:
            residual = sigmoid(residual)
        return residual

class StegaStampDecoder(nn.Module):
    """
    Decoder network that extracts the embedded fingerprint from an image.

    Args:
        resolution (int): Input image resolution.
        IMAGE_CHANNELS (int): Number of image channels (e.g., 3 for RGB).
        fingerprint_size (int): Length of the fingerprint to recover.
    """
    def __init__(self, resolution=32, IMAGE_CHANNELS=3, fingerprint_size=64):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS

        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv6 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv7 = nn.Conv2d(128, 128, 3, 2, 1)

        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(x.size(0), -1)
        return self.dense(x)

