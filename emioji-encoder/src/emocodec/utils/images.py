"""Image conversion and safety checks for inference/benchmarks."""
from __future__ import annotations
import io
import math
import numpy as np
from PIL import Image, ImageOps

MAX_DIM = 8192
MAX_PIXELS = 8192 * 8192
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

def load_rgb(path: str):
    with Image.open(path) as src:
        if src.width > MAX_DIM or src.height > MAX_DIM or src.width * src.height > MAX_PIXELS:
            raise ValueError("Image is too large")
        img = ImageOps.exif_transpose(src).convert("RGB")
        return img.copy()

def image_to_tensor(image):
    import torch
    pixels = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0).float().div(255.0)

def pad_tensor(x, stride: int = 16):
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")

def tensor_to_image(tensor, width: int, height: int):
    values = tensor.detach().cpu().clamp(0, 1)[0, :, :height, :width]
    rgb = (values.permute(1,2,0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(rgb, "RGB")

def metrics(original, reconstructed):
    a = np.asarray(original, dtype=np.float32) / 255.0
    b = np.asarray(reconstructed, dtype=np.float32) / 255.0
    mse = float(np.mean((a-b)**2))
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    output = {"mse": mse, "psnr": psnr}
    try:
        from skimage.metrics import structural_similarity
        output["ssim"] = float(structural_similarity(a,b,data_range=1.0,channel_axis=2))
    except ImportError:
        output["ssim"] = None
    return output
