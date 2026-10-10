"""CPU/MPS/CUDA training on a recursively discovered folder of RGB images."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

def pick_device(name: str = "auto") -> str:
    import torch
    if name != "auto":
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple Metal MPS not available")
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ImageFolder:
    def __init__(self, root: str, size: int = 256):
        from torch.utils.data import Dataset
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})
        if not self.files:
            raise ValueError(f"No images found in {root}; provide a training dataset")
        if size % 16 != 0:
            raise ValueError("Training size must be a multiple of 16")
        self.size = size
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        import torch
        with Image.open(self.files[idx]) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            image = ImageOps.fit(image, (self.size, self.size), method=Image.Resampling.BICUBIC)
            pixels = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(pixels).permute(2,0,1).float().div(255.0)

def model_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def load_checkpoint(path: str, device: str = "cpu"):
    import torch
    from emocodec.models.vqvae import Architecture, VQVAE
    from emocodec.codec.mapping import ALPHABET_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("alphabet_sha256") != ALPHABET_SHA256:
        raise ValueError("Checkpoint requires a different emoji alphabet")
    arch = Architecture(**payload["architecture"])
    model = VQVAE(arch)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, model_sha256(path)

def train(*, dataset: str, output: str, epochs: int = 10, batch_size: int = 8,
          size: int = 256, hidden: int = 48, embedding_dim: int = 64,
          learning_rate: float = 2e-4, beta: float = 0.25,
          device: str = "auto", seed: int = 42, workers: int = 0):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from emocodec.models.vqvae import VQVAE, Architecture
    from emocodec.codec.mapping import ALPHABET_SHA256
    if min(epochs,batch_size) <= 0:
        raise ValueError("epochs and batch size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = pick_device(device)
    data = ImageFolder(dataset, size=size)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=workers)
    model = VQVAE(Architecture(hidden=hidden, embedding_dim=embedding_dim, beta=beta)).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        reconstruction_total = 0.0
        total_loss = 0.0
        count = 0
        usage = set()
        for images in loader:
            images = images.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, vq_loss, ids = model(images)
            recon_loss = F.mse_loss(reconstructed, images)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            count += 1
            total_loss += loss.item()
            reconstruction_total += recon_loss.item()
            usage.update(ids.detach().cpu().unique().tolist())
        print(f"Epoch {epoch+1:03}/{epochs}: loss={total_loss/count:.5f} recon={reconstruction_total/count:.5f} codebook_used={len(usage)}/256", flush=True)
    p = Path(output)
    p.parent.mkdir(parents=True,exist_ok=True)
    checkpoint = {"state_dict": model.cpu().state_dict(), "architecture": model.config_dict(),
                  "alphabet_sha256": ALPHABET_SHA256, "training_epochs":epochs,
                  "training_images":len(data), "training_size":size}
    torch.save(checkpoint, p)
    sha = model_sha256(p)
    print(f"Saved {p} (model_id={sha})")
    return sha
