"""Convolutional VQ-VAE with vector quantization and a straight-through estimator."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Architecture:
    hidden: int = 48
    embedding_dim: int = 64
    codebook_size: int = 256
    beta: float = 0.25

class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(channels, channels, 1))
    def forward(self, x):
        return x + self.net(x)

class Encoder(nn.Module):
    def __init__(self, hidden: int, embedding_dim: int):
        super().__init__()
        layers = []
        in_channels = 3
        for width in (hidden, hidden, hidden*2, hidden*2):
            layers.extend([nn.Conv2d(in_channels, width, 4, 2, 1), nn.ReLU()])
            in_channels = width
        self.net = nn.Sequential(*layers, Residual(in_channels), nn.Conv2d(in_channels, embedding_dim, 1))
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, hidden: int, embedding_dim: int):
        super().__init__()
        channels = (hidden*2, hidden*2, hidden, hidden)
        layers = [nn.Conv2d(embedding_dim, hidden*2, 3, padding=1), Residual(hidden*2)]
        in_channels = hidden*2
        for width in channels:
            layers.extend([nn.ConvTranspose2d(in_channels, width, 4, 2, 1), nn.ReLU()])
            in_channels = width
        layers.extend([nn.Conv2d(in_channels, 3, 3, padding=1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.beta = beta
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def nearest(self, z: torch.Tensor) -> torch.Tensor:
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        weights = self.embedding.weight
        dist = flat.square().sum(1, keepdim=True) + weights.square().sum(1) - 2 * (flat @ weights.t())
        return dist.argmin(dim=1).reshape(b, h, w)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices.long()).permute(0, 3, 1, 2).contiguous()

    def forward(self, z):
        indices = self.nearest(z)
        quantized = self.lookup(indices)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        ste = z + (quantized - z).detach()
        return ste, codebook_loss + self.beta * commitment_loss, indices

class VQVAE(nn.Module):
    stride = 16
    def __init__(self, config: Architecture | None = None):
        super().__init__()
        self.config = config or Architecture()
        if self.config.codebook_size != 256:
            raise ValueError("Emoji alphabet v1 supports exactly 256 codebook entries")
        self.encoder = Encoder(self.config.hidden, self.config.embedding_dim)
        self.vq = VectorQuantizer(self.config.codebook_size, self.config.embedding_dim, self.config.beta)
        self.decoder = Decoder(self.config.hidden, self.config.embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        q, loss_vq, indices = self.vq(z)
        return self.decoder(q), loss_vq, indices

    @torch.inference_mode()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        return self.vq.nearest(self.encoder(x))

    @torch.inference_mode()
    def decode_indices(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.vq.lookup(ids))

    def config_dict(self):
        return asdict(self.config)
