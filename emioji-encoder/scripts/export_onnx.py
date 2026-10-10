"""Export a trained EmoCodec checkpoint to ONNX so the web app can encode/decode in the browser.

Writes <out>/encoder.onnx, <out>/decoder.onnx, <out>/model.json and <out>/alphabet.json (model_id = SHA-256 of the checkpoint,
the same id the Python codec writes into .emotxt/.emoc files, so files stay interchangeable).

    python scripts/export_onnx.py --checkpoint checkpoints/photos_v1.pt --out ../assets/emocodec

Then checks that ONNX Runtime gives the same token ids and images as PyTorch on a few test images.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from emocodec.codec.mapping import ALPHABET_SHA256  # noqa: E402
from emocodec.training.trainer import load_checkpoint  # noqa: E402


class EncodeIds(nn.Module):
    """(1, 3, H, W) float in [0, 1], H and W multiples of 16 -> (1, H/16, W/16) int64 token ids."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.vq.nearest(self.model.encoder(x))


class DecodeIds(nn.Module):
    """(1, h, w) int64 token ids -> (1, 3, 16h, 16w) float in [0, 1]."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, ids):
        return self.model.decoder(self.model.vq.lookup(ids))


def export(module, example, path, in_name, out_name, in_axes, out_axes):
    torch.onnx.export(module, (example,), path, input_names=[in_name], output_names=[out_name],
                      dynamic_axes={in_name: in_axes, out_name: out_axes}, opset_version=17, dynamo=False)


def verify(model, enc_path, dec_path, seeds=(0, 1, 2)):
    import onnxruntime as ort
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    for seed, (h, w) in zip(seeds, [(64, 96), (128, 128), (240, 320)]):
        g = torch.Generator().manual_seed(seed)
        # smooth random image (photo-like) rather than white noise
        x = torch.nn.functional.interpolate(torch.rand(1, 3, h // 8, w // 8, generator=g), size=(h, w), mode="bilinear")
        with torch.inference_mode():
            ids_pt = model.encode_indices(x)
            img_pt = model.decode_indices(ids_pt)
        ids_ox = enc.run(None, {"image": x.numpy()})[0]
        img_ox = dec.run(None, {"ids": ids_pt.numpy()})[0]
        same = float((ids_ox == ids_pt.numpy()).mean())
        diff = float(np.abs(img_ox - img_pt.numpy()).max())
        print(f"  {w}x{h}: token ids identical {same:.2%}, decoded image max abs diff {diff:.2e}")
        assert same > 0.99 and diff < 1e-3, "ONNX export does not match PyTorch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/photos_v1.pt")
    ap.add_argument("--out", default=os.path.join("..", "assets", "emocodec"))
    a = ap.parse_args()
    model, model_id = load_checkpoint(a.checkpoint)
    os.makedirs(a.out, exist_ok=True)
    enc_path, dec_path = os.path.join(a.out, "encoder.onnx"), os.path.join(a.out, "decoder.onnx")
    export(EncodeIds(model).eval(), torch.rand(1, 3, 64, 96), enc_path, "image", "ids",
           {2: "H", 3: "W"}, {1: "h", 2: "w"})
    export(DecodeIds(model).eval(), torch.zeros(1, 4, 6, dtype=torch.long), dec_path, "ids", "image",
           {1: "h", 2: "w"}, {2: "H", 3: "W"})
    info = {"model_id": model_id, "model_file": os.path.basename(a.checkpoint), "alphabet_sha256": ALPHABET_SHA256,
            "stride": 16, "codebook_size": model.config.codebook_size, "architecture": model.config_dict()}
    json.dump(info, open(os.path.join(a.out, "model.json"), "w"), indent=2)
    import shutil  # the browser codec needs the same emoji alphabet
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "src", "emocodec", "assets", "emoji_alphabet_v1.json"),
                os.path.join(a.out, "alphabet.json"))
    for p in (enc_path, dec_path):
        print(f"wrote {p} ({os.path.getsize(p) / 1e6:.2f} MB)")
    print("verifying against PyTorch:")
    verify(model, enc_path, dec_path)
    print("ok, model_id", model_id)


if __name__ == "__main__":
    main()
