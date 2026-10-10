"""Generate a self-contained, model-compatible Google Colab training notebook.

The model cell is copied verbatim from the repo's VQ-VAE implementation to
preserve checkpoint compatibility. Regenerate after editing model architecture.
Uses the standard library only; no notebook-specific package required.
"""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "EmoCodec_Train_Colab.ipynb"
cells = []

def md(body: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(body).strip("\n").splitlines(keepends=True)})

def code(body: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": textwrap.dedent(body).strip("\n").splitlines(keepends=True)})

md("""
# 🐸 EmoCodec — Train the VQ-VAE on Google Colab

A **self-contained** training notebook. Upload this .ipynb file into Google Colab;
you do not need to clone a Git repository.

**Start:** Runtime → Change runtime type → T4 GPU (if available) → Run all.

The same CNN/VQ-VAE model definition as the desktop repo is embedded below, and
the checkpoint format is compatible with EmoCodec's local encoder/decoder.

**Default dataset:** [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) — real photographs of 37 pet breeds.
Torchvision downloads the images and annotations automatically (roughly 800 MB).
We create a reproducible **train/validation split from trainval**. The published
test split is kept untouched. This dataset is intentionally narrow (pets); good
for a training prototype, **not** sufficient for general-purpose image compression.

**Larger follow-up dataset:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) —
800 high-resolution training images and 100 validation images, broad scenes but
requires multi-GB disk/download. See docs/DATASETS.md in this repo for details.

**Important:** Emoji tokens are arbitrary code IDs; not semantic image labels.
The same checkpoint and alphabet version are mandatory for decoding.
Compression quality is not guaranteed to exceed JPEG/WebP/AVIF.

Colab resources are dynamic; GPU availability, session length and quotas are not guaranteed.
Keep a checkpoint in Google Drive or download it before Colab disconnects.
""")

md("""
## 1. Imports, device and reproducibility

Colab normally includes torch, torchvision, PIL and matplotlib.
If an import is missing, install the package in a separate Colab cell, then restart.
Avoid blindly upgrading torch without matching torchvision's build.
""")
code("""
import os, sys, io, json, random, math, hashlib, struct, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageFile
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", __import__("torchvision").__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
""")

md("""
## 2. Hyperparameters and checkpoint persistence

Start with 256px crops, 8 images per batch and 30 epochs. Increase epochs after
checking metrics and reconstruction samples. Smaller Colab GPUs may require batch
size 4 or image size 128. All model versions here use a 16× stride.

Optional: Set SAVE_TO_DRIVE = True to mount Google Drive, retain best/last checkpoint,
and resume after Colab runtime resets. Otherwise artifacts stay in ephemeral /content.
""")
code("""
SEED = 42
DATASET_CHOICE = "oxford_pets"   # "oxford_pets" or "local_folder"
LOCAL_IMAGES = "/content/images" # used only with "local_folder"
DATA_ROOT = Path("/content/emocodec_data")
SAVE_TO_DRIVE = False
RESUME_IF_AVAILABLE = True

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 2e-4
HIDDEN = 48
EMBEDDING_DIM = 64
CODEBOOK_SIZE = 256
BETA = 0.25
NUM_WORKERS = 2
VALIDATION_FRACTION = 0.10
MAX_TRAIN_IMAGES = 0       # 0 means use all
MAX_VAL_IMAGES = 256
DEAD_CODE_REVIVE_EVERY = 5

assert IMAGE_SIZE % 16 == 0 and CODEBOOK_SIZE == 256
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True if DEVICE.type == "cuda" else False

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    ARTIFACT_DIR = Path("/content/drive/MyDrive/EmoCodec")
else:
    ARTIFACT_DIR = Path("/content/EmoCodec")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

BEST_PATH = ARTIFACT_DIR / "emocodec_best.pt"
LAST_PATH = ARTIFACT_DIR / "emocodec_last.pt"
print("Checkpoints directory:", ARTIFACT_DIR)
""")

md("""
## 3. Load images from Oxford-IIIT Pet

The official dataset is available through torchvision with trainval/test splits.
We hold out a deterministic 10% **validation** subset from trainval; do not use
the official test split for tuning. If running with your own images, upload/unzip
to LOCAL_IMAGES, change DATASET_CHOICE above, and rerun this cell.

No labels are needed: this is reconstruction training.
""")
code("""
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

class LocalImages(Dataset):
    def __init__(self, folder, transform):
        self.paths = sorted(p for p in Path(folder).rglob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        if not self.paths:
            raise ValueError("No local images found at " + str(folder))
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as src:
            img = ImageOps.exif_transpose(src).convert("RGB")
            return self.transform(img), 0

if DATASET_CHOICE == "oxford_pets":
    print("Downloading/reusing Oxford-IIIT Pet official images and annotations...")
    train_base = datasets.OxfordIIITPet(
        root=str(DATA_ROOT), split="trainval", target_types="category",
        transform=train_transform, download=True)
    val_base = datasets.OxfordIIITPet(
        root=str(DATA_ROOT), split="trainval", target_types="category",
        transform=val_transform, download=False)
elif DATASET_CHOICE == "local_folder":
    train_base = LocalImages(LOCAL_IMAGES, train_transform)
    val_base = LocalImages(LOCAL_IMAGES, val_transform)
else:
    raise ValueError("DATASET_CHOICE must be oxford_pets or local_folder")

rng = np.random.default_rng(SEED)
order = rng.permutation(len(train_base)).tolist()
n_val = max(1, int(len(order) * VALIDATION_FRACTION))
assert len(order) - n_val > 0
validation_indices = order[:n_val][:MAX_VAL_IMAGES if MAX_VAL_IMAGES else None]
training_indices = order[n_val:][:MAX_TRAIN_IMAGES if MAX_TRAIN_IMAGES else None]
assert not set(validation_indices).intersection(training_indices)
train_set = Subset(train_base, training_indices)
val_set = Subset(val_base, validation_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=DEVICE.type=="cuda")
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=DEVICE.type=="cuda")
print(f"Dataset: {DATASET_CHOICE} | train={len(train_set)} | val={len(val_set)}")
assert len(train_set)>0 and len(val_set)>0
images,_ = next(iter(train_loader))
print("First batch:", tuple(images.shape), "range:",float(images.min()),float(images.max()))
fig,ax = plt.subplots(1, min(4,len(images)), figsize=(12,3))
for a,img in zip(np.atleast_1d(ax), images):
    a.imshow(img.permute(1,2,0).numpy()); a.axis("off")
plt.show()
""")

md("""
## 4. Embedded VQ-VAE architecture (copied from desktop source)

**Do not reorder layers or rename state_dict keys** unless you also update
the desktop EmoCodec decoder. This cell is automatically regenerated from
src/emocodec/models/vqvae.py by scripts/build_colab_notebook.py.
""")
model_source = (ROOT / "src" / "emocodec" / "models" / "vqvae.py").read_text(encoding="utf-8")
code(model_source)

md("""
## 5. Fixed Emoji alphabet (v1) — byte-for-byte compatible

The alphabet order and SHA-256 match src/emocodec/assets/emoji_alphabet_v1.json.
It is **independent** of visual glyphs on Mac/Windows/Android.
""")
alphabet_json = (ROOT / "src" / "emocodec" / "assets" / "emoji_alphabet_v1.json").read_text(encoding="utf-8")
code("ALPHABET_META = json.loads(" + repr(alphabet_json) + ")\n" + """
ALPHABET = tuple(ALPHABET_META["emoji"])
assert len(ALPHABET) == 256 and len(set(ALPHABET)) == 256
assert all(len(ch)==1 and len(ch.encode("utf-8"))==4 for ch in ALPHABET)
ALPHABET_SHA256 = hashlib.sha256("".join(ALPHABET).encode("utf-8")).hexdigest()
print("Emoji alphabet SHA-256:", ALPHABET_SHA256)
""")

md("""
## 6. Build model, warm-start VQ codebook and optimizer

Warm-start embeddings from encoder outputs (instead of relying only on tiny
random values) and periodically revive unused codebook entries. This is
a prototype anti-collapse strategy, not a guarantee that all 256 IDs get used.
The desktop architecture and state_dict remain unchanged.
""")
code("""
model = VQVAE(Architecture(
    hidden=HIDDEN, embedding_dim=EMBEDDING_DIM,
    codebook_size=CODEBOOK_SIZE, beta=BETA)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
start_epoch = 0
best_l1 = float("inf")
history = []

if RESUME_IF_AVAILABLE and LAST_PATH.is_file():
    state = torch.load(LAST_PATH, map_location="cpu", weights_only=True)
    if state.get("architecture") != model.config_dict() or state.get("alphabet_sha256") != ALPHABET_SHA256:
        raise RuntimeError("Saved checkpoint architecture/alphabet mismatch. Change output directory or config.")
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    start_epoch = state["epoch"]
    best_l1 = state["best_l1"]
    history = state.get("history",[])
    print("Resumed after epoch",start_epoch)
else:
    model.eval()
    features=[]
    with torch.no_grad():
        for images,_ in train_loader:
            z = model.encoder(images.to(DEVICE)).permute(0,2,3,1).reshape(-1,EMBEDDING_DIM)
            features.append(z.detach())
            if sum(a.shape[0] for a in features) >= 2048: break
        pool = torch.cat(features,dim=0)
        picks = torch.randint(len(pool),(CODEBOOK_SIZE,),device=DEVICE)
        init_vectors=pool[picks] + 0.005 * torch.randn_like(pool[picks])
        model.vq.embedding.weight.copy_(init_vectors)
    print("Initialized 256 VQ vectors from real encoder activations.")

def revive_codes(usage):
    dead = torch.where(usage == 0)[0]
    if len(dead) == 0: return 0
    model.eval()
    images,_ = next(iter(train_loader))
    with torch.no_grad():
        z = model.encoder(images.to(DEVICE)).permute(0,2,3,1).reshape(-1,EMBEDDING_DIM)
        picks=torch.randint(z.shape[0],(len(dead),),device=DEVICE)
        weights=model.vq.embedding.weight
        weights[dead.to(DEVICE)] = z[picks] + torch.randn_like(z[picks])*0.005
        state=optimizer.state.get(weights,{})
        for key in ("exp_avg","exp_avg_sq"):
            if key in state: state[key][dead.to(DEVICE)] = 0
    return len(dead)

def save_checkpoint(path,epoch,include_optimizer=False):
    payload={
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "architecture":model.config_dict(),
        "alphabet_sha256":ALPHABET_SHA256,
        "epoch":int(epoch),"best_l1":float(best_l1),
        "dataset":DATASET_CHOICE,"training_size":IMAGE_SIZE,
    }
    if include_optimizer:
        payload["optimizer"]=optimizer.state_dict()
        payload["scheduler"]=scheduler.state_dict()
        payload["history"]=history
    torch.save(payload,str(path))
""")

md("""
## 7. Training & validation (runs with GPU if available)

Tracks L1, VQ loss, PSNR and codebook usage. Best validation checkpoint is
saved independently of last/resume checkpoint. Validation samples are
disjoint from training. No inference with a checkpoint from an unrelated alphabet.

To keep Colab responsive start at EPOCHS=5, confirm output, then increase to 30+.
A low PSNR or repeated single-code use means compression quality needs work.
""")
code("""
for epoch in range(start_epoch,EPOCHS):
    start = time.time()
    model.train()
    total_l1=total_loss=total_vq=0.0
    usage=torch.zeros(CODEBOOK_SIZE, dtype=torch.long)
    for batch,_ in tqdm(train_loader,desc=f"Train {epoch+1}/{EPOCHS}",leave=False):
        x=batch.to(DEVICE,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type,enabled=(DEVICE.type=="cuda")):
            reconstructed,vq_loss,ids=model(x)
            l1=F.l1_loss(reconstructed,x)
            loss=l1+vq_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss+=float(loss.item())
        total_l1+=float(l1.item())
        total_vq+=float(vq_loss.item())
        usage+=torch.bincount(ids.detach().reshape(-1).cpu(),minlength=CODEBOOK_SIZE)
    scheduler.step()

    model.eval()
    val_l1=val_mse=0.0
    seen=0
    with torch.inference_mode():
        for batch,_ in val_loader:
            x=batch.to(DEVICE,non_blocking=True)
            y,_,_=model(x)
            batch_n=len(x)
            val_l1+=F.l1_loss(y,x).item()*batch_n
            val_mse+=F.mse_loss(y,x).item()*batch_n
            seen+=batch_n
    val_l1/=seen
    val_mse/=seen
    val_psnr=-10*math.log10(max(val_mse,1e-12))
    used=int((usage>0).sum())

    # Revive empty codes only for subsequent epochs; keep best checkpoint as
    # the exact model evaluated immediately above.
    if val_l1 < best_l1:
        best_l1=val_l1
        save_checkpoint(BEST_PATH,epoch+1)
        saved="*best*"
    else:
        saved=""
    revived=0
    if (epoch+1)%DEAD_CODE_REVIVE_EVERY==0:
        revived=revive_codes(usage)
    history.append({"epoch":epoch+1,"train_l1":total_l1/len(train_loader),
                    "train_vq":total_vq/len(train_loader),
                    "train_loss":total_loss/len(train_loader),
                    "val_l1":val_l1,"val_psnr":val_psnr,
                    "codebook_used":used,"codebook_revived":revived})
    save_checkpoint(LAST_PATH,epoch+1,include_optimizer=True)
    print(f"Epoch {epoch+1:03} | L1={total_l1/len(train_loader):.4f} "
          f"Val L1={val_l1:.4f} | PSNR={val_psnr:.2f}dB "
          f"| Codes={used}/{CODEBOOK_SIZE} | Revived={revived} "
          f"| {time.time()-start:.0f}s {saved}")
print("Training complete. Best checkpoint:",BEST_PATH)
""")

md("""
## 8. Plot training history & visualize original/reconstructed images

These previews **do not** certify superiority to any classical codec.
All rate-distortion comparisons must be done at comparable actual bits/pixel.
""")
code("""
if not history:
    raise RuntimeError("No epochs run. Set EPOCHS above the completed resume epoch.")
fig,axes=plt.subplots(1,3,figsize=(15,3.5))
axes[0].plot([h["epoch"] for h in history],[h["train_l1"] for h in history],label="Train L1")
axes[0].plot([h["epoch"] for h in history],[h["val_l1"] for h in history],label="Val L1")
axes[0].set_title("Reconstruction loss");axes[0].legend()
axes[1].plot([h["epoch"] for h in history],[h["val_psnr"] for h in history])
axes[1].set_title("Validation PSNR (dB)")
axes[2].plot([h["epoch"] for h in history],[h["codebook_used"] for h in history])
axes[2].set_title("Active VQ codes");axes[2].set_ylim(0,CODEBOOK_SIZE+5)
plt.show()

best=torch.load(BEST_PATH,map_location="cpu",weights_only=True)
model.load_state_dict(best["state_dict"])
model.to(DEVICE).eval()
images,_=next(iter(val_loader))
with torch.inference_mode():
    recon,_,indices=model(images.to(DEVICE))
fig,axes=plt.subplots(2,min(4,len(images)),figsize=(13,6))
for col in range(min(4,len(images))):
    axes[0,col].imshow(images[col].permute(1,2,0).numpy())
    axes[1,col].imshow(recon[col].cpu().permute(1,2,0).numpy().clip(0,1))
    axes[0,col].axis("off");axes[1,col].axis("off")
axes[0,0].set_title("Originals (top) / neural reconstruction (bottom)")
plt.tight_layout();plt.show()
print("Best epoch:",best["epoch"],"best validation L1:",best["best_l1"])
""")

md("""
## 9. Test actual emoji → IDs → reconstructed PNG

Generate both output formats with exactly the same contract as desktop
EmoCodec. This round-trip is **exact for IDs**, but image pixels are lossy.
Output names have real checkpoint hash binding.
""")
code("""
model.eval()
with torch.inference_mode():
    x=images[:1].to(DEVICE)
    ids=model.encode_indices(x).cpu().reshape(-1).tolist()
    y=model.decode_indices(torch.tensor(ids,device=DEVICE).reshape(1,IMAGE_SIZE//16,IMAGE_SIZE//16))
assert len(ids)==(IMAGE_SIZE//16)**2
emoji="".join(ALPHABET[i] for i in ids)
back={v:i for i,v in enumerate(ALPHABET)}
assert [back[c] for c in emoji]==ids

checkpoint_hash=hashlib.sha256(BEST_PATH.read_bytes()).hexdigest()
meta={"version":1,"model_id":checkpoint_hash,"alphabet_sha256":ALPHABET_SHA256,
      "width":IMAGE_SIZE,"height":IMAGE_SIZE,
      "latent_w":IMAGE_SIZE//16,"latent_h":IMAGE_SIZE//16,"stride":16,
      "token_count":len(ids),"payload_sha256":hashlib.sha256(bytes(ids)).hexdigest()}
text_payload=json.dumps({**meta,"magic":"EMOTXT1","emoji":emoji},
                        ensure_ascii=False,separators=(",",":"))+"\\n"
header=json.dumps(meta,sort_keys=True,separators=(",",":")).encode("utf-8")
binary_payload=b"EMOC\\x01"+struct.pack(">I",len(header))+header+bytes(ids)
assert len(binary_payload)>len(ids)
(ARTIFACT_DIR/"sample.emotxt").write_text(text_payload,encoding="utf-8")
(ARTIFACT_DIR/"sample.emoc").write_bytes(binary_payload)
output=(y[0].cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
Image.fromarray(output,"RGB").save(ARTIFACT_DIR/"sample_decoded.png")
print("Tokens:",len(ids))
print("Text bytes:",len(text_payload.encode("utf-8")))
print("Binary bytes:",len(binary_payload))
print("Binary bits/pixel:",8*len(binary_payload)/(IMAGE_SIZE*IMAGE_SIZE))
print("Checkpoint SHA256:",checkpoint_hash)
print("Preview tokens:",emoji[:24])
""")

md("""
## 10. Download checkpoint and use it locally

Download **emocodec_best.pt** from the Colab Files sidebar, or use this button cell
(it triggers a browser download). Put it in your desktop repo at:

checkpoints/photos_v1.pt

Then run from the existing EmoCodec repository:

    .venv/bin/emocodec encode path/to/photo.png --checkpoint checkpoints/photos_v1.pt --output outputs/photo.emotxt
    .venv/bin/emocodec decode outputs/photo.emotxt --checkpoint checkpoints/photos_v1.pt --output outputs/reconstructed.png

To use the local web demo:
    EMOCODEC_CHECKPOINT=checkpoints/photos_v1.pt .venv/bin/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

The decoder requires the **exact same model file bytes** to satisfy model_id SHA-256.
Saving or re-exporting the same weights can change file hash; use original downloaded .pt.
""")
code("""
print("Checkpoint ready:",BEST_PATH)
print("Size (MiB):",round(BEST_PATH.stat().st_size / 1024**2,2))
print("Saved artifacts:",*[p.name for p in ARTIFACT_DIR.iterdir()],sep="\\n  ")
# Enable one-click download if running in Colab:
try:
    from google.colab import files as colab_files
    DOWNLOAD_NOW = False  # set to True and run this cell to download the .pt
    if DOWNLOAD_NOW:
        colab_files.download(str(BEST_PATH))
except ImportError:
    print("Outside Colab: copy the .pt file directly from the artifact directory.")
""")

md("""
## References / limitations

- Oxford-IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Torchvision Pet loader: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- VQ-VAE paper: https://arxiv.org/abs/1711.00937
- License/reuse of each dataset should be checked at its source before redistribution.
- Colab quotas vary and free runtimes may terminate; persist checkpoints to Drive.
- A single VQ-VAE with fixed 16× spatial bottleneck often yields blurry images or
  suffers codebook collapse. Next research steps: better latent initialization,
  EMA quantizer or RVQ, entropy coding, perceptual losses, matched-bpp testing.
""")

nb={
    "cells": cells,
    "metadata": {
        "colab": {"provenance":[]},
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python"}
    },
    "nbformat":4,"nbformat_minor":4
}
TARGET.parent.mkdir(parents=True,exist_ok=True)
TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print(f"Wrote {TARGET.relative_to(ROOT)}: {len(cells)} cells ({TARGET.stat().st_size:,} bytes)")
