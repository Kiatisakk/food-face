# Train EmoCodec on Google Colab

## One notebook, no GitHub repository necessary

1. Open https://colab.research.google.com/ .
2. Click **Upload**, choose `notebooks/EmoCodec_Train_Colab.ipynb` from this folder.
3. Go to **Runtime > Change runtime type > GPU** (T4 or equivalent if available).
4. Run the notebook from top to bottom. The first run will download ~800 MB of Oxford-IIIT Pet images/annotations.
5. For first verification, set `EPOCHS=5` (or 1 for a quick pipeline-only smoke test). Increase to 30+ only after checking output.
6. Optionally set `SAVE_TO_DRIVE=True` to persist `emocodec_last.pt` and `emocodec_best.pt` across Colab resets. The notebook will request Drive permissions.
7. At the end, download `emocodec_best.pt` using the Colab file sidebar or set `DOWNLOAD_NOW=True` in the last cell.
8. Copy it to local workspace: `checkpoints/photos_v1.pt`.

Run it from your Mac terminal (within the emioji-encoder root):

```bash
.venv/bin/emocodec encode path/to/photo.png --checkpoint checkpoints/photos_v1.pt --output outputs/photo.emotxt
.venv/bin/emocodec decode outputs/photo.emotxt --checkpoint checkpoints/photos_v1.pt --output outputs/reconstructed.png

EMOCODEC_CHECKPOINT=checkpoints/photos_v1.pt .venv/bin/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 to test visually.

## Notebook implementation

- Self-contained: model source and stable emoji alphabet are embedded automatically by `scripts/build_colab_notebook.py`.
- Rebuild the notebook from Mac any time you alter `src/emocodec/models/vqvae.py` or the alphabet; commit both together.
- Torchvision downloads Oxford data; separate subsets for training and validation.
- VQ codebook warm-start with actual encoder features + periodic revival of unused entries.
- PyTorch AMP for CUDA, Adam, cosine learning rate scheduling.
- Tracks epoch training L1/VQ loss, validation L1 and PSNR, codebook usage and revival counts.
- Saves best model for local inference and last model for optional training resume, validates alphabet metadata.
- Emits a sample `.emotxt`, `.emoc` and reconstructed PNG.
- Project input preprocessing uses an RGB 16× stride. This is lossy; watch for fuzzy reconstructions and codebook collapse.

## Limitations

- Download, training execution and GPU allocation happen in **your Colab session**, not on your Mac.
- Runtime timeouts and free GPU access are dynamic. Save outputs to Drive or download manually.
- Dataset images and model checkpoints intentionally are not committed in Git.
- The notebook initially has **no pretrained model quality guarantee**. It trains your own checkpoint.
- Oxford-IIIT Pet's domain is cats/dogs only; to improve general-image quality try diverse licensed datasets and matched-bpp benchmarks.
