# EmoCodec 🐸 — Image ↔ Emoji ↔ Image

**A research MVP for neural lossy image compression where emoji are non-semantic, reversible labels for VQ-VAE codebook IDs.**

Images are encoded into integer IDs by a *trained* VQ-VAE. Each ID has a stable v1 emoji mapping. Emoji text and compact binary store exactly the same IDs; the decoder uses identical model weights to reconstruct the image. **Emojis themselves do not compress images**; the learned latent bottleneck does. The reconstructed image will be lossy.

## Current implementation

- ✅ Fixed 256 Unicode, single-codepoint emoji alphabet, hash-checked
- ✅ Exact ID ↔ Emoji mapping
- ✅ Strict .emotxt JSON and .emoc binary formats; model hash, version, dimension bounds, SHA-256 corruption checking
- ✅ Trainable CNN VQ-VAE (downsampling stride 16, STE quantization and VQ losses)
- ✅ CLI train, encode, decode, inspect, mapping-check, benchmark
- ✅ Basic PSNR/MSE and optional SSIM; clearly **unmatched** JPEG/WebP quality-75 reference
- ✅ Unit + neural pipeline tests
- ⏳ Pretrained model **not included**. You must train one first; untrained weights cannot reconstruct images accurately.
- ⏳ True matched-bpp WebP/AVIF benchmarking, LPIPS, entropy coding, API and web UI are subsequent milestones.

## Setup (macOS / Linux)

```bash
cd /path/to/food-face/emioji-encoder
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/emocodec mapping-check
```

If you prefer pip: `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"`. Note: PyTorch install depends on your architecture/platform; using `uv` resolves a matching build where one is available.

## Train

Use a *real and appropriately licensed* diverse image dataset to make a meaningful image compressor. You can quickly verify that the entire pipeline works on synthetic patterns:

```bash
.venv/bin/emocodec generate-demo-data --output datasets/demo --count 24
.venv/bin/emocodec train --dataset datasets/demo --output checkpoints/demo.pt --epochs 2 --batch-size 4 --size 64 --hidden 16 --embedding-dim 32 --device cpu
```

The demo set contains only trivial shapes and is **not** evidence of compression performance on photographs.

For more realistic training, put your images (PNG/JPEG/WebP) under `datasets/photos/`, then:

```bash
.venv/bin/emocodec train --dataset datasets/photos --output checkpoints/photos_v1.pt --epochs 30 --batch-size 8 --size 256 --device auto
```

The v1 model has a **fixed** 16× downsampling ratio; changing `--size` changes training crop size, not latent bitrate for a given input image.

## Encode and decode

```bash
.venv/bin/emocodec encode examples/input.png --checkpoint checkpoints/photos_v1.pt --output outputs/image.emotxt
.venv/bin/emocodec encode examples/input.png --checkpoint checkpoints/photos_v1.pt --output outputs/image.emoc
.venv/bin/emocodec inspect outputs/image.emoc
.venv/bin/emocodec decode outputs/image.emotxt --checkpoint checkpoints/photos_v1.pt --output outputs/roundtrip_from_emoji.png
.venv/bin/emocodec decode outputs/image.emoc --checkpoint checkpoints/photos_v1.pt --output outputs/roundtrip_from_binary.png
.venv/bin/emocodec benchmark examples/input.png --checkpoint checkpoints/photos_v1.pt --output outputs/benchmark.json
```

You may point `encode` at any PNG/JPEG/WebP input instead of `examples/input.png`.

## Format contract

**.emotxt**: UTF-8 JSON object with `magic:"EMOTXT1"`, `emoji` (concatenated Unicode code points), and metadata: `version`, `model_id` (checkpoint SHA-256), `alphabet_sha256`, `width`, `height`, `latent_w`, `latent_h`, `stride`, `token_count`, `payload_sha256`.

**.emoc**: five magic bytes `EMOC\x01`, unsigned big-endian 32-bit JSON header length, UTF-8 metadata JSON, then one raw byte per ID (256-symbol codebook). No entropy coding yet.

- The same IDs can round-trip through text/binary losslessly.
- Pixel reconstruction is *lossy* and requires the **exact same checkpoint file**.
- A .emotxt file is meant to be **portable text**, but visual emoji rendering/alteration by social apps is outside our control. Do not silently normalize emoji.
- Fixed binary payload size is `ceil(W/16) × ceil(H/16)` bytes, *plus header*; emoji UTF-8 uses 4 bytes per token, *plus JSON metadata*.
- This is a prototype format, not intended for untrusted Internet input until further fuzzing and defensive parsing.

## Tech Stack

- Python, NumPy, Pillow, PyTorch, Python unittest
- Optional scikit-image for SSIM
- `uv` for virtual environment and local editable install
- Future: CompressAI/rANS, FastAPI + Next.js demo, matched-rate testing

See [Technical Plan](EmoCodec_Technical_Plan.md) for detailed architectural roadmap and research hypotheses.

## Research integrity

Do not interpret visually plausible generated details as exact recovered details. Model checkpoint download size, bits per pixel including metadata, inference latency, distinct codec implementation versions and cross-platform consistency matter. The initial JPEG/WebP metrics are **quality-75 examples, NOT matched bitrate comparisons**.


## Local web demo (FastAPI + let him cook)

```bash
uv pip install --python .venv/bin/python -e ".[dev,web]"
.venv/bin/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. When nested inside `food-face/`, the API serves the combined **let him cook** website and its assets. Enter the kitchen, then scroll down or choose **Emoji Encoder**. Its cream/red UI matches the first kitchen. Drag and drop a PNG/JPEG/WebP file, inspect the generated emoji tokens, compare the reconstruction, copy/download .emotxt/.emoc, or paste a full .emotxt payload and decode it. The food mosaic runs in the browser; the codec sends images to this local API. If the parent is not the Food Face frontend, the standalone EmoCodec UI can be served separately.

You can also run `npm start` from the parent `food-face/` directory to serve the combined page and API together at **http://127.0.0.1:5173** using this project's `.venv`. The API automatically selects `checkpoints/photos_v1.pt` when available, otherwise `checkpoints/demo.pt`. A synthetic demo checkpoint gives **poor reconstructions**; use a real trained checkpoint for meaningful results.

For a new real dataset, train a checkpoint in `checkpoints/photos_v1.pt` and set `EMOCODEC_CHECKPOINT=checkpoints/photos_v1.pt` when starting the server. The API binds to localhost; do not expose it publicly without request authentication, rate-limiting, fuzzing and proper inference resource controls.


## Train on Google Colab

**Ready-to-upload notebook:** [notebooks/EmoCodec_Train_Colab.ipynb](notebooks/EmoCodec_Train_Colab.ipynb)

- Open https://colab.research.google.com/ and upload the notebook from this workspace.
- Select a GPU runtime. Run cells top-to-bottom; default dataset: Oxford-IIIT Pet (~800 MB official photographs and annotations).
- Set EPOCHS=5 for an initial sanity check, then try 30+ and inspect codebook usage and visual metrics.
- Download the trained emocodec_best.pt; copy it into checkpoints/photos_v1.pt on Mac.
- Switch the local web server to the new model using EMOCODEC_CHECKPOINT=checkpoints/photos_v1.pt.
- Optional persistent checkpoints: set SAVE_TO_DRIVE=True in notebook, and authenticate Google Drive.

See [Colab walkthrough](docs/COLAB_GUIDE.md) and [dataset options and licenses](docs/DATASETS.md). For architecture changes, regenerate the notebook with python3 scripts/build_colab_notebook.py, then run tests.
