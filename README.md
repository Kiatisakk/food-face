# let him cook.

**A very unnecessary image-to-food experiment.**

One photo goes through four ridiculous kitchens. It becomes an emoji-food mosaic, gets compressed into an emoji recipe by a neural network, is cooked back into an image, and finally has its pixels rearranged into a real food photo.

**▶ Try it: <https://kiatisakk.github.io/food-face/>** · [ภาษาไทย](README.th.md)

> One photo, four kitchens, one question: why are we doing this?

Built for a Stupid Hackathon. Everything (frontend, VQ-VAE model, Python API and Food Match) lives in this one repository.

---

## Contents

- [The four kitchens](#the-four-kitchens)
- [Quick start](#quick-start)
- [How it runs: browser or local API](#how-it-runs-browser-or-local-api)
- [Deploying to GitHub Pages](#deploying-to-github-pages)
- [Project structure](#project-structure)
- [Commands](#commands)
- [Training a new model](#training-a-new-model)
- [API](#api)
- [Troubleshooting](#troubleshooting)
- [Limitations and credits](#limitations-and-credits)

---

## The four kitchens

```text
                     Your photo
                         │
     01  FOOD KITCHEN    │  photo → food-emoji mosaic + a fake receipt
                         ▼
     02  EMOJI ENCODER   │  image → VQ-VAE → 256-entry codebook ids → emoji
                         ▼
     03  EMOJI DECODER   │  emoji → ids → VQ-VAE decoder → image (lossy)
                         ▼
     04  FOOD MATCH      │  rearrange those exact pixels into the closest of 50,000 food photos
                         ▼
                     Final dish
```

| Kitchen | What it does | Runs on |
| --- | --- | --- |
| **01 Food Kitchen** | Turns the photo into a food-emoji mosaic: resolution 10–300 in steps of 10, a "chaos" slider, an optional colour tint, and a fake receipt with prices and calories. | Canvas + Web Worker |
| **02 Emoji Encoder** | A VQ-VAE turns the image into a grid of codebook ids, one per 16×16 block, and each id is shown as a fixed emoji. You can copy the recipe or download it as `.emotxt` or `.emoc`. | ONNX in the browser, or PyTorch via the local API |
| **03 Emoji Decoder** | Takes a pasted recipe or an uploaded file and rebuilds the image with the same model. | ONNX in the browser, or PyTorch via the local API |
| **04 Food Match** | Searches 50,000 Food-101 photos by colour signature. It then moves the decoded image's own pixels, without changing any colour, until they form the best match. | Web Workers ([food-match](https://github.com/Kiatisakk/food-match)) |

Every kitchen hands its result to the next one with a single button. Each kitchen also takes its own upload, so you can start anywhere.

**Notes on the formats:**

- **The emoji carry no meaning.** They are just stable labels for the 256 codebook entries. The compression comes from the learned latent bottleneck, not from the emoji.
- **`.emotxt`** is JSON. It holds the emoji string, metadata and a SHA-256 checksum of the token payload.
- **`.emoc`** is a compact binary file containing the same ids.
- **Recipes are tied to one model.** Both formats record the model's SHA-256, so a recipe only decodes with the exact checkpoint that made it.

## Quick start

### Just use it

Open **<https://kiatisakk.github.io/food-face/>**. Everything runs in your browser: nothing to install, and your photo is never uploaded.

### Run it locally

**Requirements:** Node.js 18+. Python 3.10+ is needed only for the local API, the tests and training.

```bash
git clone https://github.com/Kiatisakk/food-face.git
cd food-face
```

**Option A: static.** No Python needed. All four kitchens work, with the model running in the browser.

```bash
npm run start:static        # http://localhost:5173
```

**Option B: with the Python API.** Kitchens 2 and 3 then run PyTorch on your machine.

```bash
npm run setup:python        # first time only: creates emioji-encoder/.venv and installs EmoCodec
npm start                   # FastAPI + website on http://127.0.0.1:5173
```

Both options work on macOS, Linux and Windows. `scripts/py.mjs` finds the right Python and virtual-env paths for each platform.

## How it runs: browser or local API

Kitchens 2 and 3 have two interchangeable engines. `codec.js` picks one when the page loads:

1. **Local API.** If `GET /api/status` answers and reports a ready model, which happens under `npm start`, encode and decode go to FastAPI and PyTorch.
2. **Browser.** Otherwise, as on GitHub Pages or with `start:static`, the same model runs in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). The status label then reads *ทำในเบราว์เซอร์* ("running in the browser").

The browser engine lives in `assets/emocodec/emocodec-web.js`. It is a direct port of the Python codec: the same padding, the same `.emotxt` / `.emoc` serialization and the same validation and checksums. It loads ONNX Runtime from jsDelivr only when kitchen 2 or 3 is first used.

**Checked against the Python implementation** on the same image:

| Check | Result |
| --- | --- |
| Token ids, `.emotxt` and `.emoc` | byte-identical |
| Decoded image | 1 level max difference (out of 255) on 0.0007% of pixel values |
| ONNX graph vs PyTorch | 100% identical ids; decoder output differs by < 3e-7 |
| Encode / decode time | ~0.2 s / ~0.5 s for a 498×275 image (single-threaded WebAssembly) |

Files are interchangeable: a recipe made in the browser decodes with the CLI or the API, and the other way round.

## Deploying to GitHub Pages

The site is fully static, so GitHub Pages serves this repository as-is from the `main` branch root. A `.nojekyll` file is included.

If you retrain or swap the checkpoint, re-export the browser model before pushing:

```bash
node scripts/py.mjs -m pip install onnx onnxscript onnxruntime   # one-time, export tools
node scripts/py.mjs --cwd=emioji-encoder scripts/export_onnx.py --checkpoint checkpoints/photos_v1.pt --out ../assets/emocodec
```

The export writes `encoder.onnx`, `decoder.onnx`, `model.json` (holding `model_id`, the checkpoint's SHA-256) and `alphabet.json`. It then checks ONNX Runtime against PyTorch and fails if they disagree.

Food Match's data (signatures and photos) is not stored here. It is loaded from the [food-match](https://github.com/Kiatisakk/food-match) site at `https://kiatisakk.github.io/food-match/data/`.

## Project structure

```text
food-face/
├── index.html                  # welcome page + all four kitchens
├── style.css                   # main theme
├── app.js, renderer.js, menu.js# kitchen 01: emoji mosaic, renderer, emoji menu (price, kcal)
├── codec.js, codec.css         # kitchens 02/03 UI; picks the API or the browser engine
├── match.js, match.css         # kitchen 04 shell: lazy iframe + hand-off of the decoded image
├── assets/
│   ├── emocodec/               # browser model: encoder/decoder.onnx, model.json, alphabet.json,
│   │                           #   emocodec-web.js (port of the Python codec)
│   └── *.png, *.jpg, *.svg     # meme images and favicon
├── scripts/py.mjs              # cross-platform venv setup + Python runner used by npm scripts
├── serve.json                  # settings for `npm run start:static`
│
├── emioji-encoder/             # EmoCodec: the Python package, API, model and notebook
│   ├── apps/api/main.py        # FastAPI app; also serves the website
│   ├── src/emocodec/           # VQ-VAE model, emoji mapping, .emotxt/.emoc formats, training
│   ├── checkpoints/photos_v1.pt# the model used by the site (~3.6 MB)
│   ├── scripts/export_onnx.py  # checkpoint → ONNX for the browser, with a PyTorch check
│   ├── notebooks/EmoCodec_Train_Colab.ipynb
│   └── tests/
│
└── food-match/                 # kitchen 04 (embedded copy of the food-match app)
    ├── index.html, style.css, app.js
    ├── bridge.js               # receives the decoded PNG via postMessage, reports its height
    ├── core.js, sigcore.js     # pixel assignment + colour signatures
    ├── worker.js, score-worker.js, assign-worker.js
    └── test.js
```

No submodules: `emioji-encoder/` and `food-match/` are ordinary folders in this repository.

## Commands

| Command | What it does |
| --- | --- |
| `npm run start:static` | Static site on `localhost:5173`; kitchens 2 and 3 run in the browser |
| `npm run setup:python` | Creates `emioji-encoder/.venv` and installs EmoCodec with its web dependencies |
| `npm start` | FastAPI + website on `127.0.0.1:5173`; kitchens 2 and 3 use PyTorch |
| `npm run test:api` | EmoCodec + FastAPI unit tests (21 tests) |
| `npm run test:match` | Food Match search and pixel-assignment tests |

## Training a new model

Notebook: [`emioji-encoder/notebooks/EmoCodec_Train_Colab.ipynb`](emioji-encoder/notebooks/EmoCodec_Train_Colab.ipynb)

1. Upload the notebook to [Google Colab](https://colab.research.google.com/), choose a GPU runtime and run it.
2. Put the resulting checkpoint in `emioji-encoder/checkpoints/`.
3. Choose which checkpoint to use with an environment variable:

   ```bash
   EMOCODEC_CHECKPOINT=emioji-encoder/checkpoints/photos_v1.pt npm start
   ```

4. Re-export it for the browser (see [Deploying](#deploying-to-github-pages)).

You don't need to train anything to try the site: the default `photos_v1.pt` is already in the repository.

More: [EmoCodec README](emioji-encoder/README.md) · [Colab guide](emioji-encoder/docs/COLAB_GUIDE.md) · [Dataset notes](emioji-encoder/docs/DATASETS.md)

## API

Available when running `npm start`. The website and the API share one port.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The website |
| `GET /api/status` | Whether a model is loaded, plus its id |
| `POST /api/encode` | Image upload → emoji text, binary (base64), reconstruction and stats |
| `POST /api/decode` | `.emotxt` / `.emoc` upload, or pasted text → decoded PNG (base64) |
| `GET /food-match/index.html?embedded=1` | Food Match UI embedded in kitchen 04 |

Limits for both the API and the browser engine: uploads up to **8 MB**, images up to **2048×2048 px**.

```bash
curl http://127.0.0.1:5173/api/status
```

## Troubleshooting

**Port 5173 is already in use.** Find the process with `lsof -nP -iTCP:5173 -sTCP:LISTEN` on macOS/Linux, or `netstat -ano | findstr :5173` on Windows. Stop the old server, then start again.

**"Python environment missing".** Run `npm run setup:python` first. It needs Python 3.10+ on your `PATH` (`python3`, `python` or `py -3`).

**Kitchens 2 and 3 show an error in the browser.** The browser engine downloads ONNX Runtime from `cdn.jsdelivr.net` the first time it is used. Check your connection and any content blockers, or use `npm start` to run the model locally instead.

**Food Match never shows dishes.** Its data loads from `https://kiatisakk.github.io/food-match/data/`, so it needs internet access. The chosen photo is fetched from Hugging Face, with a hosted fallback.

**Opening `index.html` directly (`file://`).** Browsers block workers and fetches from local files, so serve the page with `npm run start:static` or `npm start` instead.

## Limitations and credits

- **Kitchen 01 is entertainment.** Receipt prices and calories are made up.
- **EmoCodec is lossy compression, not encryption.** Anyone with the same checkpoint can decode a recipe, so don't treat emoji as a way to keep a photo private.
- **Food Match moves pixels; it does not generate images.** The result contains exactly the decoded image's pixels, rearranged. Its index is built from [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (ETH Zürich), which is licensed for research use only, so check those terms before any commercial use.
- **Third-party code and images:**
  - Browser inference uses [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT).
  - Memes: [Would you let him cook? — Imgflip](https://imgflip.com/i/7imzuy) · [Let Him Cook — Know Your Meme](https://knowyourmeme.com/photos/2517089-let-him-cook-let-that-boy-cook).

---

*0 Michelin stars. Many experiments. Let him cook.*
