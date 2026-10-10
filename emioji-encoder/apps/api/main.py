"""Local-only EmoCodec web API. Start: uvicorn apps.api.main:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from emocodec.codec.formats import decode_text, encode_binary
from emocodec.service import encode_image, decode_image
from emocodec.training.trainer import model_sha256
from emocodec.utils.images import load_rgb, metrics

app = FastAPI(title="EmoCodec API", version="0.1.0")
ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT.parent  # food-face repository root (EmoCodec is nested)
FRONTEND_FILES = {"style.css", "codec.css", "menu.js", "renderer.js", "app.js", "codec.js", "match.js", "match.css"}
FOOD_MATCH_FILES = {"index.html", "style.css", "app.js", "bridge.js", "core.js", "sigcore.js", "worker.js", "assign-worker.js", "score-worker.js"}
MAX_UPLOAD = 8 * 1024 * 1024
MAX_DIMENSION = 2048

def checkpoint() -> Path:
    trained = ROOT / "checkpoints" / "photos_v1.pt"
    default = trained if trained.is_file() else ROOT / "checkpoints" / "demo.pt"
    p = Path(os.environ.get("EMOCODEC_CHECKPOINT", str(default)))
    if not p.is_file():
        raise HTTPException(503, f"Model checkpoint not found: {p}. Train a model first.")
    return p

def b64_png(img: Image.Image) -> str:
    out=BytesIO()
    img.save(out,format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")

@app.get("/",response_class=HTMLResponse)
def index():
    page = FRONTEND / "index.html"
    if not page.is_file():
        page = ROOT / "apps" / "web" / "index.html"
    return page.read_text(encoding="utf-8")

if (FRONTEND / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="food-assets")

@app.get("/api/status")
def status():
    try:
        p=checkpoint()
        return {"ready":True,"model_id":model_sha256(p),"model_file":p.name,"demo_model":p.name=="demo.pt"}
    except HTTPException:
        return {"ready":False,"model_id":None,"model_file":None,"demo_model":False}

@app.post("/api/encode")
async def encode(image: UploadFile = File(...)):
    raw=await image.read(MAX_UPLOAD+1)
    if len(raw)>MAX_UPLOAD:
        raise HTTPException(413,"Image upload exceeds 8 MB")
    if not raw:
        raise HTTPException(400,"Empty image upload")
    try:
        with Image.open(BytesIO(raw)) as im:
            if max(im.size)>MAX_DIMENSION or im.width*im.height>MAX_DIMENSION**2:
                raise HTTPException(413,"Demo image max size is 2048×2048")
            im.verify()
    except (UnidentifiedImageError,OSError,ValueError) as exc:
        raise HTTPException(400,"Not a valid image file") from exc
    p=checkpoint()
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode="wb",suffix=".img",delete=False) as tmp:
            name=tmp.name
            tmp.write(raw)
        text_bytes=encode_image(name,str(p),fmt="text")
        metadata,ids=decode_text(text_bytes.decode("utf-8"))
        binary_bytes=encode_binary(ids,metadata)
        reconstructed,_=decode_image(binary_bytes,str(p))
        original=load_rgb(name)
        quality=metrics(original,reconstructed)
        return {
            "text":text_bytes.decode("utf-8"),
            "binary_base64":base64.b64encode(binary_bytes).decode("ascii"),
            "reconstructed_png_base64":b64_png(reconstructed),
            "tokens":len(ids),
            "text_bytes":len(text_bytes),
            "binary_bytes":len(binary_bytes),
            "original_dimensions":[original.width,original.height],
            "model_id":metadata["model_id"],
            "bpp":round(8*len(binary_bytes)/(original.width*original.height),5),
            "psnr":quality["psnr"],
            "ssim":quality["ssim"]
        }
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(400,str(exc)) from exc
    finally:
        if name:
            Path(name).unlink(missing_ok=True)

@app.post("/api/decode")
async def decode(upload: UploadFile | None = File(None), text: str | None = Form(None)):
    if upload is None and not text:
        raise HTTPException(400,"Upload a .emotxt/.emoc file or paste the full .emotxt JSON")
    raw=await upload.read(MAX_UPLOAD+1) if upload else text.encode("utf-8")
    if len(raw)>MAX_UPLOAD:
        raise HTTPException(413,"Encoded payload exceeds 8 MB")
    try:
        img,metadata=decode_image(raw,str(checkpoint()))
        return {"reconstructed_png_base64":b64_png(img),"tokens":metadata["token_count"],
                "width":img.width,"height":img.height,"model_id":metadata["model_id"]}
    except (ValueError,RuntimeError,UnicodeError) as exc:
        raise HTTPException(400,str(exc)) from exc

@app.get("/{filename}", include_in_schema=False)
def frontend_file(filename: str):
    if filename not in FRONTEND_FILES or not (FRONTEND / filename).is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(FRONTEND / filename)


@app.get("/food-match/{filename}", include_in_schema=False)
def food_match_file(filename: str):
    """Serve only explicit Food Match browser resources from this repository."""
    directory = FRONTEND / "food-match"
    if filename not in FOOD_MATCH_FILES or not (directory / filename).is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(directory / filename)
