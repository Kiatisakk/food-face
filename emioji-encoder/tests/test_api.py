import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from fastapi.testclient import TestClient
    from PIL import Image
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "Web or ML dependencies missing")
class APITests(unittest.TestCase):
    def test_combined_frontend_and_assets(self):
        from apps.api.main import app, FRONTEND
        if not FRONTEND.is_dir():
            self.skipTest("Food Face repository root is not present")
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="welcome"', response.text)
        self.assertLess(response.text.index('id="food-kitchen"'), response.text.index('id="emoji-codec"'))
        for name in ("style.css", "codec.css", "menu.js", "renderer.js", "app.js", "codec.js", "assets/shef.png"):
            response = client.get(f"/{name}")
            self.assertEqual(response.status_code, 200, name)
            self.assertTrue(response.content, name)
        for path in ("/README.md", "/package.json", "/.git/config", "/missing.js"):
            self.assertEqual(client.get(path).status_code, 404, path)

    def test_standalone_frontend_fallback(self):
        from apps.api.main import app
        with tempfile.TemporaryDirectory() as directory:
            with patch("apps.api.main.FRONTEND", Path(directory) / "missing"):
                response = TestClient(app).get("/")
                self.assertEqual(response.status_code, 200)
                self.assertIn("EmoCodec — Images to emojis and back", response.text)

    def test_checkpoint_selection(self):
        from apps.api.main import checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            demo = checkpoints / "demo.pt"
            trained = checkpoints / "photos_v1.pt"
            demo.touch()
            with patch("apps.api.main.ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(checkpoint(), demo)
                trained.touch()
                self.assertEqual(checkpoint(), trained)
                with patch.dict(os.environ, {"EMOCODEC_CHECKPOINT": str(demo)}):
                    self.assertEqual(checkpoint(), demo)

    def test_status_and_end_to_end(self):
        from apps.api.main import app
        from emocodec.models.vqvae import Architecture,VQVAE
        from emocodec.codec.mapping import ALPHABET_SHA256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=VQVAE(Architecture(hidden=8,embedding_dim=16))
            p=root/"sample.pt"
            torch.save({"state_dict":model.state_dict(),"architecture":model.config_dict(),
                        "alphabet_sha256":ALPHABET_SHA256},p)
            im=root/"sample.png"
            Image.new("RGB",(45,37),(15,140,210)).save(im)
            old=os.environ.get("EMOCODEC_CHECKPOINT")
            os.environ["EMOCODEC_CHECKPOINT"]=str(p)
            try:
                client=TestClient(app)
                self.assertEqual(client.get("/").status_code,200)
                self.assertIn("EMOCODEC",client.get("/").text)
                self.assertTrue(client.get("/api/status").json()["ready"])
                with im.open("rb") as f:
                    response=client.post("/api/encode", files={"image":("sample.png",f,"image/png")})
                self.assertEqual(response.status_code,200,response.text)
                payload=response.json()
                self.assertEqual(payload["tokens"],9)
                self.assertGreater(payload["text_bytes"],payload["binary_bytes"])
                response=client.post("/api/decode",data={"text":payload["text"]})
                self.assertEqual(response.status_code,200,response.text)
                image_bytes=base64.b64decode(response.json()["reconstructed_png_base64"])
                from io import BytesIO
                with Image.open(BytesIO(image_bytes)) as decoded:
                    self.assertEqual(decoded.size,(45,37))
                response=client.post("/api/decode",files={"upload":("data.emoc",base64.b64decode(payload["binary_base64"]),"application/octet-stream")})
                self.assertEqual(response.status_code,200,response.text)
            finally:
                if old is None: os.environ.pop("EMOCODEC_CHECKPOINT",None)
                else: os.environ["EMOCODEC_CHECKPOINT"]=old

if __name__=="__main__":
    unittest.main()
