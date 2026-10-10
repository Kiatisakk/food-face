"""Ensure the standalone Colab architecture and file formats match desktop EmoCodec."""
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "EmoCodec_Train_Colab.ipynb"

class NotebookTests(unittest.TestCase):
    def test_all_code_cells_compile(self):
        nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"],4)
        self.assertGreaterEqual(len(nb["cells"]),20)
        for idx,cell in enumerate(nb["cells"]):
            if cell["cell_type"]=="code":
                compile("".join(cell["source"]),f"notebook_cell_{idx}","exec")

    def test_notebook_checkpoint_and_codec_compatibility(self):
        try:
            import torch
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("Requires torch, numpy and Pillow")
        from emocodec.codec.formats import deserialize
        from emocodec.service import decode_image
        from emocodec.codec.mapping import ALPHABET_SHA256
        nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        codes={idx:"".join(c["source"]) for idx,c in enumerate(nb["cells"]) if c["cell_type"]=="code"}
        ns={"__name__":"__main__","json":json,"torch":torch,
            "hashlib":hashlib,"struct":struct,"np":np,"Image":Image}
        exec(compile(codes[8],"<model_definition>","exec"),ns)
        exec(compile(codes[10],"<alphabet>","exec"),ns)
        self.assertEqual(ns["ALPHABET_SHA256"],ALPHABET_SHA256)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config=ns["Architecture"](hidden=8,embedding_dim=16)
            model=ns["VQVAE"](config).eval()
            weights_path=root/"best.pt"
            torch.save({"state_dict":model.state_dict(),"architecture":model.config_dict(),
                        "alphabet_sha256":ALPHABET_SHA256},weights_path)
            ns.update({"BEST_PATH":weights_path,"IMAGE_SIZE":32,"DEVICE":torch.device("cpu"),
                       "images":torch.rand(2,3,32,32),"model":model,
                       "ARTIFACT_DIR":root})
            exec(compile(codes[18],"<codec_output>","exec"),ns)
            text_bytes=(root/"sample.emotxt").read_bytes()
            binary_bytes=(root/"sample.emoc").read_bytes()
            meta_a, ids_a=deserialize(text_bytes)
            meta_b, ids_b=deserialize(binary_bytes)
            self.assertEqual(ids_a,ids_b)
            self.assertEqual(meta_a,meta_b)
            self.assertEqual(meta_a["model_id"],hashlib.sha256(weights_path.read_bytes()).hexdigest())
            a,_=decode_image(text_bytes,str(weights_path))
            b,_=decode_image(binary_bytes,str(weights_path))
            self.assertEqual(a.size,(32,32))
            self.assertEqual(a.tobytes(),b.tobytes())
            self.assertEqual(a.tobytes(),Image.open(root/"sample_decoded.png").tobytes())

if __name__=="__main__":
    unittest.main()
