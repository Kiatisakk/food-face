import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"PyTorch not installed")
class NeuralTests(unittest.TestCase):
    def test_shapes_forward_backward(self):
        from emocodec.models.vqvae import VQVAE,Architecture
        from torch.nn import functional as F
        model=VQVAE(Architecture(hidden=8,embedding_dim=16))
        x=torch.rand(2,3,32,48)
        y,vq_loss,ids=model(x)
        self.assertEqual(tuple(y.shape),tuple(x.shape))
        self.assertEqual(tuple(ids.shape),(2,2,3))
        (F.mse_loss(y,x)+vq_loss).backward()
        self.assertIsNotNone(model.encoder.net[0].weight.grad)
        self.assertIsNotNone(model.vq.embedding.weight.grad)
        self.assertEqual(tuple(model.decode_indices(ids).shape),tuple(x.shape))
    def test_full_service_decode(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from emocodec.models.vqvae import VQVAE,Architecture
        from emocodec.codec.mapping import ALPHABET_SHA256
        from emocodec.service import encode_image,decode_image
        from emocodec.codec.formats import deserialize
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            model=VQVAE(Architecture(hidden=8,embedding_dim=16))
            ckpt=root/"model.pt"
            torch.save({"state_dict":model.state_dict(),"architecture":model.config_dict(),
                        "alphabet_sha256":ALPHABET_SHA256},ckpt)
            image=root/"test.png"
            Image.new("RGB",(47,33),(20,100,180)).save(image)
            for fmt in ("text","binary"):
                data=encode_image(str(image),str(ckpt),fmt)
                recon,metadata=decode_image(data,str(ckpt))
                self.assertEqual(recon.size,(47,33))
                self.assertEqual(metadata["token_count"],3*3)
            a=encode_image(str(image),str(ckpt),"text")
            b=encode_image(str(image),str(ckpt),"binary")
            self.assertEqual(deserialize(a)[1],deserialize(b)[1])
if __name__=="__main__":
    unittest.main()
