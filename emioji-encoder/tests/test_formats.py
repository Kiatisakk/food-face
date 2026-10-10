import unittest
from emocodec.codec.formats import (
    BIN_MAGIC, decode_binary, decode_text, encode_binary, encode_text,
    make_metadata, deserialize
)
from emocodec.codec.mapping import ALPHABET

MODEL_ID="a"*64
class CodecFormatTests(unittest.TestCase):
    def setUp(self):
        self.ids=list(range(16*16))
        self.meta=make_metadata(self.ids,width=256,height=256,latent_w=16,latent_h=16,model_id=MODEL_ID)
    def test_exact_text_round_trip(self):
        text=encode_text(self.ids,self.meta)
        self.assertIn(ALPHABET[42],text)
        self.assertEqual(decode_text(text),(self.meta,self.ids))
    def test_exact_binary_round_trip(self):
        data=encode_binary(self.ids,self.meta)
        self.assertTrue(data.startswith(BIN_MAGIC))
        self.assertEqual(decode_binary(data),(self.meta,self.ids))
        self.assertEqual(deserialize(data),(self.meta,self.ids))
    def test_unicode_bytes_larger(self):
        self.assertGreater(len(encode_text(self.ids,self.meta).encode("utf-8")),len(encode_binary(self.ids,self.meta)))
    def test_corrupted_payload_rejected(self):
        data=bytearray(encode_binary(self.ids,self.meta))
        data[-1]^=1
        with self.assertRaisesRegex(ValueError,"checksum"): decode_binary(bytes(data))
        with self.assertRaises(ValueError): decode_binary(bytes(data[:-5]))
    def test_wrong_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,"Wrong model"): decode_binary(encode_binary(self.ids,self.meta),expected_model_id="b"*64)
    def test_bad_alphabet_rejected(self):
        broken=dict(self.meta,alphabet_sha256="f"*64)
        with self.assertRaises(ValueError): encode_binary(self.ids,broken)
    def test_unknown_emoji_rejected(self):
        text=encode_text(self.ids,self.meta).replace(ALPHABET[42],"☂",1)
        with self.assertRaisesRegex(ValueError,"Unknown"): decode_text(text)
    def test_malformed_metadata_rejected(self):
        import json
        obj=json.loads(encode_text(self.ids,self.meta))
        obj["width"]=1024
        with self.assertRaises(ValueError): decode_text(json.dumps(obj))
    def test_nonsquare_dims(self):
        ids=[5]*(15*20)
        meta=make_metadata(ids,width=320,height=240,latent_w=20,latent_h=15,model_id=MODEL_ID)
        self.assertEqual(decode_text(encode_text(ids,meta))[1],ids)
    def test_binary_header_length_rejected(self):
        with self.assertRaises(ValueError): decode_binary(BIN_MAGIC + b"\\xff\\xff\\xff\\xff")
if __name__=="__main__":
    unittest.main()
