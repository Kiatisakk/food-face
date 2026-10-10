import unittest
from emocodec.codec.mapping import ALPHABET, ALPHABET_SHA256, from_emoji, to_emoji

class MappingTests(unittest.TestCase):
    def test_256_unique_single_codepoint(self):
        self.assertEqual(len(ALPHABET), 256)
        self.assertEqual(len(set(ALPHABET)), 256)
        self.assertTrue(all(len(c)==1 and len(c.encode("utf-8"))==4 for c in ALPHABET))
        self.assertEqual(len(ALPHABET_SHA256),64)
    def test_all_ids_exact(self):
        ids=list(range(256)) + [0,255,42,42]
        self.assertEqual(from_emoji(to_emoji(ids)),ids)
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError): from_emoji("A")
        with self.assertRaises(ValueError): to_emoji([256])
        with self.assertRaises(ValueError): to_emoji([-1])

if __name__=="__main__":
    unittest.main()
