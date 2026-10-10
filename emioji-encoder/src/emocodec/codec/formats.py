"""Strict, versioned text/binary files with payload integrity checking."""
from __future__ import annotations
import hashlib
import json
import struct
from .mapping import ALPHABET_SHA256, from_emoji, to_emoji

TEXT_MAGIC = "EMOTXT1"
BIN_MAGIC = b"EMOC\x01"
MAX_HEADER = 8192
MAX_PIXELS = 8192 * 8192
MAX_TOKENS = 1024 * 1024
REQUIRED = {"version", "model_id", "alphabet_sha256", "width", "height", "latent_w", "latent_h", "stride", "token_count", "payload_sha256"}

def _digest(ids: list[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()

def make_metadata(ids: list[int], *, width: int, height: int, latent_w: int, latent_h: int, model_id: str, stride: int = 16) -> dict:
    meta = dict(version=1, model_id=model_id, alphabet_sha256=ALPHABET_SHA256,
                width=width, height=height, latent_w=latent_w, latent_h=latent_h,
                stride=stride, token_count=len(ids), payload_sha256=_digest(ids))
    validate(meta, ids)
    return meta

def validate(meta: dict, ids: list[int], *, expected_model_id: str | None = None) -> None:
    if not isinstance(meta, dict) or not REQUIRED.issubset(meta):
        raise ValueError("Invalid/missing codec metadata fields")
    if type(meta["version"]) is not int or meta["version"] != 1 or meta["alphabet_sha256"] != ALPHABET_SHA256:
        raise ValueError("Unsupported format/alphabet version")
    model_id = meta["model_id"]
    if not isinstance(model_id, str) or len(model_id) != 64 or any(ch not in "0123456789abcdef" for ch in model_id):
        raise ValueError("Invalid model_id (SHA-256 expected)")
    if expected_model_id is not None and model_id != expected_model_id:
        raise ValueError("Wrong model checkpoint: image cannot be decoded with a different model")
    for key in ("width", "height", "latent_w", "latent_h", "stride", "token_count"):
        if type(meta[key]) is not int or meta[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if meta["width"] * meta["height"] > MAX_PIXELS or max(meta["width"], meta["height"]) > 8192:
        raise ValueError("Image dimensions exceed safety limits")
    if meta["stride"] != 16:
        raise ValueError("Unsupported encoder stride")
    if meta["latent_w"] != (meta["width"] + 15)//16 or meta["latent_h"] != (meta["height"] + 15)//16:
        raise ValueError("Latent grid does not match image dimensions")
    if meta["token_count"] != meta["latent_w"] * meta["latent_h"] or meta["token_count"] > MAX_TOKENS:
        raise ValueError("Invalid token count or excessive allocation")
    if len(ids) != meta["token_count"] or any(not 0 <= int(x) <= 255 for x in ids):
        raise ValueError("Token length/range mismatch")
    if meta["payload_sha256"] != _digest(ids):
        raise ValueError("Token payload checksum mismatch")

def encode_text(ids: list[int], meta: dict) -> str:
    validate(meta, ids)
    return json.dumps({**meta, "magic": TEXT_MAGIC, "emoji": to_emoji(ids)}, ensure_ascii=False, separators=(",", ":")) + "\n"

def decode_text(content: str, *, expected_model_id: str | None = None) -> tuple[dict, list[int]]:
    if len(content.encode("utf-8")) > 5_000_000:
        raise ValueError("Emoji file is too large")
    try:
        obj = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed emoji JSON") from exc
    if not isinstance(obj, dict) or obj.get("magic") != TEXT_MAGIC or not isinstance(obj.get("emoji"), str):
        raise ValueError("Invalid text codec magic or emoji payload")
    ids = from_emoji(obj["emoji"])
    meta = {k: v for k, v in obj.items() if k not in {"magic", "emoji"}}
    validate(meta, ids, expected_model_id=expected_model_id)
    return meta, ids

def encode_binary(ids: list[int], meta: dict) -> bytes:
    validate(meta, ids)
    header = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header) > MAX_HEADER:
        raise ValueError("Codec header too large")
    return BIN_MAGIC + struct.pack(">I", len(header)) + header + bytes(ids)

def decode_binary(data: bytes, *, expected_model_id: str | None = None) -> tuple[dict, list[int]]:
    if len(data) < 9 or not data.startswith(BIN_MAGIC):
        raise ValueError("Invalid binary codec magic")
    hlen = struct.unpack(">I", data[5:9])[0]
    if not 0 < hlen <= MAX_HEADER or len(data) < 9 + hlen:
        raise ValueError("Truncated/oversized binary header")
    try:
        meta = json.loads(data[9:9 + hlen].decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Malformed binary header") from exc
    ids = list(data[9 + hlen:])
    validate(meta, ids, expected_model_id=expected_model_id)
    return meta, ids

def deserialize(data: bytes, *, expected_model_id: str | None = None) -> tuple[dict, list[int]]:
    if data.startswith(BIN_MAGIC):
        return decode_binary(data, expected_model_id=expected_model_id)
    try:
        return decode_text(data.decode("utf-8"), expected_model_id=expected_model_id)
    except UnicodeDecodeError as exc:
        raise ValueError("Unknown codec file format") from exc
