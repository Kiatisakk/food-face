"""Stable 256-symbol alphabet. Emoji are arbitrary IDs, not semantic labels."""
from __future__ import annotations
import hashlib
import json
from importlib import resources

def alphabet() -> tuple[str, ...]:
    data = json.loads(resources.files("emocodec").joinpath("assets/emoji_alphabet_v1.json").read_text(encoding="utf-8"))
    symbols = tuple(data["emoji"])
    if data["version"] != 1 or len(symbols) != 256 or len(set(symbols)) != 256:
        raise ValueError("Invalid emoji alphabet version/size/uniqueness")
    if any(len(ch) != 1 or len(ch.encode("utf-8")) != 4 for ch in symbols):
        raise ValueError("Alphabet must use 256 single 4-byte Unicode code points")
    return symbols

ALPHABET = alphabet()
REVERSE = {emoji: idx for idx, emoji in enumerate(ALPHABET)}
ALPHABET_SHA256 = hashlib.sha256("".join(ALPHABET).encode("utf-8")).hexdigest()

def to_emoji(ids) -> str:
    output = []
    for idx in ids:
        idx = int(idx)
        if not 0 <= idx < len(ALPHABET):
            raise ValueError(f"Invalid token ID: {idx}")
        output.append(ALPHABET[idx])
    return "".join(output)

def from_emoji(text: str) -> list[int]:
    result = []
    for pos, ch in enumerate(text):
        if ch not in REVERSE:
            raise ValueError(f"Unknown or modified emoji at token position {pos}: {ch!r}")
        result.append(REVERSE[ch])
    return result
