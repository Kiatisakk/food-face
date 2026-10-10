"""Deterministic geometric images strictly for pipeline smoke tests, not benchmark claims."""
from __future__ import annotations
from pathlib import Path
import random
from PIL import Image, ImageDraw

def generate(output: str, count: int=24):
    if not 1 <= count <= 10000:
        raise ValueError("Count must be between 1 and 10000")
    root=Path(output)
    root.mkdir(parents=True,exist_ok=True)
    rng=random.Random(2026)
    for i in range(count):
        bg=tuple(rng.randrange(20,230) for _ in range(3))
        im=Image.new("RGB",(256,256),bg)
        draw=ImageDraw.Draw(im)
        for j in range(7):
            x=rng.randrange(0,200)
            y=rng.randrange(0,200)
            color=tuple(rng.randrange(255) for _ in range(3))
            if j%2: draw.rectangle((x,y,x+rng.randrange(10,56),y+rng.randrange(10,56)),fill=color)
            else: draw.ellipse((x,y,x+50,y+50),fill=color)
        im.save(root/f"demo_{i:03}.png")
    print(f"Generated {count} synthetic demo samples at {root}")
