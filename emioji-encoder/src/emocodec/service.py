"""Reusable encode/decode functions shared by CLI and future APIs."""
from __future__ import annotations
from pathlib import Path

def encode_image(image_path: str, checkpoint: str, fmt: str = "text", device: str = "cpu") -> bytes:
    from emocodec.training.trainer import load_checkpoint
    from emocodec.utils.images import image_to_tensor, load_rgb, pad_tensor
    from emocodec.codec.formats import encode_binary, encode_text, make_metadata
    image = load_rgb(image_path)
    model, model_id = load_checkpoint(checkpoint, device=device)
    tensor = pad_tensor(image_to_tensor(image)).to(device)
    indices = model.encode_indices(tensor).cpu()
    latent_h,latent_w = indices.shape[-2:]
    ids = indices.reshape(-1).tolist()
    meta = make_metadata(ids,width=image.width,height=image.height,latent_w=latent_w,latent_h=latent_h,model_id=model_id)
    if fmt == "text":
        return encode_text(ids,meta).encode("utf-8")
    if fmt == "binary":
        return encode_binary(ids,meta)
    raise ValueError("Format must be text or binary")

def decode_image(data: bytes, checkpoint: str, device: str = "cpu"):
    import torch
    from emocodec.codec.formats import deserialize
    from emocodec.training.trainer import load_checkpoint
    from emocodec.utils.images import tensor_to_image
    model, model_id = load_checkpoint(checkpoint, device=device)
    meta, ids = deserialize(data,expected_model_id=model_id)
    latent = torch.tensor(ids, dtype=torch.long,device=device).reshape(1,meta["latent_h"],meta["latent_w"])
    reconstructed = model.decode_indices(latent)
    return tensor_to_image(reconstructed,meta["width"],meta["height"]),meta

def benchmark_image(image_path: str, checkpoint: str, device: str = "cpu") -> dict:
    import io
    from emocodec.codec.formats import deserialize
    from emocodec.utils.images import load_rgb,metrics
    original=load_rgb(image_path)
    text=encode_image(image_path, checkpoint, fmt="text", device=device)
    binary=encode_image(image_path, checkpoint, fmt="binary", device=device)
    reconstruction,meta=decode_image(binary,checkpoint,device=device)
    quality=metrics(original,reconstruction)
    reference={}
    for codec,options in (("JPEG",{"format":"JPEG","quality":75}),("WEBP",{"format":"WEBP","quality":75})):
        buf=io.BytesIO()
        original.save(buf,**options)
        size=len(buf.getvalue())
        buf.seek(0)
        from PIL import Image
        with Image.open(buf) as im:
            recon=im.convert("RGB")
        reference[codec.lower()]={"bytes":size,"bpp":8*size/(original.width*original.height),**metrics(original,recon)}
    return {"image":str(image_path),"width":original.width,"height":original.height,
            "tokens":meta["token_count"],"model_id":meta["model_id"],
            "emoji_text_bytes":len(text),"binary_bytes":len(binary),
            "binary_bpp":8*len(binary)/(original.width*original.height),
            **quality,"baselines_quality75_not_matched_bpp":reference}
