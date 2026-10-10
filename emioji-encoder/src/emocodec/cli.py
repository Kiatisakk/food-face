"""Command-line entry point: python -m emocodec --help."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(prog="emocodec",description="Emoji-based VQ-VAE image codec (requires a trained checkpoint)")
    commands=p.add_subparsers(dest="command",required=True)
    commands.add_parser("mapping-check",help="Verify static Emoji ID round-trip without torch")
    tr=commands.add_parser("train",help="Train VQ-VAE from real images in a folder")
    tr.add_argument("--dataset",required=True)
    tr.add_argument("--output",default="checkpoints/emocodec_v1.pt")
    tr.add_argument("--epochs",type=int,default=10)
    tr.add_argument("--batch-size",type=int,default=8)
    tr.add_argument("--size",type=int,default=256)
    tr.add_argument("--hidden",type=int,default=48)
    tr.add_argument("--embedding-dim",type=int,default=64)
    tr.add_argument("--learning-rate",type=float,default=2e-4)
    tr.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    tr.add_argument("--workers",type=int,default=0)
    en=commands.add_parser("encode",help="Encode an image to .emotxt or .emoc")
    en.add_argument("image")
    en.add_argument("--checkpoint",required=True)
    en.add_argument("--output",required=True)
    en.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
    de=commands.add_parser("decode",help="Decode .emotxt/.emoc to reconstructed PNG")
    de.add_argument("codec_file")
    de.add_argument("--checkpoint",required=True)
    de.add_argument("--output",required=True)
    de.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
    ins=commands.add_parser("inspect",help="Validate and inspect a codec file without torch")
    ins.add_argument("codec_file")
    be=commands.add_parser("benchmark",help="Basic rate/distortion report and unmatched quality-75 baselines")
    be.add_argument("image")
    be.add_argument("--checkpoint",required=True)
    be.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
    be.add_argument("--output")
    dem=commands.add_parser("generate-demo-data",help="Create tiny synthetic images for a smoke test (NOT quality training)")
    dem.add_argument("--output",default="datasets/demo")
    dem.add_argument("--count",type=int,default=24)
    args=p.parse_args(argv)
    if args.command=="mapping-check":
        from emocodec.codec.mapping import ALPHABET, ALPHABET_SHA256, from_emoji, to_emoji
        assert from_emoji(to_emoji(range(256))) == list(range(256))
        print(f"PASS: {len(ALPHABET)} unique emoji IDs, alphabet_sha256={ALPHABET_SHA256}")
    elif args.command=="train":
        from emocodec.training.trainer import train
        train(dataset=args.dataset,output=args.output,epochs=args.epochs,batch_size=args.batch_size,
              size=args.size,hidden=args.hidden,embedding_dim=args.embedding_dim,
              learning_rate=args.learning_rate,device=args.device,workers=args.workers)
    elif args.command=="generate-demo-data":
        from emocodec.utils.demo_data import generate
        generate(args.output,args.count)
    elif args.command=="encode":
        from emocodec.service import encode_image
        fmt="binary" if Path(args.output).suffix.lower()==".emoc" else "text" if Path(args.output).suffix.lower()==".emotxt" else None
        if fmt is None:
            p.error("Output must end in .emoc or .emotxt")
        encoded=encode_image(args.image,args.checkpoint,fmt=fmt,device=args.device)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_bytes(encoded)
        print(f"Encoded {args.image} → {args.output}, {len(encoded)} actual bytes")
    elif args.command=="decode":
        from emocodec.service import decode_image
        image,metadata=decode_image(Path(args.codec_file).read_bytes(),args.checkpoint,device=args.device)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        image.save(args.output,"PNG")
        print(f"Decoded {metadata['token_count']} tokens → {args.output} ({image.width}×{image.height})")
    elif args.command=="inspect":
        from emocodec.codec.formats import deserialize
        metadata,ids=deserialize(Path(args.codec_file).read_bytes())
        print(json.dumps({**metadata,"emoji_tokens":len(ids),"actual_bytes":Path(args.codec_file).stat().st_size},indent=2))
    elif args.command=="benchmark":
        from emocodec.service import benchmark_image
        report=benchmark_image(args.image,args.checkpoint,device=args.device)
        s=json.dumps(report,indent=2,allow_nan=False) if report["psnr"]!=float("inf") else json.dumps({**report,"psnr":"infinity"},indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(s+"\n",encoding="utf-8")
        print(s)

if __name__=="__main__":
    main()
