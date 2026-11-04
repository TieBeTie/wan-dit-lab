"""Reference-free quality metrics for a pair of generated videos (left = reference, right = candidate).

Per video: CLIP-T (frame to its active prompt), CLIP temporal consistency, Laplacian sharpness,
seam ratio (frame change at chunk boundaries / inside chunks). Per pair: CLIP-I of candidate frames
to reference frames. Both videos are resized to the same size before pixel metrics.

    python scripts/quality_metrics.py --run run.json --chain fox_cabin --left a.mp4 --right b.mp4
"""
import argparse
import json

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

SAMPLE = 12          # every 12th frame for CLIP and sharpness
SIZE = (928, 512)    # common size for pixel metrics
CHUNK = 32           # frames per latent chunk after decoding (8 latent frames x 4)
FIRST = 29           # first chunk boundary: 8 latent frames -> 29 decoded frames


def read(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA))
    return frames


def video_metrics(frames, scenes, fps, model, preprocess, text):
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [float(cv2.absdiff(a, b).mean()) for a, b in zip(gray, gray[1:])]
    seams = set(range(FIRST, len(frames), CHUNK))
    seam = np.mean([diffs[i - 1] for i in seams if i - 1 < len(diffs)])
    inner = np.mean([d for i, d in enumerate(diffs, 1) if i not in seams])
    idx = list(range(0, len(frames), SAMPLE))
    images = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))) for i in idx])
    with torch.no_grad():
        emb = torch.cat([model.encode_image(images[i:i + 16]).float() for i in range(0, len(images), 16)])
    emb = torch.nn.functional.normalize(emb, dim=-1)
    active = [max(k for k, s in enumerate(scenes) if s["begins_s"] <= i / fps) for i in idx]
    return emb, {
        "frames": len(frames),
        "clip_text": (emb * text[active]).sum(-1).mean().item(),
        "clip_temporal": (emb[1:] * emb[:-1]).sum(-1).mean().item(),
        "sharpness": float(np.mean([cv2.Laplacian(gray[i], cv2.CV_64F).var() for i in idx])),
        "seam_ratio": float(seam / inner),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--chain", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--fps", type=float, default=24.0)
    a = p.parse_args()
    scenes = json.load(open(a.run, encoding="utf-8"))["chains"][a.chain]
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        text = torch.nn.functional.normalize(model.encode_text(tok([s["text"] for s in scenes])).float(), dim=-1)
    left, right = read(a.left), read(a.right)
    n = min(len(left), len(right))
    el, ml = video_metrics(left[:n], scenes, a.fps, model, preprocess, text)
    er, mr = video_metrics(right[:n], scenes, a.fps, model, preprocess, text)
    mr["clip_to_left"] = (el * er).sum(-1).mean().item()
    print(json.dumps({"chain": a.chain, "left": ml, "right": mr}, indent=2))


if __name__ == "__main__":
    main()
