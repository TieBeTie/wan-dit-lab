"""Quality of the Rust host's videos against Live's Python path, on the same prompts and noise, beyond the distance of
the latents: metrics of each video on its own, and of the pair.

    python scripts/conversion/live_reference.py --render DIR      # video_<prompt>_<seed>.safetensors
    live render --reference DIR                                    # rust_<prompt>_<seed>.bin
    python scripts/conversion/live_quality.py DIR

Of each video (sampled frames):
    clip       CLIP ViT-B/32 (LAION-2B) cosine of a frame to the prompt, x100: does it show the prompt
    consist    CLIP cosine of a frame to the frame 4 before: the subject and background stay (VBench's
               subject/background consistency, with CLIP for DINO)
    warp       mean |frame t - frame t-1 warped to t| over pixels the flow sees both ways (Farneback flow, 240x432):
               temporal stability against the motion
    flicker    mean |frame t - frame t-1|
    sharp      variance of the Laplacian of the grey frame: detail (imaging quality, no reference)
Of the pair: LPIPS (AlexNet) and SSIM of the Rust frame to the Python frame.

Writes a table per video, the means over videos by chunk (does the frame fall apart with time), and a sheet of
frames of both paths side by side (quality_sheet.png)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open

DEVICE = "cuda"
SIZE = (432, 240)  # width, height of the frames the flow, LPIPS and SSIM see


def frames_of(directory, name):
    with safe_open(str(directory / f"video_{name}.safetensors"), "pt") as f:
        meta = f.metadata()
        python = f.get_tensor("frames").numpy()
    rust = np.fromfile(directory / f"rust_{name}.bin", dtype=np.uint8).reshape(-1, *python.shape[1:])
    assert rust.shape == python.shape, (name, rust.shape, python.shape)
    return meta["prompt"], python, rust


class Clip:
    def __init__(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model = self.model.to(DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def images(self, frames):
        from PIL import Image
        batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(DEVICE)
        e = self.model.encode_image(batch).float()
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text(self, prompt):
        e = self.model.encode_text(self.tokenizer([prompt]).to(DEVICE)).float()
        return e / e.norm(dim=-1, keepdim=True)


def warp_error(a, b):
    """|b - a warped to b| over the pixels whose flow agrees both ways, a and b uint8 RGB at SIZE."""
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b))
    forward = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # b's pixel -> where in a
    backward = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = ga.shape
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
    to = grid + forward
    warped = cv2.remap(a, to[..., 0], to[..., 1], cv2.INTER_LINEAR)
    back = cv2.remap(backward, to[..., 0], to[..., 1], cv2.INTER_LINEAR)
    seen = np.linalg.norm(forward + back, axis=-1) < 0.01 * np.linalg.norm(forward, axis=-1) ** 2 + 0.5
    return float(np.abs(warped.astype(np.float32) - b.astype(np.float32))[seen].mean())


def video_metrics(frames, clip, text, step=4):
    small = [cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA) for f in frames]
    picked = list(range(0, len(frames), step))
    e = clip.images([frames[i] for i in picked])
    rows = []
    for k, i in enumerate(picked):
        row = dict(frame=i, clip=float(100 * e[k] @ text[0]), consist=float(e[k] @ e[k - 1]) if k else np.nan,
                   sharp=float(cv2.Laplacian(cv2.cvtColor(small[i], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
        if i:
            row["warp"] = warp_error(small[i - 1], small[i])
            row["flicker"] = float(np.abs(small[i].astype(np.float32) - small[i - 1].astype(np.float32)).mean())
        rows.append(row)
    return rows


def pair_metrics(python, rust, lpips_model, step=8):
    from skimage.metrics import structural_similarity
    rows = []
    for i in range(0, len(python), step):
        a, b = (cv2.resize(x[i], SIZE, interpolation=cv2.INTER_AREA) for x in (python, rust))
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE)
        with torch.no_grad():
            lp = float(lpips_model(t(a), t(b)))
        ssim = structural_similarity(a, b, channel_axis=2, data_range=255)
        rows.append(dict(frame=i, lpips=lp, ssim=ssim))
    return rows


def chunk_of(frame):
    return 0 if frame < 29 else 1 + (frame - 29) // 32


def mean(rows, key, where=lambda r: True):
    values = [r[key] for r in rows if key in r and where(r) and not np.isnan(r[key])]
    return float(np.mean(values)) if values else float("nan")


def main():
    import lpips
    directory = Path(sys.argv[1])
    names = sorted(p.stem.removeprefix("video_") for p in directory.glob("video_*.safetensors")
                   if (directory / f"rust_{p.stem.removeprefix('video_')}.bin").exists())
    clip, lpips_model = Clip(), lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    keys = ["clip", "consist", "warp", "flicker", "sharp"]
    per_video, by_chunk = [], {"python": [], "rust": [], "pair": []}
    sheet = []
    for name in names:
        prompt, python, rust = frames_of(directory, name)
        text = clip.text(prompt)
        m = {path: video_metrics(f, clip, text) for path, f in (("python", python), ("rust", rust))}
        pair = pair_metrics(python, rust, lpips_model)
        per_video.append((name, {p: {k: mean(m[p], k) for k in keys} for p in m}, mean(pair, "lpips"), mean(pair, "ssim")))
        for p in m:
            by_chunk[p] += m[p]
        by_chunk["pair"] += pair
        if len(sheet) < 3:
            chunks = (0, 4, 8, 11)
            picks = [min(len(python) - 1, 29 + 32 * (c - 1) + 16 if c else 14) for c in chunks]
            row = lambda f: np.concatenate([cv2.resize(f[i], (288, 160)) for i in picks], axis=1)
            sheet.append(np.concatenate([row(python), row(rust)], axis=0))
    print("| video | path | " + " | ".join(keys) + " | LPIPS to Python | SSIM to Python |")
    print("|---|---|" + "---:|" * (len(keys) + 2))
    for name, values, lp, ss in per_video:
        for p in ("python", "rust"):
            extra = f"{lp:.3f} | {ss:.3f}" if p == "rust" else "n/a | n/a"
            print(f"| {name} | {p} | " + " | ".join(f"{values[p][k]:.3f}" for k in keys) + f" | {extra} |")
    print()
    chunks = sorted({chunk_of(r["frame"]) for r in by_chunk["python"]})
    print("| chunk | clip Python | clip Rust | warp Python | warp Rust | sharp Python | sharp Rust | LPIPS | SSIM |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in chunks:
        at = lambda r: chunk_of(r["frame"]) == c
        print(f"| {c} | {mean(by_chunk['python'], 'clip', at):.2f} | {mean(by_chunk['rust'], 'clip', at):.2f} | "
              f"{mean(by_chunk['python'], 'warp', at):.2f} | {mean(by_chunk['rust'], 'warp', at):.2f} | "
              f"{mean(by_chunk['python'], 'sharp', at):.0f} | {mean(by_chunk['rust'], 'sharp', at):.0f} | "
              f"{mean(by_chunk['pair'], 'lpips', at):.3f} | {mean(by_chunk['pair'], 'ssim', at):.3f} |")
    print()
    print("| all videos | " + " | ".join(keys) + " |")
    print("|---|" + "---:|" * len(keys))
    for p in ("python", "rust"):
        print(f"| {p} | " + " | ".join(f"{mean(by_chunk[p], k):.3f}" for k in keys) + " |")
    if sheet:
        out = directory / "quality_sheet.png"
        cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(sheet, axis=0), cv2.COLOR_RGB2BGR))
        print(f"\n{out}: per video, Python above Rust, chunks {chunks[:1] + [4, 8, 11]}")


if __name__ == "__main__":
    main()
