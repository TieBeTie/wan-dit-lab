"""Metrics and frame sheets of the videos of an experiment (run_experiment.py) against reference videos of the same
prompts, at any resolution of either.

    .venv\\Scripts\\python.exe scripts\\evaluate_videos.py --experiment outputs\\experiments\\<date>\\<experiment>
        --reference outputs\\experiments\\2026-10-21\\01_Эталон_LongLive2_S2_BF16_704x1280

A run's prompt is the part of its id before "__" (or the whole id); the reference run of a prompt has that id, with
--reference-suffix after it (e.g. "__wbf16__asdpa__kvbf16": a cell of the same experiment as the reference).
Per run, over every `every`-th frame:
- clip_text: mean cosine of CLIP (ViT-B-32, LAION-2B) between the frames and the prompt (CLIP reads 77 tokens of it)
- clip_reference: mean cosine of CLIP between a frame and the reference's frame at the same time: how much of the
  reference's content it keeps, whatever the resolution
- seam: mean |frame_t - frame_t-1| at the first frame of a chunk over the mean inside chunks (1: no seams)
- flicker: mean |frame_t - frame_t-1| of the pixels, 0..255 (also how much moves)
- sharpness: variance of the Laplacian of the grey frame, mean over the frames CLIP sees (blur lowers it)
and when the reference has the same size: psnr and lpips (AlexNet) to it, over the whole video and over its first
chunk (psnr_first, lpips_first: before the trajectories part, the error of the computation itself).
Writes evaluation[_<name>].json and sheet_<prompt>.jpg (a row per run, the reference first) into the experiment
folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.experiments.run_experiment import read_json, with_base  # noqa: E402
from wan_dit_lab.harness.config import located  # noqa: E402


def frames(path):
    return np.stack(imageio.mimread(path, memtest=False))  # [T, H, W, 3] uint8


def prompt_of(run):
    """The text of a run's first prompt, from its prompts or prompt_refs."""
    if run.get("prompts"):
        return run["prompts"][0]
    path, _, key = run["prompt_refs"][0].partition("#")
    return {p["id"]: p["text"] for p in read_json(ROOT / path)["prompts"]}[key]


def sharpness(video):
    """Variance of the 4-neighbour Laplacian of the grey frames, mean over them."""
    g = video.astype(np.float32).mean(axis=-1)
    lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    return float(lap.var(axis=(1, 2)).mean())


def chunk_starts(count, first=29, chunk=32):
    """Frames that begin a chunk: the first chunk decodes to `first` frames, every later one to `chunk`."""
    return list(range(first, count, chunk))


def sheet(rows, labels, path, height=180):
    """A row of frames at 0, 1/4, 1/2, 3/4 and the end of the shortest video, per row."""
    n = min(len(r) for r in rows)
    picks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    tiles = []
    for video, label in zip(rows, labels):
        images = [Image.fromarray(video[i]) for i in picks]
        images = [im.resize((round(im.width * height / im.height), height)) for im in images]
        width = sum(im.width for im in images)
        row = Image.new("RGB", (max(width, 900), height + 22), "black")
        x = 0
        for im in images:
            row.paste(im, (x, 22))
            x += im.width
        ImageDraw.Draw(row).text((6, 4), f"{label}   frames {picks}", fill="white")
        tiles.append(row)
    out = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)))
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.height
    out.save(path, quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=located, required=True)
    parser.add_argument("--reference", type=located, required=True)
    parser.add_argument("--every", type=int, default=4, help="frames between those CLIP sees")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-suffix", default="", help="after the prompt in the id of its reference run")
    parser.add_argument("--name", default="", help="evaluation_<name>.json")
    args = parser.parse_args()
    import lpips
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()

    @torch.no_grad()
    def lpips_mean(a, b):
        to = lambda v: torch.from_numpy(v).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(args.device)
        return float(torch.cat([perceptual(to(a[i:i + 8]), to(b[i:i + 8])).flatten() for i in range(0, len(a), 8)]).mean())

    def psnr(a, b):
        mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
        return float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-9))))
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k",
                                                                 device=args.device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    @torch.no_grad()
    def embed_images(video):
        batch = torch.stack([preprocess(Image.fromarray(f)) for f in video]).to(args.device)
        return torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1)

    @torch.no_grad()
    def embed_text(text):
        return torch.nn.functional.normalize(model.encode_text(tokenizer([text]).to(args.device)).float(), dim=-1)[0]

    experiment = with_base(read_json(args.experiment / "experiment.json"))
    reference = with_base(read_json(args.reference / "experiment.json"))
    reference_runs = {r["id"]: r for r in reference["runs"]}
    results, by_prompt, cache = [], {}, {}
    for run in experiment["runs"]:
        video_path = args.experiment / run["id"] / "video.mp4"
        if not video_path.exists():
            print(f"{run['id']}: no video", flush=True)
            continue
        key = run["id"].split("__")[0]
        video = frames(video_path)
        ref_id = key + args.reference_suffix
        if key not in cache and ref_id in reference_runs and (args.reference / ref_id / "video.mp4").exists():
            ref = frames(args.reference / ref_id / "video.mp4")
            cache[key] = (ref, embed_images(ref[::args.every]))
        picked = embed_images(video[::args.every])
        text = embed_text(prompt_of(run))
        diffs = np.abs(np.diff(video.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))  # diffs[t-1]: frame t vs t-1
        starts = [t for t in chunk_starts(len(video)) if t < len(video)]
        inside = np.delete(diffs, [t - 1 for t in starts])
        row = dict(run=run["id"], prompt=key, frames=len(video), size=list(video.shape[1:3]),
                   clip_text=float((picked @ text).mean()),
                   seam=float(diffs[[t - 1 for t in starts]].mean() / inside.mean()) if starts else None,
                   flicker=float(diffs.mean()), sharpness=sharpness(video[::args.every]))
        if key in cache:
            ref_embed = cache[key][1]
            n = min(len(ref_embed), len(picked))
            row["clip_reference"] = float((picked[:n] * ref_embed[:n]).sum(-1).mean())
            ref = cache[key][0]
            if ref.shape[1:] == video.shape[1:]:
                n, first = min(len(ref), len(video)), 29  # the first chunk decodes to 29 frames
                row.update(psnr=psnr(video[:n], ref[:n]), psnr_first=psnr(video[:first], ref[:first]),
                           lpips=lpips_mean(video[:n:args.every], ref[:n:args.every]),
                           lpips_first=lpips_mean(video[:first], ref[:first]))
        results.append(row)
        by_prompt.setdefault(key, []).append((run["id"], video))
        print(json.dumps(row), flush=True)
    name = f"evaluation_{args.name}.json" if args.name else "evaluation.json"
    (args.experiment / name).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for key, runs in by_prompt.items():
        rows = ([cache[key][0][:min(len(v) for _, v in runs)]] if key in cache else []) + [v for _, v in runs]
        labels = (["reference " + str(cache[key][0].shape[1:3])] if key in cache else []) + [r for r, _ in runs]
        sheet(rows, labels, args.experiment / f"sheet_{key}.jpg")


if __name__ == "__main__":
    main()
