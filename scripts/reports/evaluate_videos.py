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
At any size of either, after the zoom and shift that best match the video to the reference (the original 704x1280 frames
the scene closer than 512x928 runs; wan_dit_lab.harness.video_metrics.align):
- zoom, dx, dy, align_score: that cut (zoom > 1: the video is cut, < 1: the reference) and its grey correlation
- lpips_aligned, psnr_aligned and their _first over the first chunk: LPIPS and PSNR of the cut pair at 352x640
- lpips_aligned_crop (with --crop y0,x0,y1,x1): LPIPS of the cut pair inside that box, fractions of the frame: a
  detail the whole frame's numbers hardly see (the handle of a cup)
- dino_reference and dino_reference_first: mean cosine of DINOv2-small (CLS) between a frame and the reference's frame
  at the same time, uncut: where the objects are and what they are, little moved by the zoom (--no-dino skips it)
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


class Evaluator:
    """The models of the metrics (CLIP ViT-B-32, AlexNet LPIPS, DINOv2-small unless dino=False), loaded once on
    `device`: a search that scores many generations keeps one, moves it to RAM with offload() while wan.exe holds the
    card and back with to(device)."""

    def __init__(self, device, dino=True, every=4):
        self.device, self.every = device, every
        self.dino = None
        if dino:
            from transformers import AutoImageProcessor, AutoModel
            self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        import lpips
        self.perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        import open_clip
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k",
                                                                              device=device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip.eval()

    def modules(self):
        return [m for m in (self.dino, self.perceptual, self.clip) if m is not None]

    def parameters(self):
        return [p for m in self.modules() for p in m.parameters()]

    def to(self, device):
        self.device = device
        for m in self.modules():
            m.to(device)
        return self

    def offload(self):
        """The models to RAM, the card's memory back to the driver (torch's cache emptied)."""
        for m in self.modules():
            m.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def embed_dino(self, video):
        out = []
        for i in range(0, len(video), 16):
            batch = self.dino_processor(images=list(video[i:i + 16]), return_tensors="pt")["pixel_values"].to(self.device)
            out.append(self.dino(pixel_values=batch).pooler_output.float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def lpips_mean(self, a, b):
        to = lambda v: torch.from_numpy(v).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        return float(torch.cat([self.perceptual(to(a[i:i + 8]), to(b[i:i + 8])).flatten() for i in range(0, len(a), 8)]).mean())

    @torch.no_grad()
    def embed_images(self, video):
        batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in video]).to(self.device)
        return torch.nn.functional.normalize(self.clip.encode_image(batch).float(), dim=-1)

    @torch.no_grad()
    def embed_text(self, text):
        return torch.nn.functional.normalize(self.clip.encode_text(self.tokenizer([text]).to(self.device)).float(), dim=-1)[0]

    def evaluate(self, experiment_dir, reference_dir, name="", reference_suffix="", crop=None, sheets=True):
        """Rows of every run of the experiment against its reference; writes evaluation[_<name>].json (and the
        sheets) into the experiment folder, as the command does."""
        from wan_dit_lab.harness.video_metrics import align, aligned_pair, cosine_by_time, region
        experiment_dir, reference_dir = Path(experiment_dir), Path(reference_dir)
        box = tuple(float(v) for v in crop.split(",")) if crop else None
        every = self.every
        experiment = with_base(read_json(experiment_dir / "experiment.json"))
        reference = with_base(read_json(reference_dir / "experiment.json"))
        reference_runs = {r["id"]: r for r in reference["runs"]}
        results, by_prompt, cache = [], {}, {}
        for run in experiment["runs"]:
            video_path = experiment_dir / run["id"] / "video.mp4"
            if not video_path.exists():
                print(f"{run['id']}: no video", flush=True)
                continue
            key = run["id"].split("__")[0]
            video = frames(video_path)
            ref_id = key + reference_suffix
            if key not in cache and ref_id in reference_runs and (reference_dir / ref_id / "video.mp4").exists():
                ref = frames(reference_dir / ref_id / "video.mp4")
                cache[key] = (ref, self.embed_images(ref[::every]), self.embed_dino(ref[::every]) if self.dino else None)
            picked = self.embed_images(video[::every])
            text = self.embed_text(prompt_of(run))
            diffs = np.abs(np.diff(video.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))  # diffs[t-1]: frame t vs t-1
            starts = [t for t in chunk_starts(len(video)) if t < len(video)]
            inside = np.delete(diffs, [t - 1 for t in starts])
            row = dict(run=run["id"], prompt=key, frames=len(video), size=list(video.shape[1:3]),
                       clip_text=float((picked @ text).mean()),
                       seam=float(diffs[[t - 1 for t in starts]].mean() / inside.mean()) if starts else None,
                       flicker=float(diffs.mean()), sharpness=sharpness(video[::every]))
            if key in cache:
                ref_embed = cache[key][1]
                n = min(len(ref_embed), len(picked))
                row["clip_reference"] = float((picked[:n] * ref_embed[:n]).sum(-1).mean())
                ref = cache[key][0]
                if ref.shape[1:] == video.shape[1:]:
                    n, first = min(len(ref), len(video)), 29  # the first chunk decodes to 29 frames
                    row.update(psnr=psnr(video[:n], ref[:n]), psnr_first=psnr(video[:first], ref[:first]),
                               lpips=self.lpips_mean(video[:n:every], ref[:n:every]),
                               lpips_first=self.lpips_mean(video[:first], ref[:first]))
                first = 29  # the first chunk decodes to 29 frames
                cut = align(video, ref)
                a, b = aligned_pair(video, ref, cut, size=(352, 640))
                row.update(zoom=cut["zoom"], dx=cut["dx"], dy=cut["dy"], align_score=cut["score"],
                           lpips_aligned=self.lpips_mean(a[::every], b[::every]),
                           lpips_aligned_first=self.lpips_mean(a[:first], b[:first]),
                           psnr_aligned=psnr(a, b), psnr_aligned_first=psnr(a[:first], b[:first]))
                if box is not None:
                    row["lpips_aligned_crop"] = self.lpips_mean(region(a, box)[::every], region(b, box)[::every])
                if self.dino:
                    mine, theirs = self.embed_dino(video[::every]), cache[key][2]
                    row.update(dino_reference=cosine_by_time(mine, theirs),
                               dino_reference_first=cosine_by_time(mine[:-(-first // every)], theirs[:-(-first // every)]))
            results.append(row)
            by_prompt.setdefault(key, []).append((run["id"], video))
            print(json.dumps(row), flush=True)
        file = f"evaluation_{name}.json" if name else "evaluation.json"
        (experiment_dir / file).write_text(json.dumps(results, indent=1), encoding="utf-8")
        if sheets:
            for key, runs in by_prompt.items():
                rows = ([cache[key][0][:min(len(v) for _, v in runs)]] if key in cache else []) + [v for _, v in runs]
                labels = (["reference " + str(cache[key][0].shape[1:3])] if key in cache else []) + [r for r, _ in runs]
                sheet(rows, labels, experiment_dir / f"sheet_{key}.jpg")
        return json.loads(json.dumps(results))


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    return float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-9))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=located, required=True)
    parser.add_argument("--reference", type=located, required=True)
    parser.add_argument("--every", type=int, default=4, help="frames between those CLIP sees")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-suffix", default="", help="after the prompt in the id of its reference run")
    parser.add_argument("--name", default="", help="evaluation_<name>.json")
    parser.add_argument("--no-dino", action="store_true", help="skip dino_reference")
    parser.add_argument("--crop", default=None,
                        help="y0,x0,y1,x1: a box in fractions of the aligned frame, its LPIPS as lpips_aligned_crop")
    args = parser.parse_args()
    Evaluator(args.device, dino=not args.no_dino, every=args.every).evaluate(
        args.experiment, args.reference, name=args.name, reference_suffix=args.reference_suffix, crop=args.crop)


if __name__ == "__main__":
    main()
