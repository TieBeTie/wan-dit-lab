"""Does the paired loss to the 704x1280 reference measure how plausible a video is? Measurement only, over finished
videos (no generation): the same loss L = LPIPS_aligned + (1 - DINO) between

- d_rr: reference videos of one scene with different seeds,
- d_oo: our videos of one scene with different seeds (544x960 window 32, 512x928),
- d_ro: reference against ours on different seeds (unpaired), and on the same seed (paired, the loss we search with),

and the energy distance D = d_ro - (d_rr + d_oo) / 2 (zero when our videos are just other draws of the reference's
distribution). Then reference-free metrics per video: CLIP score to the prompt, subject consistency of DINO between
frames, flicker after aligning neighbouring frames. MUSIQ is skipped: pyiqa and its weights are not on this machine.

    .venv\\Scripts\\python.exe -m wan_dit_lab.harness.metric_check [out.json]
"""

import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

FIRST = 29  # frames of the first chunk after decoding
EVERY = 4  # frames between those the embeddings see
SCENES = ("teacup_pour", "fox_runs", "astronaut_3_3_4_5", "two_foxes_snowy_cabin")
SEEDS = (0, 1, 2)
OUTPUTS = Path("D:/Experiments/DIT/outputs")
EVAL = OUTPUTS / "speed-2026-11-12" / "eval"
GROUPS = {  # group: (folder of eval, path of the video of a scene inside s<seed>/eval)
    "ref704": ("final-544x960", "reference/{scene}/video.mp4"),
    "544x960": ("final-544x960", "runs/{scene}__final-544x960/video.mp4"),
    "512x928": ("ladder-512x928", "runs/{scene}__ladder-512x928/video.mp4"),
}


def loss_pair(row):
    """(first chunk, whole clip) of L = LPIPS_aligned + 1 - DINO."""
    return row["lpips_first"] + 1.0 - row["dino_first"], row["lpips_all"] + 1.0 - row["dino_all"]


def pairs(a, b, kind):
    """Seed pairs: 'within' one group (unordered, i < j), 'paired' (same seed) or 'unpaired' (i != j) across two."""
    if kind == "within":
        return list(combinations(a, 2))
    if kind == "paired":
        return [(s, s) for s in a if s in b]
    if kind == "unpaired":
        return [(i, j) for i in a for j in b if i != j]
    raise ValueError(kind)


def energy(d_ro, d_rr, d_oo):
    """Energy distance from mean distances across and within the two groups."""
    return d_ro - (d_rr + d_oo) / 2


def subject_consistency(embeddings):
    """Mean over frames t >= 1 of the cosine to the first frame and to the frame before, halved (as VBench)."""
    e = np.asarray(embeddings, dtype=np.float64)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    first, previous = (e[1:] * e[0]).sum(1), (e[1:] * e[:-1]).sum(1)
    return float(((first + previous) / 2).mean())


def aligned_flicker(grey, max_shift=4):
    """Mean |frame_t - frame_t-1| over t, each pair at the integer shift (|dy|, |dx| <= max_shift) that matches them
    best, on their overlap: a pan of the camera is not counted, a change of the content is."""
    g = np.asarray(grey, dtype=np.float64)
    out = []
    for prev, cur in zip(g[:-1], g[1:]):
        h, w = cur.shape
        best = np.inf
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                a = cur[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
                b = prev[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
                best = min(best, float(np.abs(a - b).mean()))
        out.append(best)
    return float(np.mean(out))


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def inside(value, reference):
    return min(reference) <= value <= max(reference)


def video_path(group, seed, scene):
    folder, inner = GROUPS[group]
    return EVAL / folder / f"s{seed}" / "eval" / inner.format(scene=scene)


def prompts():
    run = next((OUTPUTS / "speed-2026-11-12" / "ladder" / "704x1280-s0").glob("*/run.json"))
    return {k: v[0]["text"] for k, v in json.loads(run.read_text(encoding="utf-8"))["chains"].items()}


class Videos:
    """Frames and embeddings of every video, read and embedded once."""

    def __init__(self, evaluator):
        self.ev, self.cache = evaluator, {}

    def get(self, group, seed, scene):
        key = (group, seed, scene)
        if key not in self.cache:
            from scripts.reports.evaluate_videos import frames
            video = frames(video_path(group, seed, scene))
            self.cache[key] = dict(video=video, dino=self.ev.embed_dino(video[::EVERY]),
                                   clip=self.ev.embed_images(video[::EVERY]))
        return self.cache[key]


def distance(videos, a, b):
    """The loss row of video a against video b (b plays the reference: the cut fits a to b)."""
    from wan_dit_lab.harness.video_metrics import align, aligned_pair, cosine_by_time
    va, vb = videos.get(*a), videos.get(*b)
    cut = align(va["video"], vb["video"])
    x, y = aligned_pair(va["video"], vb["video"], cut, size=(352, 640))
    k = -(-FIRST // EVERY)
    row = dict(lpips_first=videos.ev.lpips_mean(x[:FIRST], y[:FIRST]), lpips_all=videos.ev.lpips_mean(x[::EVERY], y[::EVERY]),
               dino_first=cosine_by_time(va["dino"][:k], vb["dino"][:k]), dino_all=cosine_by_time(va["dino"], vb["dino"]))
    row["first"], row["all"] = loss_pair(row)
    return row


def reference_free(videos, group, seed, scene, text):
    from PIL import Image
    v = videos.get(group, seed, scene)
    grey = np.stack([np.asarray(Image.fromarray(f).convert("L").resize((320, 176), Image.BILINEAR)) for f in v["video"]])
    return dict(clip_text=float((v["clip"] @ videos.ev.embed_text(text)).mean()),
                subject_consistency=subject_consistency(v["dino"]), flicker_aligned=aligned_flicker(grey))


def main(out):
    import torch
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from scripts.reports.evaluate_videos import Evaluator
    videos = Videos(Evaluator("cuda" if torch.cuda.is_available() else "cpu"))
    texts = prompts()
    result = {"distances": {}, "reference_free": {}}
    for scene in SCENES:
        d = result["distances"][scene] = {}
        d["rr"] = [distance(videos, ("ref704", i, scene), ("ref704", j, scene)) | {"seeds": [i, j]}
                   for i, j in pairs(SEEDS, SEEDS, "within")]
        for group in ("544x960", "512x928"):
            d[f"oo_{group}"] = [distance(videos, (group, i, scene), (group, j, scene)) | {"seeds": [i, j]}
                                for i, j in pairs(SEEDS, SEEDS, "within")]
            for kind in ("unpaired", "paired"):
                d[f"ro_{kind}_{group}"] = [distance(videos, (group, j, scene), ("ref704", i, scene)) | {"seeds": [i, j]}
                                           for i, j in pairs(SEEDS, SEEDS, kind)]
        for group in GROUPS:
            result["reference_free"].setdefault(group, {})[scene] = {
                str(s): reference_free(videos, group, s, scene, texts[scene]) for s in SEEDS}
        print(scene, "done", flush=True)
        videos.cache = {k: v for k, v in videos.cache.items() if k[2] != scene}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(OUTPUTS / "metric-check-2026-11-17" / "result.json"))
