"""A video against a reference that frames the same scene closer or further: the zoom and shift that make them match
(searched on a few grey frames by normalized cross-correlation), and the pair of videos cut and resized by them, so
that frame metrics (LPIPS, PSNR) compare the same part of the scene.

Zoom z >= 1: the video is cut to 1/z of its height and width (the reference shows the scene larger).
Zoom z < 1: the reference is cut to z of its sides (the video shows the scene larger).
dx, dy in [-1, 1]: where the cut window sits in the slack, -1 left or top, 0 centre, +1 right or bottom.
"""

import cv2
import numpy as np


def crop(frame, zoom, dx=0.0, dy=0.0):
    """The window of 1/zoom of the sides of `frame` (zoom >= 1) at the offset (dx, dy) of the slack."""
    h, w = frame.shape[:2]
    ch, cw = max(1, round(h / zoom)), max(1, round(w / zoom))
    top, left = round((dy + 1) / 2 * (h - ch)), round((dx + 1) / 2 * (w - cw))
    return frame[top:top + ch, left:left + cw]


def _cut(video, ref, zoom, dx, dy):
    """(video frames, reference frames) after the cut of `zoom`, each still at its own size."""
    if zoom >= 1:
        return [crop(f, zoom, dx, dy) for f in video], list(ref)
    return list(video), [crop(f, 1 / zoom, dx, dy) for f in ref]


def _resized(frames, size):
    h, w = size
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def _normalized(grey):
    g = grey.reshape(len(grey), -1).astype(np.float64)
    g -= g.mean(axis=1, keepdims=True)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def _grey(frames):
    return np.stack([cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in frames])


def align(video, ref, frames=8, width=96, max_zoom=1.45):
    """{"zoom", "dx", "dy", "score"}: the cut that best matches `video` to `ref` over `frames` frames spread over
    their common length; score is the mean normalized cross-correlation of the grey frames (1: identical)."""
    n = min(len(video), len(ref))
    picks = np.unique(np.linspace(0, n - 1, min(frames, n)).round().astype(int))
    rh, rw = ref.shape[1:3]
    small = (max(8, round(width * rh / rw)), width)
    # a first reduction to 4x the comparison size keeps the many cuts cheap
    pre = lambda v: _resized(v[picks], (4 * small[0], 4 * small[1]))
    vid, rf = _grey(pre(video)), _grey(pre(ref))

    def score(z, dx, dy):
        a, b = _cut(vid, rf, z, dx, dy)
        return float((_normalized(_resized(a, small)) * _normalized(_resized(b, small))).sum(axis=1).mean())

    def search(zooms, shifts):
        best = None
        for z in zooms:
            for dx, dy in ([(0.0, 0.0)] if abs(z - 1) < 1e-9 else [(x, y) for x in shifts for y in shifts]):
                s = score(z, dx, dy)
                if best is None or s > best[0] + 1e-12:
                    best = (s, float(z), float(dx), float(dy))
        return best

    zooms = np.unique(np.concatenate([np.geomspace(1 / max_zoom, max_zoom, 33), [1.0]]))
    s, z, dx, dy = search(zooms, np.linspace(-1, 1, 5))
    step = max_zoom ** (2 / 32)
    if abs(z - 1) > 1e-9:  # a finer look around the coarse best
        fine = np.geomspace(z / step, z * step, 9)
        fine = fine[(fine >= 1) == (z >= 1)]
        shifts = np.unique(np.clip(np.concatenate([np.linspace(dx - 0.25, dx + 0.25, 3), np.linspace(dy - 0.25, dy + 0.25, 3)]), -1, 1))
        better = search(fine, shifts)
        if better[0] > s:
            s, z, dx, dy = better
    return {"zoom": z, "dx": dx, "dy": dy, "score": s}


def region(video, box):
    """The frames of `video` [frames, height, width, 3] inside `box` = (y0, x0, y1, x1), fractions of the frame."""
    y0, x0, y1, x1 = box
    if not (0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1):
        raise ValueError(f"box {box}: 0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1")
    h, w = video.shape[1:3]
    return video[:, round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]


def aligned_pair(video, ref, alignment, size):
    """(video, reference) over their common length, cut by `alignment` and resized to `size` (height, width)."""
    n = min(len(video), len(ref))
    a, b = _cut(video[:n], ref[:n], alignment["zoom"], alignment["dx"], alignment["dy"])
    return _resized(a, size), _resized(b, size)


def cosine_by_time(a, b):
    """Mean cosine between rows of `a` and `b` at the same index, over their common length."""
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((a * b).sum(axis=1).mean())
