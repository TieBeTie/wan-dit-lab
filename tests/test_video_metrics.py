"""Comparison of a video with a reference that frames the scene closer or further (the original 704x1280 is zoomed in
against 512x928 runs): the zoom and shift that make them match, and the metrics after that alignment."""

import cv2
import numpy as np
import pytest

from wan_dit_lab.harness.video_metrics import align, aligned_pair, cosine_by_time, crop


def scene(frames=6, height=180, width=320, seed=0):
    """A smooth random video: blurred noise that drifts a little from frame to frame."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (height + 2 * frames, width + 2 * frames, 3)).astype(np.float32), (0, 0), 6)
    base = (base - base.min()) / (base.max() - base.min()) * 255
    return np.stack([base[2 * t:2 * t + height, 2 * t:2 * t + width] for t in range(frames)]).astype(np.uint8)


def zoomed(video, zoom, dx=0.0, dy=0.0):
    """Every frame cropped by `zoom` at the offset (dx, dy) in [-1, 1] of the slack, resized back to its size."""
    h, w = video.shape[1:3]
    return np.stack([cv2.resize(crop(f, zoom, dx, dy), (w, h), interpolation=cv2.INTER_AREA) for f in video])


def test_identical_videos_align_at_no_zoom_and_no_shift():
    v = scene()
    a = align(v, v)
    assert a["zoom"] == pytest.approx(1.0)
    assert (a["dx"], a["dy"]) == (0.0, 0.0)


def test_a_closer_reference_is_matched_by_cropping_the_video():
    v = scene()
    ref = zoomed(v, 1.25)  # the reference shows the scene larger
    a = align(v, ref)
    assert a["zoom"] == pytest.approx(1.25, rel=0.04)
    assert abs(a["dx"]) <= 0.25 and abs(a["dy"]) <= 0.25


def test_a_wider_reference_is_matched_by_cropping_the_reference():
    ref = scene()
    v = zoomed(ref, 1.25)  # the video shows the scene larger
    a = align(v, ref)
    assert a["zoom"] == pytest.approx(1 / 1.25, rel=0.04)


def test_an_off_centre_crop_is_found_with_its_side():
    v = scene()
    ref = zoomed(v, 1.3, dx=1.0, dy=-1.0)  # right and up part of the video
    a = align(v, ref)
    assert a["zoom"] == pytest.approx(1.3, rel=0.04)
    assert a["dx"] > 0.5 and a["dy"] < -0.5


def test_aligned_pair_has_one_size_and_matching_content():
    v = scene()
    ref = zoomed(v, 1.25)
    a, b = aligned_pair(v, ref, align(v, ref), size=(160, 288))
    assert a.shape == b.shape == (len(v), 160, 288, 3)
    before = np.abs(cv2.resize(v[0], (288, 160)).astype(float) - b[0].astype(float)).mean()
    after = np.abs(a.astype(float) - b.astype(float)).mean()
    assert after < before / 3, (before, after)


def test_videos_of_different_lengths_align_on_their_common_part():
    v = scene(frames=8)
    ref = zoomed(v[:5], 1.2)
    assert align(v, ref)["zoom"] == pytest.approx(1.2, rel=0.04)
    a, b = aligned_pair(v, ref, align(v, ref), size=(90, 160))
    assert len(a) == len(b) == 5


def test_cosine_by_time_compares_frames_at_the_same_time_only():
    x = np.eye(4)
    assert cosine_by_time(x, x) == pytest.approx(1.0)
    assert cosine_by_time(x, x[::-1]) == pytest.approx(0.0)
    assert cosine_by_time(x, x[:2]) == pytest.approx(1.0)


def test_region_takes_the_box_given_as_fractions_of_the_frame():
    from wan_dit_lab.harness.video_metrics import region
    video = np.arange(2 * 10 * 20 * 3, dtype=np.uint8).reshape(2, 10, 20, 3)
    part = region(video, (0.2, 0.25, 0.8, 0.75))  # y0, x0, y1, x1
    assert part.shape == (2, 6, 10, 3)
    assert np.array_equal(part, video[:, 2:8, 5:15])
    with pytest.raises(ValueError):
        region(video, (0.5, 0.2, 0.4, 0.9))
