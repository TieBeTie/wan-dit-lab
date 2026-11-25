import numpy as np

from wan_dit_lab.harness.metric_check import (aligned_flicker, energy, inside, loss_pair, pairs, subject_consistency,
                                              summary)


def test_the_loss_of_a_pair_is_lpips_plus_one_minus_dino_for_the_first_chunk_and_the_clip():
    row = {"lpips_first": 0.30, "dino_first": 0.90, "lpips_all": 0.40, "dino_all": 0.80}
    first, whole = loss_pair(row)
    assert abs(first - 0.40) < 1e-12 and abs(whole - 0.60) < 1e-12


def test_pairs_within_one_group_are_unordered_and_across_groups_paired_or_not():
    seeds = [0, 1, 2]
    assert pairs(seeds, seeds, "within") == [(0, 1), (0, 2), (1, 2)]
    assert pairs(seeds, seeds, "paired") == [(0, 0), (1, 1), (2, 2)]
    assert pairs(seeds, seeds, "unpaired") == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def test_energy_distance_is_zero_when_the_point_is_another_draw_of_the_reference():
    # two videos of one distribution: the distance across equals the distance within
    assert energy(0.7, 0.7, 0.7) == 0.0
    assert abs(energy(0.8, 0.7, 0.5) - 0.2) < 1e-12


def test_subject_consistency_is_one_for_a_still_subject_and_falls_when_it_turns():
    still = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    assert abs(subject_consistency(still) - 1.0) < 1e-12
    angles = np.linspace(0, np.pi / 2, 5)
    turning = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert subject_consistency(turning) < 0.95


def test_aligned_flicker_does_not_count_a_pure_shift_of_the_frame():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, (40, 60))
    shifted = np.stack([base[:, 2 + t:42 + t] for t in range(3)])  # the camera pans one pixel per frame
    assert aligned_flicker(shifted, max_shift=3) < 1e-9
    noisy = shifted + rng.normal(0, 10, shifted.shape)
    assert aligned_flicker(noisy, max_shift=3) > 5


def test_summary_and_whether_a_value_is_inside_the_reference_spread():
    s = summary([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and abs(s["std"] - 1.0) < 1e-12 and s["min"] == 1.0 and s["max"] == 3.0
    assert inside(2.5, [1.0, 2.0, 3.0]) and not inside(3.5, [1.0, 2.0, 3.0])


def test_tables_average_the_distances_of_each_scene_and_give_the_energy_distance():
    from wan_dit_lab.harness.metric_check import SCENES
    from wan_dit_lab.harness.metric_tables import POINTS, distance_tables
    row = lambda v: {"first": v, "all": v}
    scene = {"rr": [row(0.6), row(0.8)]}
    for p in POINTS:
        scene |= {f"oo_{p}": [row(0.5)], f"ro_unpaired_{p}": [row(0.9), row(0.7)], f"ro_paired_{p}": [row(0.4)]}
    _, per_point = distance_tables({s: scene for s in SCENES}, "first")
    r = per_point[POINTS[0]][0]
    assert abs(r["rr"] - 0.7) < 1e-12 and abs(r["ro"] - 0.8) < 1e-12
    assert abs(r["D"] - (0.8 - (0.7 + 0.5) / 2)) < 1e-12 and abs(r["paired"] - 0.4) < 1e-12
