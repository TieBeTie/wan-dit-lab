"""rope_search: the pieces of the gradient-free search over the RoPE profile (scripts/experiments/rope_cma.py)."""

import json

import pytest

from wan_dit_lab.experiments import rope_search as rs

SPACE = [{"name": "height", "low": 0.9, "high": 1.4}, {"name": "width", "low": 0.9, "high": 1.4},
         {"name": "ntk_pi_mix", "low": 0.0, "high": 1.0}]


def test_a_vector_becomes_a_sweep_point_inside_the_bounds_on_the_tag_grid():
    p = rs.point([1.23456, 0.5, 1.7], SPACE)
    assert p == {"height": 1.235, "width": 0.9, "ntk_pi_mix": 1.0}
    with pytest.raises(ValueError, match="3 numbers"):
        rs.point([1.0], SPACE)


def test_the_tag_is_the_folder_wan_exe_writes_for_the_point():
    # crates/wan/src/rope.rs point_tag
    assert rs.tag({"height": 1.1, "width": 1.0, "ntk_pi_mix": 1.0}) == "h1100-w1000-mix1000"
    assert rs.tag({"height": 1.2, "width": 1.2}) == "h1200-w1200"
    assert rs.tag({"height": 1.05, "width": 1.25, "ntk_pi_mix": 0.25}) == "h1050-w1250-mix250"


def test_the_generation_config_sweeps_the_points_for_one_seed_and_keeps_the_rest():
    base = [{"component": "export", "rope": {"ntk_aware": True}}, {"component": "generate", "seed": 0,
             "output": {"directory": "x", "crf": 16}}]
    points = [{"height": 1.1, "width": 1.0, "ntk_pi_mix": 1.0}]
    c = rs.generation_config(base, points, seed=2, directory="D:/out/g3-s2")
    assert c[1]["rope_sweep"] == points and c[1]["seed"] == 2
    assert c[1]["output"] == {"directory": "D:/out/g3-s2", "crf": 16}
    assert base[1]["seed"] == 0 and "rope_sweep" not in base[1], "the base stays as it was"


def test_the_first_chunk_loss_is_lpips_after_alignment_plus_one_minus_dino():
    row = {"run": "teacup_pour__h1100-w1000-mix1000", "lpips_aligned_first": 0.2, "dino_reference_first": 0.9}
    assert rs.first_chunk_loss(row) == pytest.approx(0.3)


def test_point_losses_average_the_chains_named_and_refuse_a_missing_one():
    rows = [{"run": f"{c}__{t}", "lpips_aligned_first": l, "dino_reference_first": 1.0}
            for c, t, l in [("a", "h1000-w1000", 0.2), ("b", "h1000-w1000", 0.4), ("c", "h1000-w1000", 9.0),
                            ("a", "h1100-w1100", 0.1), ("b", "h1100-w1100", 0.3)]]
    assert rs.point_losses(rows, ["a", "b"]) == pytest.approx({"h1000-w1000": 0.3, "h1100-w1100": 0.2})
    with pytest.raises(KeyError, match="c__h1100-w1100"):
        rs.point_losses(rows, ["a", "c"])


def test_the_evaluation_folders_link_each_point_and_chain_to_its_reference(tmp_path):
    run, ref = tmp_path / "run", tmp_path / "ref"
    for tag in ("h1000-w1000", "h1100-w1100"):
        (run / tag).mkdir(parents=True)
        for chain in ("a", "b"):
            (run / tag / f"{chain}.mp4").write_bytes(f"{tag}{chain}".encode())
    ref.mkdir()
    for chain in ("a", "b"):
        (ref / f"{chain}.mp4").write_bytes(chain.encode())
    experiment, reference = rs.evaluation_layout(run, ref, ["a", "b"], ["h1000-w1000", "h1100-w1100"], tmp_path / "eval")
    runs = json.loads((experiment / "experiment.json").read_text())["runs"]
    assert {r["id"] for r in runs} == {"a__h1000-w1000", "b__h1000-w1000", "a__h1100-w1100", "b__h1100-w1100"}
    assert all(r["prompts"] == [r["id"].split("__")[0]] for r in runs)
    assert (experiment / "b__h1100-w1100" / "video.mp4").read_bytes() == b"h1100-w1100b"
    assert json.loads((reference / "experiment.json").read_text())["runs"] == [
        {"id": "a", "prompts": ["a"]}, {"id": "b", "prompts": ["b"]}]
    assert (reference / "a" / "video.mp4").read_bytes() == b"a"


def test_the_log_replays_every_point_told_so_a_search_resumes(tmp_path):
    log = tmp_path / "log.jsonl"
    rs.append_generation(log, 0, [[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]], [0.3, 0.2], {"seeds": [0, 1]})
    rs.append_generation(log, 1, [[1.2, 1.1, 0.5]], [0.25], {"seeds": [0, 1]})
    assert rs.replay(log) == [([[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]], [0.3, 0.2]), ([[1.2, 1.1, 0.5]], [0.25])]
    assert rs.replay(tmp_path / "none.jsonl") == []
