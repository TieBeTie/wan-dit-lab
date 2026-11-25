"""evaluate_videos.py end to end on tiny videos: a reference that frames the scene closer than the run, at another size."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_video_metrics import scene, zoomed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def write(folder, run_id, video):
    (folder / run_id).mkdir(parents=True)
    imageio.mimwrite(folder / run_id / "video.mp4", list(video), fps=24, quality=9, macro_block_size=8)


def test_a_closer_reference_at_another_size_gets_its_zoom_and_aligned_metrics(tmp_path):
    run = scene(frames=32, height=144, width=256)
    ref = zoomed(scene(frames=32, height=144, width=256), 1.25)
    ref = np.stack([cv2.resize(f, (320, 176), interpolation=cv2.INTER_AREA) for f in ref])  # another size
    experiment, reference = tmp_path / "experiment", tmp_path / "reference"
    write(experiment, "cup__candidate", run)
    write(experiment, "cup__other", scene(frames=32, height=144, width=256, seed=7))
    write(reference, "cup", ref)
    for folder, ids in ((experiment, ["cup__candidate", "cup__other"]), (reference, ["cup"])):
        (folder / "experiment.json").write_text(json.dumps({"runs": [{"id": i, "prompts": ["a cup"]} for i in ids]}))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "reports" / "evaluate_videos.py"), "--experiment", str(experiment),
                    "--reference", str(reference), "--no-dino"], check=True, cwd=ROOT)
    rows = {r["run"]: r for r in json.loads((experiment / "evaluation.json").read_text())}
    same, other = rows["cup__candidate"], rows["cup__other"]
    assert abs(same["zoom"] - 1.25) < 0.05
    assert "psnr" not in same  # the sizes differ: only the aligned frame metrics
    assert same["lpips_aligned"] < other["lpips_aligned"] / 2, (same, other)
    assert same["psnr_aligned"] > other["psnr_aligned"] + 5
    assert same["lpips_aligned_first"] <= same["lpips_aligned"] + 0.05


def test_a_box_gets_its_own_aligned_lpips(tmp_path):
    # --crop y0,x0,y1,x1 (fractions of the aligned frame): lpips_aligned_crop beside the whole frame's, of the box only
    run = scene(frames=32, height=144, width=256)
    broken = run.copy()
    broken[:, 36:108, 64:192] = 255 - broken[:, 36:108, 64:192]  # the inside of the box differs, the rest is the same
    experiment, reference = tmp_path / "experiment", tmp_path / "reference"
    write(experiment, "cup__same", run)
    write(experiment, "cup__broken", broken)
    write(reference, "cup", run)
    for folder, ids in ((experiment, ["cup__same", "cup__broken"]), (reference, ["cup"])):
        (folder / "experiment.json").write_text(json.dumps({"runs": [{"id": i, "prompts": ["a cup"]} for i in ids]}))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "reports" / "evaluate_videos.py"), "--experiment", str(experiment),
                    "--reference", str(reference), "--no-dino", "--crop", "0.25,0.25,0.75,0.75"], check=True, cwd=ROOT)
    rows = {r["run"]: r for r in json.loads((experiment / "evaluation.json").read_text())}
    same, broken = rows["cup__same"], rows["cup__broken"]
    assert same["lpips_aligned_crop"] < 0.05
    assert broken["lpips_aligned_crop"] > broken["lpips_aligned"], broken


def test_an_evaluator_kept_in_one_process_gives_the_rows_of_the_command_and_moves_its_models(tmp_path):
    # a search scores many generations: the models load once (Evaluator), go to RAM while wan.exe holds the card
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_videos", ROOT / "scripts" / "reports" / "evaluate_videos.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    run = scene(frames=32, height=144, width=256)
    experiment, reference = tmp_path / "experiment", tmp_path / "reference"
    write(experiment, "cup__a", run)
    write(experiment, "cup__b", scene(frames=32, height=144, width=256, seed=7))
    write(reference, "cup", run)
    for folder, ids in ((experiment, ["cup__a", "cup__b"]), (reference, ["cup"])):
        (folder / "experiment.json").write_text(json.dumps({"runs": [{"id": i, "prompts": ["a cup"]} for i in ids]}))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "reports" / "evaluate_videos.py"), "--experiment", str(experiment),
                    "--reference", str(reference), "--no-dino", "--name", "cli"], check=True, cwd=ROOT)
    command = json.loads((experiment / "evaluation_cli.json").read_text())
    evaluator = ev.Evaluator(device="cpu", dino=False, every=4)
    evaluator.offload()
    assert all(p.device.type == "cpu" for p in evaluator.parameters())
    evaluator.to("cpu")
    rows = evaluator.evaluate(experiment, reference, name="kept", sheets=False)
    assert json.loads((experiment / "evaluation_kept.json").read_text()) == rows
    assert [r["run"] for r in rows] == [r["run"] for r in command]
    for mine, theirs in zip(rows, command):
        for key, value in theirs.items():
            assert mine[key] == value if not isinstance(value, float) else abs(mine[key] - value) < 1e-6, key
    assert not (experiment / "sheet_cup.jpg").exists() or (experiment / "sheet_cup.jpg").stat().st_size > 0
