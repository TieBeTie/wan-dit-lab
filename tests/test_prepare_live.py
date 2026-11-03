"""Cold weight conversion must release its process before export starts."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.conversion import prepare_live


def test_dense_conversion_runs_in_a_child_before_export(monkeypatch, tmp_path):
    recipe = json.loads((prepare_live.ROOT / "configs/experiments/director-original/point-a-fp8-kv.proposed.json").read_text())
    path = tmp_path / "config.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(prepare_live, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare_live.py", "--config", str(path)])
    inline = Mock(side_effect=AssertionError("conversion must not retain Torch in the supervisor"))
    monkeypatch.setitem(sys.modules, "wan_dit_lab.quantization.weight_pack", SimpleNamespace(prepare_dense_weights=inline))
    child = Mock()
    monkeypatch.setattr(prepare_live.subprocess, "run", child)
    monkeypatch.setattr(prepare_live, "export_identity", lambda e: "identity")
    monkeypatch.setattr(prepare_live, "prepare", lambda *a: False)
    prepare_live.main()
    inline.assert_not_called()
    assert "--dense-weights-only" in child.call_args.args[0]
    assert child.call_args.kwargs["check"] is True
