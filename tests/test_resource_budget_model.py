import pytest

from wan_dit_lab.offloading.resource_budget_model import BackingReplay
from wan_dit_lab.offloading.resource_cost import ResourceCosts


def costs(*names):
    return ResourceCosts({name: {"disk_restore": [2.], "disk_write": [1.]} for name in names})


def test_replay_only_counts_actual_required_write_and_restore():
    events = [dict(event="save", resource="a", bytes=4),
              dict(event="save", resource="b", bytes=4),
              dict(event="restore", resource="a"),
              dict(event="restore", resource="a")]
    model = BackingReplay(events, costs("a", "b"))
    ample = model.run(8)
    assert ample["read_bytes"] == ample["write_bytes"] == 0
    limited = model.run(4)
    assert limited["write_bytes"] == 4
    assert limited["read_bytes"] == 8
    assert limited["serialized_read_s"] == 4
    assert limited["peak_bytes"] <= 4


def test_same_key_replacement_invalidates_old_file_version():
    model = BackingReplay([dict(event="save", resource="a", bytes=4),
                          dict(event="save", resource="a", bytes=4),
                          dict(event="restore", resource="a")], costs("a"))
    result = model.run(0)
    assert result["write_bytes"] == 8
    assert result["read_bytes"] == 4


def test_replay_releases_expired_history():
    events = [dict(event="save", resource="history/recent/" + name * 32, bytes=4)
              for name in ("a", "b", "c")]
    model = BackingReplay(events, costs("history/recent/item"), recent_limit=1)
    result = model.run(8)
    assert result["write_bytes"] == 0
    assert result["peak_bytes"] == 8


def test_unknown_lifetime_is_an_error():
    model = BackingReplay([dict(event="restore", resource="a")], costs("a"))
    with pytest.raises(ValueError, match="Missing lifetime"):
        model.run(4)
