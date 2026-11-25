"""ncu_report: the per-kernel table of an `ncu --page raw --csv` export (docs/report-2026-11-16-bottlenecks.md)."""

import pytest

from wan_dit_lab.harness import ncu_report as nr

CSV = '''"ID","Kernel Name","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed","launch__registers_per_thread"
"","","nsecond","%","register/thread"
"0","qk_int_sv_f8_attn_kernel","9,284,000","88.5","168"
"1","invoke_kernel","1,707,000","71.0","255"
'''


def test_rows_are_kernels_with_numbers_read_past_thousands_separators():
    rows = nr.parse(CSV)
    assert [r["Kernel Name"] for r in rows] == ["qk_int_sv_f8_attn_kernel", "invoke_kernel"]
    assert rows[0]["gpu__time_duration.sum"] == 9284000.0
    assert rows[1]["launch__registers_per_thread"] == 255.0


def test_the_units_row_is_not_a_kernel():
    assert len(nr.parse(CSV)) == 2


def test_a_missing_metric_is_refused_by_name():
    with pytest.raises(KeyError, match="no_such_metric"):
        nr.metric(nr.parse(CSV)[0], "no_such_metric")


def test_kernels_are_grouped_by_the_part_of_the_block():
    rows = [{"Kernel Name": "qk_int_sv_f8_attn_kernel", "gpu__time_duration.sum": 9.0},
            {"Kernel Name": "_sum_max", "gpu__time_duration.sum": 1.0},
            {"Kernel Name": "_quant_k", "gpu__time_duration.sum": 0.5}]
    groups = nr.group(rows)
    assert groups["self-attention: Sage kernel"][0]["gpu__time_duration.sum"] == 9.0
    assert len(groups["self-attention: Sage preparation"]) == 2
