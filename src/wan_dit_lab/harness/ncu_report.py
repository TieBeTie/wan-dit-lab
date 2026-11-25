r"""Per-kernel table of an Nsight Compute raw CSV export. Measurement only.

    ncu --import block3.ncu-repz --page raw --csv > block3-raw.csv
    .venv\\Scripts\python.exe -m wan_dit_lab.harness.ncu_report block3-raw.csv
"""

import csv
import io
import sys

from wan_dit_lab.harness import block_profile as bp

COLUMNS = [
    ("ms", "gpu__time_duration.sum", 1),
    ("SM %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("DRAM %", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("L1 %", "l1tex__throughput.avg.pct_of_peak_sustained_active", 1),
    ("L2 %", "lts__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("L1 hit %", "l1tex__t_sector_hit_rate.pct", 1),
    ("L2 hit %", "lts__t_sector_hit_rate.pct", 1),
    ("tensor imma %", "sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active", 1),
    ("tensor hmma %", "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active", 1),
    ("FMA %", "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", 1),
    ("ALU %", "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", 1),
    ("issue %", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1),
    ("warps active %", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    ("warps max %", "sm__maximum_warps_per_active_cycle_pct", 1),
    ("regs", "launch__registers_per_thread", 1),
    ("stall long sb", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", 1),
    ("stall math", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", 1),
    ("stall mio", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", 1),
    ("stall wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", 1),
    ("stall barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", 1),
    ("stall short sb", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", 1),
]


def _number(cell):
    try:
        return float(cell.replace(",", ""))
    except ValueError:
        return cell


def parse(text):
    """Rows of kernels as {column: value}; the units row is dropped, numbers lose their thousands separators."""
    lines = [l for l in text.splitlines() if l.startswith('"')]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header = rows[0]
    return [{h: _number(c) for h, c in zip(header, row)} for row in rows[2:]]


def metric(row, name):
    if name not in row:
        raise KeyError(name)
    return row[name]


def group(rows):
    out = {}
    for row in rows:
        out.setdefault(bp.part_of(row["Kernel Name"]) or "other", []).append(row)
    return out


def main():
    rows = parse(open(sys.argv[1], encoding="utf-8-sig").read())
    print("| # | kernel | " + " | ".join(c[0] for c in COLUMNS) + " |")
    for i, row in enumerate(rows):
        cells = []
        for _, name, scale in COLUMNS:
            v = row.get(name)
            cells.append(f"{v * scale:.2f}" if isinstance(v, float) else "")
        print(f"| {i} | {str(row['Kernel Name'])[:34]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
