"""Tables of metric_check's result.json for the report: distances within and across groups, energy distance per
scene, the reference-free panel against the spread of the reference.

    .venv\\Scripts\\python.exe -m wan_dit_lab.harness.metric_tables result.json
"""

import json
import sys

from wan_dit_lab.harness.metric_check import SCENES, energy, inside, summary

POINTS = ("544x960", "512x928")


def mean_of(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def distance_tables(d, part):
    """Markdown rows: per scene and point, d_rr, d_oo, d_ro unpaired, paired L, D; then the mean and spread."""
    lines = [f"| сцена | точка | d_rr | d_oo | d_ro непарное | L парное | D = d_ro − (d_rr + d_oo)/2 |",
             "|---|---|---|---|---|---|---|"]
    per_point = {p: [] for p in POINTS}
    for scene in SCENES:
        s = d[scene]
        rr = mean_of(s["rr"], part)
        for p in POINTS:
            oo, ro, paired = mean_of(s[f"oo_{p}"], part), mean_of(s[f"ro_unpaired_{p}"], part), mean_of(s[f"ro_paired_{p}"], part)
            e = energy(ro, rr, oo)
            per_point[p].append(dict(rr=rr, oo=oo, ro=ro, paired=paired, D=e))
            lines.append(f"| {scene} | {p} | {rr:.3f} | {oo:.3f} | {ro:.3f} | {paired:.3f} | {e:+.3f} |")
    for p in POINTS:
        rows = per_point[p]
        cells = []
        for k in ("rr", "oo", "ro", "paired", "D"):
            st = summary([r[k] for r in rows])
            cells.append(f"{st['mean']:+.3f} ± {st['std']:.3f}" if k == "D" else f"{st['mean']:.3f} ± {st['std']:.3f}")
        lines.append(f"| среднее по 4 сценам | {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines), per_point


def free_table(free):
    """Markdown rows: per metric and group, mean ± std over 12 videos, and in how many scenes each seed of the point
    falls inside the reference's range over its 3 seeds."""
    metrics = ("clip_text", "subject_consistency", "flicker_aligned")
    lines = ["| метрика | эталон 704, среднее ± σ (12 роликов) | точка | среднее ± σ | роликов точки в разбросе эталона своей сцены |",
             "|---|---|---|---|---|"]
    for m in metrics:
        ref = [free["ref704"][sc][s][m] for sc in SCENES for s in free["ref704"][sc]]
        rs = summary(ref)
        for p in POINTS:
            vals = [free[p][sc][s][m] for sc in SCENES for s in free[p][sc]]
            ps = summary(vals)
            hits = sum(inside(free[p][sc][s][m], [free["ref704"][sc][t][m] for t in free["ref704"][sc]])
                       for sc in SCENES for s in free[p][sc])
            lines.append(f"| {m} | {rs['mean']:.4f} ± {rs['std']:.4f} | {p} | {ps['mean']:.4f} ± {ps['std']:.4f} | {hits} из {len(vals)} |")
    return "\n".join(lines)


def main(path):
    r = json.loads(open(path, encoding="utf-8").read())
    for part, title in (("first", "первый чанк"), ("all", "весь ролик (3 чанка)")):
        table, _ = distance_tables(r["distances"], part)
        print(f"\n### Расстояния, {title}\n\n{table}")
    print("\n### Безэталонная панель\n\n" + free_table(r["reference_free"]))


if __name__ == "__main__":
    main(sys.argv[1])
