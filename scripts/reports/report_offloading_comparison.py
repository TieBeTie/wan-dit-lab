"""Historical baseline and current resource run; missing cells stay missing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import json

from wan_dit_lab.json_files import read_json as read  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/experiments/2026-10-18/12_Offloading_full_pipeline_30s"
NEW = ROOT / "outputs/experiments/2026-10-19/22_Calibrated_prefetch_5s"


def current(folder):
    if not (folder / "complete.json").exists():
        return {}
    record = read(folder / "record.json")
    from wan_dit_lab.harness import stream_summary

    metrics = stream_summary(record, warmup=3)
    result = dict(full_run_s=record["full_run_s"], first_frame_s=record["first_frame_s"],
                  completed_runs=1, compile=record["compile_mode"],
                  graph_count=record.get("torch_compile", {}).get("unique_graphs"))
    aliases = dict(steady_fps="steady_fps", chunk_delivery_p50_s="steady_median_s",
                   chunk_delivery_p90_s="steady_p90_s", chunk_delivery_p99_s="steady_p99_s",
                   pinned_peak_gib="pinned_peak_gib", cache_peak_gib="ram_cache_peak_gib",
                   ram_rss_peak_gib="rss_peak_gib", ram_private_peak_gib="private_commit_peak_gib",
                   vram_peak_live_gib="vram_peak_gib", vram_peak_reserved_gib="vram_reserved_peak_gib",
                   vram_device_peak_used_gib="vram_device_peak_used_gib",
                   vram_device_min_free_gib="vram_device_min_free_gib",
                   disk_read_gib="disk_read_gib", disk_write_gib="disk_write_gib",
                   disk_read_s="disk_read_wall_s", disk_write_s="disk_write_wall_s")
    result.update({target: metrics.get(source) for target, source in aliases.items()})
    return result


def main():
    old = read(OLD / "summary.json")["methods"]
    baseline = old["fixed_2x2"] | dict(completed_runs=5, compile="default", video_s=30, warmup_chunks=7)
    latest = current(NEW / "resident_first") | dict(video_s=5, warmup_chunks=3)
    measured = read(NEW / "comparison.json")["methods"][-1]
    readiness = measured["resource_readiness"]
    latest["readiness"] = f"{readiness['misses']} / {readiness['consumers']}"
    rows = [baseline, latest]
    metrics = [("Роликов", "completed_runs"), ("Длительность видео, с", "video_s"), ("torch.compile", "compile"),
               ("Прогрев, чанков", "warmup_chunks"),
               ("FPS после прогрева", "steady_fps"),
               ("Генерация, с", "full_run_s"), ("Первый кадр, с", "first_frame_s"),
               ("Чанк p50, с", "chunk_delivery_p50_s"), ("Чанк p90, с", "chunk_delivery_p90_s"),
               ("Чанк p99, с", "chunk_delivery_p99_s"), ("VRAM пик тензоров, GiB", "vram_peak_live_gib"),
               ("VRAM пик резерва, GiB", "vram_peak_reserved_gib"),
               ("VRAM устройства: пик занятой, GiB", "vram_device_peak_used_gib"),
               ("VRAM устройства: минимум свободной, GiB", "vram_device_min_free_gib"),
               ("RAM RSS пик, GiB", "ram_rss_peak_gib"),
               ("Private commit пик, GiB", "ram_private_peak_gib"),
               ("Pinned RAM пик, GiB", "pinned_peak_gib"), ("RAM-кэш пик, GiB", "cache_peak_gib"),
               ("Диск Read, GiB", "disk_read_gib"), ("Диск Write, GiB", "disk_write_gib"),
               ("Диск Read, с", "disk_read_s"), ("Диск Write, с", "disk_write_s"),
               ("Опоздания / обращения к ресурсам", "readiness")]
    lines = ['| <span class="wan-offload-results wan-offload-comparison">Метрика</span> | Веса: 2×2 | Ресурсы: текущий |',
             '| :--- | ---: | ---: |']
    for label, key in metrics:
        values = ["·" if row.get(key) is None else f"{row[key]:.3f}" if isinstance(row[key], float) else str(row[key]) for row in rows]
        lines.append(f"| {label} | {' | '.join(values)} |")
    table = "\n".join(lines)
    text = ("480×832, compile=default. Веса: пять промптов по 30 с; ресурсы: один промпт, 5 с. "
            "Прогрев различается, прямое сравнение FPS и пиков ограничено. · = нет замера.\n\n" + table + "\n")
    (NEW / "Comparison_current.md").write_text(text, encoding="utf-8")
    (NEW / "comparison_current.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    note = ROOT / "notes/research/Offloading.md"
    content = note.read_text(encoding="utf-8")
    start, end = '<!-- offloading-comparison:start -->', '<!-- offloading-comparison:end -->'
    block = (start + "\n\n" + text + "\n[Полные замеры и ролики](../../outputs/experiments/2026-10-19/22_Calibrated_prefetch_5s/Comparison_current.md) · [Старые исходные данные](../../outputs/experiments/2026-10-18/12_Offloading_full_pipeline_30s/Сравнение.md)\n\n" + end)
    if start in content:
        a, b = content.index(start), content.index(end) + len(end)
    else:
        a = content.index("**Проверка:**")
        b = content.index("**Источники:**", a)
    content = content[:a] + block + "\n\n" + content[b:]
    short_start, short_end = '<!-- offloading-readiness-short:start -->', '<!-- offloading-readiness-short:end -->'
    if short_start in content:
        a, b = content.index(short_start), content.index(short_end) + len(short_end)
        brief = ("Кэш ресурсов: VRAM 6.19 GiB, RAM ≤ 832 MiB. Кадры совпадают с исходным прогоном 18.10. "
                 "На том же коротком префиксе веса 2×2: 1.596 FPS.\n\n"
                 "[Калибровка и сопоставимый префикс](../../outputs/experiments/2026-10-19/22_Calibrated_prefetch_5s/Comparison.md) · "
                 "[Текущий профиль](../../configs/experiments/offloading_managed_cost_guarded.json)")
        content = content[:a] + brief + content[b:]
    note.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
