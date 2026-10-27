"""Dependency-free presentation of local RoPE results, never a second benchmark."""

import html
import json
from pathlib import Path

NA = "нет данных"
STATUSES = {"passed": "Пройдено", "running": "Выполняется", "failed": "Ошибка",
            "regression": "Регрессия скорости", "accuracy_regression": "Регрессия точности"}


def at(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def number(value, digits=3):
    return NA if value is None else f"{value:.{digits}f}"


def error(value):
    return NA if value is None else f"{value:.3g}"


def comparison_rows(row, *, block=False):
    rows = []
    for percentile in ("p50", "p90", "p99"):
        rows.append((f"GPU {percentile}, мс", *[
            number(at(row, "timing", label, f"{percentile}_ms"))
            for label in ("original", "triton")]))
    rows.append(("Вызов с подачей из Python, p50, мс", *[
        number(at(row, "wall", label, "p50_ms")) for label in ("original", "triton")]))
    memory = [at(row, "memory", label, "additional_peak_allocated_bytes")
              for label in ("original", "triton")]
    rows.append(("Дополнительная пиковая VRAM, MiB", *[
        number(value / 2**20, 1) if value is not None else NA for value in memory]))
    rows.append(("GPU-ядра", *[str(at(row, "kernels", label, "count"))
                               if at(row, "kernels", label, "count") is not None else NA
                               for label in ("original", "triton")]))
    rows += memory_rows(row.get("memory", {}))
    if block:
        for label, field in (("max |Δ| к FP32", "max_abs"), ("Относительная L2 к FP32", "relative_l2")):
            rows.append((label, error(at(row, "accuracy_gate", "baseline", field)),
                         error(at(row, "accuracy_gate", "candidate", field))))
    else:
        numerical = row.get("numerical", {})
        previous = numerical.get("compiled_original_against_fp64", numerical.get("original_against_fp64", {}))
        candidate = numerical.get("compiled_against_fp64", numerical.get("against_fp64", {}))
        rows += [("max |Δ| к FP64", error(previous.get("max_abs")), error(candidate.get("max_abs"))),
                 ("Относительная L2 к FP64", error(previous.get("relative_l2")), error(candidate.get("relative_l2"))),
                 ("Изменение нормы, относительная L2", error(at(numerical, "original_norm_drift", "relative_l2")),
                  error(at(numerical, "norm_drift", "relative_l2"))),
                 ("max |Δ градиента| к FP64", error(at(row, "gradient", "original_against_fp64", "max_abs")),
                  error(at(row, "gradient", "against_fp64", "max_abs")))]
    return rows


def memory_rows(records):
    fields = [("VRAM исходная, GiB", "vram", "baseline_bytes"),
              ("VRAM текущая, GiB", "vram", "current_bytes"),
              ("VRAM пиковая, GiB", "vram", "peak_bytes"),
              ("VRAM reserved, пик, GiB", "vram", "reserved_peak_bytes"),
              ("VRAM свободная, минимум, GiB", "vram", "free_min_bytes"),
              ("RAM исходная RSS, GiB", "ram", "baseline_bytes"),
              ("RAM текущая RSS, GiB", "ram", "current_bytes"),
              ("RAM пиковая RSS, выборки, GiB", "ram", "peak_bytes"),
              ("RAM свободная, минимум, GiB", "ram", "free_min_bytes")]
    rows = []
    for label, group, field in fields:
        values = [at(records, arm, group, field) for arm in ("original", "triton")]
        if any(value is not None for value in values):
            rows.append((label, *[number(value / 2**30) if value is not None else NA for value in values]))
    return rows


def selected_case(report):
    rows = report.get("operators", [])
    target = at(report, "config", "summary_case") or "video_b1_bf16"
    return next((row for row in rows if at(row, "case", "name") == target), rows[0] if rows else None)


def context(report, row):
    pieces = [at(report, "metadata", "environment", "gpu")]
    if row:
        case = row["case"]
        pieces += [case.get("dtype", "").replace("bfloat16", "BF16").replace("float", "FP"),
                   f"B={case.get('batch', 1)}", "×".join(map(str, case.get("grid", [])))]
    rounds = at(report, "config", "harness", "number") or at(report, "config", "rounds")
    pieces += ["torch.compile", f"{rounds} раундов"]
    return " · ".join(str(piece) for piece in pieces if piece)


def test_label(report):
    count = at(report, "tests", "passed")
    if count is None:
        return "результат тестов в tests.log"
    noun = "тестов" if count % 100 in range(11, 15) else {1: "тест", 2: "теста", 3: "теста", 4: "теста"}.get(count % 10, "тестов")
    return f"{count} {noun}"


def speedup(row):
    previous, candidate = at(row, "timing", "original", "p50_ms"), at(row, "timing", "triton", "p50_ms")
    return f"×{previous / candidate:.2f}" if previous and candidate else NA


def all_cases(report):
    rows = report.get("operators", []) + ([report["block"]] if "block" in report else [])
    return [(at(row, "case", "name") or "Wan-блок",
             number(at(row, "timing", "original", "p50_ms")),
             number(at(row, "timing", "triton", "p50_ms")), speedup(row)) for row in rows]


def live_rows(live):
    fields = [("Генерация, кадров/с", "generated_fps"),
              ("Подача кадров в окно, кадров/с", "displayed_fps"),
              ("Чанк p50, мс", "p50_chunk_ms"), ("Чанк p99, мс", "p99_chunk_ms")]
    rows = [(label, number(at(live, "original", "summary", field)),
             number(at(live, "triton", "summary", field))) for label, field in fields]
    common = memory_rows({arm: at(live, arm, "memory") or {} for arm in ("original", "triton")})
    rows += common
    for label, field in (("Пиковая занятая VRAM, GiB", "peak_allocated_bytes"),
                         ("Пиковая reserved VRAM, GiB", "peak_reserved_bytes")):
        if not common and any(at(live, arm, field) is not None for arm in ("original", "triton")):
            rows.append((label, *[number(at(live, arm, field) / 2**30)
                                 if at(live, arm, field) is not None else NA
                                 for arm in ("original", "triton")]))
    profiles = [at(live, arm, "calibration", "profiles") for arm in ("original", "triton")]
    if any(profiles):
        rows.append(("Опоздания prefetch / загрузки, прогрев", *[
            f"{sum(p['late_consumers'] for p in value)} / {sum(p['consumers'] for p in value)}"
            if value else NA for value in profiles]))
    return rows


def live_context(live):
    settings = live.get("settings", {})
    return f"Live · {settings.get('height')}×{settings.get('width')} · LongLive2 S2 · W4A4 · SageAttention · Windows"


def cache_store_rows(row):
    return comparison_rows(row)[:6] + memory_rows(row.get("memory", {})) + [("FP8 и масштабы побитно", "эталон",
                                       "совпадают" if row.get("bitwise") else NA)]


def markdown_table(rows, headings=("Метрика", "PyTorch", "Triton")):
    return ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headings)-1)) + " |"] + [
        "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows]


def render_markdown(report):
    row = selected_case(report)
    lines = ["# RoPE 3D", "", f"**{STATUSES.get(report['status'], report['status'])}** · {test_label(report)}",
             "", context(report, row), ""]
    if row:
        lines += [f"**RoPE {speedup(row)}**", ""] + markdown_table(comparison_rows(row))
    if "block" in report:
        block = report["block"]
        lines += ["", f"**Wan-блок {speedup(block)}** · одинаковые случайные веса", ""]
        lines += markdown_table(comparison_rows(block, block=True))
        lines += ["", "Выходы compiled-блока могут различаться из-за округлений при fusion.",
                  "Точность обеих версий проверяется относительно одного FP32-эталона."]
    if "live" in report:
        lines += ["", f"**{live_context(report['live'])}**", ""]
        lines += markdown_table(live_rows(report["live"]))
        lines += ["", "Один промпт и seed. После прогрева. Генерация включает DiT, декодер и перенос пикселей.",
                  "Очередь показа измеряется отдельно, окно ограничено 24 FPS."]
    if "cache_store" in report:
        lines += ["", f"**FP8 KV · запись в кэш {speedup(report['cache_store'])}**", ""]
        lines += markdown_table(cache_store_rows(report["cache_store"]))
        lines += ["", "PyTorch: текущая eager-запись внутри отключённого от compile attention. Triton: одно ядро."]
        tests = at(report, "cache_store", "artifacts", "tests")
        if tests:
            lines += ["", f"[{at(report, 'cache_store', 'tests', 'passed')} тестов]({tests})"]
    if "cache_store_live" in report:
        lines += ["", "**Live · FP8 KV · RoPE Triton в обоих вариантах**", ""]
        lines += markdown_table(live_rows(report["cache_store_live"]), ("Метрика", "Прежняя запись", "Fusion"))
    if all_cases(report):
        lines += ["", "<details>", "<summary>Все размеры</summary>", ""]
        lines += markdown_table(all_cases(report), ("Случай", "PyTorch, мс", "Triton, мс", "Ускорение"))
        lines += ["", "</details>"]
    if report.get("error"):
        lines += ["", "```text", report["error"], "```"]
    lines += ["", "GPU-время: CUDA graph replay. Полное время вызова измерено отдельно.",
              "p99 описывает эту небольшую выборку. FPS live приведён отдельно, если измерен.",
              "DRAM-трафик и аппаратные счётчики не измерены.", "",
              "[Все метрики](results.json) · [Тесты](tests.log)", ""]
    return "\n".join(lines)


def html_table(rows, headings=("Метрика", "PyTorch", "Triton")):
    def esc(value):
        return html.escape(str(value))
    head = "".join(f"<th>{esc(cell)}</th>" for cell in headings)
    body = "".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report):
    row = selected_case(report)
    esc = html.escape
    status = STATUSES.get(report["status"], report["status"])
    state_class = "ok" if report["status"] == "passed" else ""
    pieces = [f'<header><h1>RoPE 3D</h1><span class="status {state_class}">{esc(status)}</span></header>',
              f'<p class="muted">{esc(test_label(report))} · {esc(context(report, row))}</p>']
    if row:
        pieces += [f"<h2>RoPE <strong>{speedup(row)}</strong></h2>", html_table(comparison_rows(row))]
    if "block" in report:
        block = report["block"]
        pieces += [f"<h2>Wan-блок <strong>{speedup(block)}</strong></h2>",
                   html_table(comparison_rows(block, block=True)),
                   '<p class="muted">Одинаковые случайные веса. Различия compiled-выходов проверяются '
                   'по одному FP32-эталону с учётом округлений при fusion.</p>']
    if "live" in report:
        live = report["live"]
        pieces += [f"<h2>{esc(live_context(live))}</h2>", html_table(live_rows(live)),
                   '<p class="muted">Один промпт и seed, после прогрева. Генерация: DiT + декодер + пиксели.'
                   ' Показ измеряется отдельно, окно ограничено 24 FPS.</p>']
    if "cache_store" in report:
        cache = report["cache_store"]
        pieces += [f"<h2>FP8 KV · запись в кэш <strong>{speedup(cache)}</strong></h2>",
                   html_table(cache_store_rows(cache)),
                   '<p class="muted">PyTorch: текущая eager-запись внутри отключённого от compile attention.'
                   ' Triton: одно ядро, без временных FP8-тензоров.</p>']
        tests = at(cache, "artifacts", "tests")
        if tests:
            pieces += [f'<p class="muted"><a href="{esc(tests)}">{at(cache, "tests", "passed")} тестов</a></p>']
    if "cache_store_live" in report:
        pieces += ['<h2>Live · FP8 KV · RoPE Triton в обоих вариантах</h2>',
                   html_table(live_rows(report["cache_store_live"]), ("Метрика", "Прежняя запись", "Fusion"))]
    pieces += ['<details><summary>Все размеры</summary>',
               html_table(all_cases(report), ("Случай", "PyTorch, мс", "Triton, мс", "Ускорение")), '</details>']
    if report.get("error"):
        pieces += [f'<pre>{esc(report["error"])}</pre>']
    pieces += ['<footer>GPU-время: CUDA graph replay. Полное время вызова измерено отдельно.<br>'
               'p99 описывает небольшую выборку. FPS live приведён отдельно, если измерен.<br>'
               'DRAM-трафик и аппаратные счётчики не измерены.<br><br>'
               '<a href="results.json">Все метрики</a> · <a href="tests.log">Тесты</a></footer>']
    css = """
body{margin:0;background:#fff;color:#202124;font:15px/1.5 'Segoe UI',Arial,sans-serif}
main{max-width:880px;margin:auto;padding:36px 28px 48px}header{display:flex;align-items:center;justify-content:space-between}
h1{font-size:26px;font-weight:500;margin:0}h2{font-size:17px;font-weight:500;margin:28px 0 8px}h2 strong{float:right;font-weight:500}
.status{font-size:12px;padding:3px 9px;border:1px solid #ddd;border-radius:4px}.status.ok{color:#237546;border-color:#b7d4c0}
.muted,footer{color:#707070;font-size:13px}table{width:100%;border-collapse:collapse;table-layout:fixed}
th,td{padding:7px 0;border-bottom:1px solid #e9e9e9;vertical-align:top;overflow-wrap:anywhere}
th{color:#707070;font-weight:500;text-align:left}th:first-child,td:first-child{width:50%}
th:not(:first-child),td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}
details{margin-top:26px}summary{cursor:pointer;color:#707070;font-size:13px;margin-bottom:10px}
footer{margin-top:28px}a{color:inherit;text-underline-offset:3px}pre{white-space:pre-wrap;font-size:12px;border-left:2px solid #ddd;padding-left:12px}
@media(max-width:600px){main{padding:20px 14px}body{font-size:13px}th:first-child,td:first-child{width:46%}}
"""
    return ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>RoPE 3D</title><style>{css}</style><body><main>' + "\n".join(pieces) + '</main></body></html>')


def write_latest(directory, status):
    directory = Path(directory)
    pointer = dict(status=status, run_dir=str(directory), report_html=str(directory / "Report.html"),
                   report_md=str(directory / "Report.md"), results_json=str(directory / "results.json"))
    (directory.parent / "latest.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (directory / "Report.md").write_text(render_markdown(report), encoding="utf-8")
    (directory / "Report.html").write_text(render_html(report), encoding="utf-8")


def write_comparison(directory, config, records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    live = dict(status="passed", settings=config, **records)
    old = at(records, "original", "summary", "generated_fps")
    new = at(records, "triton", "summary", "generated_fps")
    if old and new:
        live["speedup"] = new / old
    (directory / "comparison.json").write_text(json.dumps(live, indent=2), encoding="utf-8")
    destination = Path(config["report_dir"]) if config.get("report_dir") else directory
    path = destination / "results.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(status="passed", config=config)
    report[config.get("report_key", "live")] = live
    write_report(report, destination)
    return destination / "Report.html"
