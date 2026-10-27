"""The report of one live_tutor_bench.py run: summary.json, report.md and trace.json (open in ui.perfetto.dev).

    .\\.venv\\Scripts\\python.exe scripts/reports/live_tutor_report.py <run folder>

Wall time throughout. Live's own clock is perf_counter; record.json carries the offset to wall time.
"""

import json
import re
import statistics
import sys
from pathlib import Path


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return dict(n=len(values), p50=statistics.median(values), p90=pick(.9), p99=pick(.99), max=values[-1],
                mean=statistics.mean(values))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def tutor_intervals(tutor_events, tutor_log):
    """(begin, end, kind, label): recognitions from the log ("transcribed Xs of audio in Ys" ends at its stamp),
    and whole replies from the journal ("latency": from the end of my speech to the start of its voice)."""
    spans = []
    for line in tutor_log:
        match = re.match(r"(\d+) \[[\d:]+\] timing: transcribed ([\d.]+)s of audio in ([\d.]+)s", line)
        if match:
            end = float(match[1])
            spans.append((end - float(match[3]), end, "stt", f"{match[2]} s of speech"))
    for event in tutor_events:
        if event["kind"] == "latency":
            spans.append((event["t"] - event["total"], event["t"], "reply", event.get("detail", "")))
    return sorted(spans)


def from_live_log(out, measured_start, frames_per_chunk=32):
    """A run whose window closed before the harness saved: chunk and T5 times from Live's own printed lines."""
    chunks, t5 = [], []
    for line in (out / "live.log").read_text(encoding="utf-8").splitlines():
        stamp, _, text = line.partition(" ")
        if match := re.match(r"chunk \d+: ([\d.]+) s", text):
            chunks.append(dict(index=len(chunks), delivery_s=float(match[1]), finished_at=float(stamp),
                               frames=frames_per_chunk))
        elif match := re.match(r"prompt encoded in ([\d.]+) s", text):
            # from the prompt taken to its encoding done: the wait for the GPU between chunks included
            t5.append(dict(kind="t5", begin=float(stamp) - float(match[1]), seconds=float(match[1]),
                           disk_read_mb=0., prompt="(from live.log)"))
    warmup = sum(c["finished_at"] < measured_start - 0.5 for c in chunks)
    steady = chunks[warmup:]
    return dict(clock_offset_s=0., source="live.log", config=dict(harness=dict(warmup_chunks=warmup)),
                summary=dict(raw_chunks=chunks, events=t5,
                             generated_fps=sum(c["frames"] for c in steady) / sum(c["delivery_s"] for c in steady)))


def overlaps(begin, end, spans):
    return any(b < end and begin < e for b, e, *_ in spans)


def window(samples, begin, end):
    return [s for s in samples if begin <= s["t"] <= end]


def main():
    out = Path(sys.argv[1])
    phases = json.loads((out / "phases.json").read_text(encoding="utf-8"))
    record = (json.loads((out / "record.json").read_text(encoding="utf-8")) if (out / "record.json").exists()
              else from_live_log(out, phases.get("measured_start", phases["live_start"])))
    system, gpu = rows(out / "system.jsonl"), rows(out / "gpu.jsonl")
    tutor_events = rows(out / "tutor.jsonl")
    tutor_log = (out / "tutor.log").read_text(encoding="utf-8").splitlines() if (out / "tutor.log").exists() else []
    spans = tutor_intervals(tutor_events, tutor_log)
    t0 = phases["started"]
    calm = (t0, phases["live_start"])
    measured = (phases.get("measured_start", phases["live_start"]), phases["live_exit"])
    summary = dict(phases={k: round(v - t0, 1) for k, v in phases.items() if isinstance(v, float)})

    def machine(begin, end):
        part = window(system, begin, end)
        roles = {}
        for role in sorted({r for s in part for r in s["roles"]}):
            got = [s["roles"][role] for s in part if role in s["roles"]]
            roles[role] = dict(cpu_mean=mean([g["cpu"] for g in got]), cpu_max=max(g["cpu"] for g in got),
                               private_max_gib=max(g["private"] for g in got) / 2**30,
                               rss_max_gib=max(g["rss"] for g in got) / 2**30,
                               disk_read_mib=(got[-1]["read"] - got[0]["read"]) / 2**20,
                               page_faults=got[-1]["faults"] - got[0]["faults"])
        graphics = window(gpu, begin, end)
        return dict(samples=len(part),
                    ram_available_min_gib=min((s["ram_available"] for s in part), default=0) / 2**30,
                    ram_available_mean_gib=(mean([s["ram_available"] for s in part]) or 0) / 2**30,
                    pagefile_growth_gib=(part[-1]["pagefile_used"] - part[0]["pagefile_used"]) / 2**30 if part else None,
                    cores_mean=[round(mean(c), 1) for c in zip(*[s["cores"] for s in part])] if part else None,
                    roles=roles,
                    gpu=dict(power_mean_w=mean([g.get("pwr") for g in graphics]),
                             sm_mean=mean([g.get("sm") for g in graphics]),
                             fb_max_mib=max((g.get("fb") or 0 for g in graphics), default=None),
                             pcie_rx_mean_mbs=mean([g.get("rxpci") for g in graphics]),
                             pcie_tx_mean_mbs=mean([g.get("txpci") for g in graphics])))

    summary["calibration"] = machine(*calm)
    summary["measured"] = machine(*measured)
    events = []  # trace.json
    us = lambda t: (t - t0) * 1e6  # noqa: E731

    if record:
        offset = record["clock_offset_s"]
        live = record["summary"]
        chunks = live["raw_chunks"]
        warmup = record["config"]["harness"]["warmup_chunks"]
        steady = chunks[warmup:]
        for c in chunks:
            c["end"] = c["finished_at"] + offset
            c["begin"] = c["end"] - c["delivery_s"]
            c["tutor_busy"] = overlaps(c["begin"], c["end"], spans)
            c["rest_ms"] = c["delivery_s"] * 1000 - c.get("dit_ms", 0) - c.get("decoder_ms", 0)
            events.append(dict(name=f"chunk {c['index']}", ph="X", pid=1, tid=1, ts=us(c["begin"]),
                               dur=c["delivery_s"] * 1e6, args={k: c.get(k) for k in ("dit_ms", "decoder_ms", "rest_ms", "tutor_busy")}))
        t5 = [e for e in live.get("events", []) if e["kind"] == "t5"]
        for e in t5:
            e["wall"] = e["begin"] + offset
            events.append(dict(name="T5", ph="X", pid=1, tid=2, ts=us(e["wall"]), dur=e["seconds"] * 1e6,
                               args=dict(prompt=e["prompt"], disk_read_mb=round(e["disk_read_mb"], 1))))
        for begin, end, queued in live.get("raw_ticks", []):
            events.append(dict(name="tick", ph="X", pid=1, tid=3, ts=us(begin + offset), dur=(end - begin) * 1e6,
                               args=dict(queued=queued)))
        # the gap after a prompt: a chunk waits for T5 to give the GPU back
        gaps = [b["begin"] - a["end"] for a, b in zip(steady, steady[1:])]
        prompt_gaps = [dict(prompt=e["prompt"], t5_s=round(e["seconds"], 2), disk_read_mb=round(e["disk_read_mb"], 1),
                            stall_s=round(max((b["begin"] - a["end"] for a, b in zip(chunks, chunks[1:])
                                               if a["end"] <= e["wall"] + 0.05 and b["begin"] >= e["wall"]), default=0), 2))
                       for e in t5]
        busy = [c["delivery_s"] for c in steady if c["tutor_busy"]]
        idle = [c["delivery_s"] for c in steady if not c["tutor_busy"]]
        blocks = (record.get("steady_prefetch") or {}).get("samples", [])
        late_by_chunk = {}
        for b in blocks:
            late_by_chunk[b["chunk"]] = late_by_chunk.get(b["chunk"], 0.) + max(b["late_ms"], 0.)
        busy_chunks = {c["index"] for c in steady if c["tutor_busy"]}
        display = live.get("display") or {}
        target = None
        if display.get("tick_work"):
            made = statistics.median([c["frames"] / c["delivery_s"] for c in steady])
            target = min(24, .95 * made)  # live_video.py tick(): min(FPS, 0.95 * made)
            work = display["tick_work"]["p50_ms"]
            display["target_fps"] = target
            display["predicted_fps_if_tick_work_adds"] = 1000 / (int(1000 / target) + work)
        summary["live"] = dict(
            source=record.get("source", "record.json"), status=record.get("status"),
            generated_fps=live.get("generated_fps"), displayed_fps=live.get("displayed_fps"),
            chunk_s=quantiles([c["delivery_s"] for c in steady]),
            dit_ms=quantiles([c["dit_ms"] for c in steady if "dit_ms" in c]),
            decoder_ms=quantiles([c["decoder_ms"] for c in steady if "decoder_ms" in c]),
            rest_ms=quantiles([c["rest_ms"] for c in steady]),
            between_chunks_s=quantiles(gaps),
            chunk_s_tutor_busy=quantiles(busy), chunk_s_tutor_idle=quantiles(idle),
            display=display, t5=prompt_gaps,
            vram=record.get("memory", {}).get("vram"),
            peak_allocated_gib=(record.get("peak_allocated_bytes") or 0) / 2**30,
            peak_reserved_gib=(record.get("peak_reserved_bytes") or 0) / 2**30,
            calibration_prefetch={k: v for k, v in (record["calibration"]["profiles"][0] if record.get("calibration")
                                  and record["calibration"]["profiles"] else {}).items()
                                  if k not in ("samples", "per_block")},
            steady_prefetch={k: v for k, v in (record.get("steady_prefetch") or {}).items()
                             if k not in ("samples", "per_block")},
            late_ms_per_chunk_tutor_busy=mean([late_by_chunk.get(i, 0.) for i in busy_chunks]) if busy_chunks else None,
            late_ms_per_chunk_tutor_idle=mean([late_by_chunk.get(c["index"], 0.) for c in steady
                                               if c["index"] not in busy_chunks]))
    summary["tutor"] = dict(
        recognitions=quantiles([e - b for b, e, kind, _ in spans if kind == "stt" and measured[0] <= e <= measured[1]]),
        replies=quantiles([e - b for b, e, kind, _ in spans if kind == "reply" and measured[0] <= e <= measured[1]]),
        replies_detail=[label for b, e, kind, label in spans if kind == "reply" and measured[0] <= e <= measured[1]],
        recognitions_calibration=quantiles([e - b for b, e, kind, _ in spans if kind == "stt" and calm[0] <= e <= calm[1]]))
    speech = [(float(line.split()[0]), line.split("] ", 1)[1]) for line in tutor_log
              if "] tts" in line and measured[0] <= float(line.split()[0]) <= measured[1]]
    summary["tutor"]["tts"] = dict(first_chunk_s=quantiles([float(m[1]) for _, text in speech
                                                            if (m := re.search(r"first chunk ([\d.]+)s", text))]),
                                   retries=sum("asking again" in text for _, text in speech),
                                   lost=sum("tts error" in text for _, text in speech))
    for at, text in speech:
        if "first chunk" not in text:
            events.append(dict(name=text[:60], ph="i", s="p", pid=2, tid=3, ts=us(at)))
    for b, e, kind, label in spans:
        events.append(dict(name=kind, ph="X", pid=2, tid=1 if kind == "stt" else 2, ts=us(b), dur=(e - b) * 1e6,
                           args=dict(detail=label)))
    for s in system:
        for role, got in s["roles"].items():
            events.append(dict(name=f"CPU {role}", ph="C", pid=3, ts=us(s["t"]), args=dict(percent=got["cpu"])))
            events.append(dict(name=f"private GiB {role}", ph="C", pid=3, ts=us(s["t"]),
                               args=dict(gib=round(got["private"] / 2**30, 3))))
        events.append(dict(name="RAM available GiB", ph="C", pid=3, ts=us(s["t"]),
                           args=dict(gib=round(s["ram_available"] / 2**30, 3))))
    for g in gpu:
        for key in ("pwr", "sm", "fb", "rxpci", "txpci"):
            if g.get(key) is not None:
                events.append(dict(name=f"GPU {key}", ph="C", pid=4, ts=us(g["t"]), args={key: g[key]}))
    for name, at in phases.items():
        if isinstance(at, float):
            events.append(dict(name=name, ph="i", s="g", pid=1, tid=1, ts=us(at)))
    meta = [dict(name="process_name", ph="M", pid=p, args=dict(name=n))
            for p, n in ((1, "Live"), (2, "tutor"), (3, "machine"), (4, "GPU"))]
    meta += [dict(name="thread_name", ph="M", pid=1, tid=t, args=dict(name=n))
             for t, n in ((1, "chunks"), (2, "T5"), (3, "window ticks"))]
    (out / "trace.json").write_text(json.dumps(dict(traceEvents=meta + events)), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(s):
    lines = ["# Live + тьютор", ""]
    for name, part in (("Калибровка (без Live)", s["calibration"]), ("Измерение", s["measured"])):
        lines += [f"## {name}", "",
                  f"RAM свободно: мин {fmt(part['ram_available_min_gib'])} GiB, среднее {fmt(part['ram_available_mean_gib'])} GiB; "
                  f"рост файла подкачки {fmt(part['pagefile_growth_gib'])} GiB",
                  f"GPU: {fmt(part['gpu']['power_mean_w'], 1)} W, SM {fmt(part['gpu']['sm_mean'], 0)} %, "
                  f"память до {part['gpu']['fb_max_mib']} MiB, PCIe rx {fmt(part['gpu']['pcie_rx_mean_mbs'], 0)} MB/s", "",
                  "| процесс | CPU среднее % | CPU max % | private max GiB | чтение с диска MiB | page faults |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {role} | {fmt(r['cpu_mean'], 0)} | {fmt(r['cpu_max'], 0)} | {fmt(r['private_max_gib'])} | "
                  f"{fmt(r['disk_read_mib'], 0)} | {r['page_faults']} |" for role, r in part["roles"].items()]
        lines.append("")
    if live := s.get("live"):
        q = lambda x, k="p50", d=2: fmt(x[k] if x else None, d)  # noqa: E731
        d = live["display"] or {}
        lines += ["## Live", "",
                  f"Сгенерировано {fmt(live['generated_fps'])} fps, показано в окне {fmt(live['displayed_fps'])} fps.",
                  f"Чанк: p50 {q(live['chunk_s'])} s, p90 {q(live['chunk_s'], 'p90')} s, max {q(live['chunk_s'], 'max')} s; "
                  f"из них DiT {q(live['dit_ms'], d=0)} ms, декодер {q(live['decoder_ms'], d=0)} ms, "
                  f"остальное (копия кадров, Python) {q(live['rest_ms'], d=0)} ms; пауза между чанками p50 {q(live['between_chunks_s'], d=3)} s.",
                  f"Чанк при занятом тьюторе p50 {q(live['chunk_s_tutor_busy'])} s (n={live['chunk_s_tutor_busy']['n'] if live['chunk_s_tutor_busy'] else 0}), "
                  f"при свободном {q(live['chunk_s_tutor_idle'])} s.",
                  f"Окно: работа тика p50 {q(d.get('tick_work'), 'p50_ms', 1) if d.get('tick_work') else 'n/a'} ms, "
                  f"интервал кадров p50 {q(d.get('frame_interval'), 'p50_ms', 1) if d.get('frame_interval') else 'n/a'} ms, "
                  f"целевая частота {fmt(d.get('target_fps'))} fps, прогноз «тик добавляется к периоду» "
                  f"{fmt(d.get('predicted_fps_if_tick_work_adds'))} fps, пустая очередь в {d.get('queue_empty_ticks')} тиках.",
                  f"VRAM: peak allocated {fmt(live['peak_allocated_gib'])} GiB, reserved {fmt(live['peak_reserved_gib'])} GiB.",
                  f"Потоковые блоки, калибровка: {live['calibration_prefetch'].get('late_consumers')} поздних из "
                  f"{live['calibration_prefetch'].get('consumers')}; измерение: {live['steady_prefetch'].get('late_consumers')} из "
                  f"{live['steady_prefetch'].get('consumers')}, опоздание всего {fmt(live['steady_prefetch'].get('late_ms_total'), 1)} ms; "
                  f"на чанк при занятом тьюторе {fmt(live['late_ms_per_chunk_tutor_busy'], 1)} ms, при свободном {fmt(live['late_ms_per_chunk_tutor_idle'], 1)} ms.",
                  "", "| промпт | T5 s | чтение с диска MB | простой Live s |", "|---|---|---|---|"]
        lines += [f"| {t['prompt'][:50]} | {t['t5_s']} | {t['disk_read_mb']} | {t['stall_s']} |" for t in live["t5"]]
        lines.append("")
    t = s["tutor"]
    lines += ["## Тьютор", "",
              f"Распознавание во время Live: {fmt(t['recognitions']['p50'] if t['recognitions'] else None)} s p50 "
              f"(n={t['recognitions']['n'] if t['recognitions'] else 0}); в калибровке "
              f"{fmt(t['recognitions_calibration']['p50'] if t['recognitions_calibration'] else None)} s.",
              f"Ответ от конца речи до голоса: p50 {fmt(t['replies']['p50'] if t['replies'] else None)} s "
              f"(n={t['replies']['n'] if t['replies'] else 0}).",
              f"Озвучка: первый звук фразы p50 {fmt(t['tts']['first_chunk_s']['p50'] if t['tts']['first_chunk_s'] else None)} s, "
              f"повторных запросов {t['tts']['retries']}, фраз без звука {t['tts']['lost']}.", ""]
    lines += [f"- {detail}" for detail in t["replies_detail"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
