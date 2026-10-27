"""Run the existing Live UI with process-local RoPE selection and bounded telemetry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import ast
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from wan_dit_lab import harness  # noqa: E402
from wan_dit_lab.experiments.live_rope_metrics import LiveMetrics, instrument_live_source  # noqa: E402
from wan_dit_lab.runtime.pipeline import install, live_arguments, resolve_experiment  # noqa: E402
from wan_dit_lab.harness.memory import MemorySampler, summarize_memory  # noqa: E402


class StageTimers:
    """GPU time of the DiT and the decoder of each chunk, from CUDA events read at the chunk's own D2H fence, so
    nothing is synchronized that Live does not already wait for; and the wall time of each T5 encoding, with the
    bytes the process read from disk meanwhile (T5 is a mapped file: a cold page cache shows up there)."""

    def __init__(self, metrics):
        from wan_dit_lab.generation.causal import Stream
        from wan_dit_lab.models.text_encoder import FastTextEncoder, TextEncoder
        from wan_dit_lab.models.vae import TAEHVStream

        self.metrics, self.pending = metrics, []
        Stream.next_chunk = self.timed(Stream.next_chunk, "dit_ms")
        TAEHVStream.__call__ = self.timed(TAEHVStream.__call__, "decoder_ms")
        for encoder in (FastTextEncoder, TextEncoder):
            encoder.__call__ = self.encoding(encoder.__call__)

    def timed(self, function, name):
        def wrapper(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = function(*args, **kwargs)
            end.record()
            self.pending.append((name, begin, end))
            return result
        return wrapper

    def encoding(self, function):
        import psutil

        process = psutil.Process()

        def wrapper(encoder, prompts):
            read, begin = process.io_counters().read_bytes, time.perf_counter()
            result = function(encoder, prompts)
            torch.cuda.synchronize()  # the GPU is held for T5 between chunks; this wait is part of its cost
            self.metrics.record_event("t5", begin=begin, seconds=time.perf_counter() - begin,
                                      disk_read_mb=(process.io_counters().read_bytes - read) / 2**20,
                                      encoder=type(encoder).__name__, prompt=prompts[0][:80])
            return result
        return wrapper

    def resolve(self):
        totals = {}
        for name, begin, end in self.pending:
            totals[name] = totals.get(name, 0.) + begin.elapsed_time(end)
        self.pending.clear()
        return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()
    config = resolve_experiment(json.loads(args.config.read_text(encoding="utf-8")))
    cfg = harness.load_config(config["harness"])
    args.warmup_chunks = cfg.warmup_chunks
    args.measured_chunks = cfg.measured_chunks
    selected = next((v for v in config.get("variants", []) if v["label"] == args.variant), None)
    pipeline = selected["pipeline"] if selected else config["pipeline"]
    choices = {step["name"]: step for step in pipeline}
    args.backend = choices.get("rope", {}).get("backend", "original")
    args.kv_store = choices.get("kv_store", {}).get("backend", "original")
    forwarded = live_arguments(config)
    args.keep_open = args.keep_open or (config["run"]["mode"] == "live" and config["run"]["keep_open"])
    prompt = forwarded[forwarded.index("--prompt") + 1]
    # Both comparison arms use the fixed shapes of the same video preset.
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.specialize_int = True
    metrics = LiveMetrics(args.warmup_chunks, args.measured_chunks, prompt, config=cfg)
    metadata = harness.metadata() if cfg.metrics.metadata else {}
    metadata["compile_policy"] = "fixed shapes and specialized integer metadata, both variants"
    from wan_dit_lab.offloading.live_prefetch import TracedStreamedRows, calibration_key, summarize_prefetch
    import wan_dit_lab.offloading.offload as offload

    # Instrument only the existing warmup, and leave timed steady execution alone.
    if cfg.metrics.readiness:
        offload.StreamedRows = TracedStreamedRows
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT / "src/wan_dit_lab").rglob("*.py"))}
    for path in (ROOT / "scripts/live/live_video.py", Path(__file__)):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    key = calibration_key(sources, dict(argv=forwarded, pipeline=pipeline, harness=cfg.model_dump()),
                          metadata.get("environment", dict(torch=torch.__version__)))
    calibration = None
    if args.warmup_chunks < 1:
        raise ValueError("At least one warmup chunk is needed for fresh prefetch calibration")
    saved, sampler = False, None
    steady_blocks = []  # streamed blocks of the measured chunks, when metrics.readiness_steady
    stages = StageTimers(metrics) if cfg.metrics.stages else None

    def record_chunk(elapsed, frames, current_prompt):
        nonlocal saved, calibration, sampler
        metrics.record_chunk(elapsed, frames, current_prompt)
        index = len(metrics.chunks) - 1
        if stages:  # the chunk's pixels are on the CPU: its GPU events are complete
            metrics.chunks[-1].update(stages.resolve())
        if calibration and cfg.metrics.readiness_steady:
            for offloader in TracedStreamedRows.instances:
                steady_blocks.extend(dict(row, chunk=index) for row in offloader.drain())
        if len(metrics.chunks) == args.warmup_chunks // 2:
            for offloader in TracedStreamedRows.instances:
                offloader.samples.clear()  # omit initial compilation and the first short chunk
        if len(metrics.chunks) == args.warmup_chunks:
            calibration = dict(key=key, reused=False, sources=sources,
                profiles=[offloader.finish_calibration() for offloader in TracedStreamedRows.instances])
            print(f"Fresh prefetch calibration: {sum(p['consumers'] for p in calibration['profiles'])} "
                  "consumers, fixed two slots", flush=True)
            if cfg.metrics.readiness_steady:
                for offloader in TracedStreamedRows.instances:
                    offloader.samples.clear()  # the warmup's are in the calibration already
                    offloader.tracing = True
            print(f"Measured chunks begin at {time.time():.3f}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            if cfg.metrics.memory:
                sampler = MemorySampler(cfg.memory.sample_interval_s)
                sampler.start()
        if metrics.complete and not saved:
            save("passed")

    def save(status):
        nonlocal saved
        if sampler:
            sampler.close()
        record = dict(status=status, backend=args.backend, pid=os.getpid(),
                      config=config, pipeline=pipeline,
                      kv_store=args.kv_store, calibration=calibration,
                      summary=metrics.summary(partial=status == "incomplete"), metadata=metadata, argv=forwarded,
                      # wall time = perf_counter + clock_offset_s, to line Live up with other processes' logs
                      clock_offset_s=time.time() - time.perf_counter(),
                      steady_prefetch=(dict(summarize_prefetch(steady_blocks), samples=steady_blocks,
                                            scope="measured chunks") if steady_blocks else None),
                      source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in ("scripts/live/live_video.py", "src/wan_dit_lab/models/rope.py",
                                       "src/wan_dit_lab/kernels/rope_triton.py")})
        if sampler:
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            record["memory"] = summarize_memory(sampler.samples)
            record["memory_samples"] = sampler.samples
        if status == "passed" and not record["summary"]["prompt_unchanged"]:
            record["status"] = "prompt_changed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        saved = True
        print(f"Measured {args.variant or args.backend}: "
              f"generation FPS={record['summary'].get('generated_fps', 'disabled')}", flush=True)

    original = ROOT / "scripts/live/live_video.py"
    tree = instrument_live_source(original.read_text(encoding="utf-8"), ticks=cfg.metrics.display)
    namespace = dict(__name__="_measured_live", __file__=str(original),
                     __live_chunk=record_chunk, __live_frame=metrics.record_frame,
                     __live_tick=metrics.record_tick, __live_clock=metrics.clock,
                     __live_done=lambda: metrics.complete and not args.keep_open)
    sys.argv = [str(original), *forwarded]
    with install(pipeline) as compilation:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.compile":
                node.func = ast.Name(id="__compile_model", ctx=ast.Load())
        namespace["__compile_model"] = lambda model: torch.compile(model, **compilation) if compilation else model
        if config["debug"]["dump_graph"] or config["debug"]["recompiles"]:
            torch._logging.set_logs(graph_code=config["debug"]["dump_graph"], recompiles=config["debug"]["recompiles"])
        exec(compile(ast.fix_missing_locations(tree), str(original), "exec"), namespace)
        from wan_dit_lab.runtime.compile_all import compile_model_edges

        namespace["compile_model_edges"] = lambda model: compile_model_edges(model, **compilation) if compilation else None
        namespace["TITLE"] = f"Live · {args.backend} RoPE · {args.kv_store} KV"
        try:
            namespace["main"]()
        finally:
            # closed before the end: what was measured is kept, marked incomplete
            if not saved and calibration and len(metrics.chunks) > args.warmup_chunks:
                save("incomplete")
            elif sampler and not saved:
                sampler.close()
    if not metrics.complete:
        raise SystemExit("Live closed before the comparison completed")


if __name__ == "__main__":
    main()
