"""Measurements of the real Live loop, without editing its source file."""

import ast
import time


class LiveMetrics:
    def __init__(self, warmup_chunks, measured_chunks, prompt, clock=time.perf_counter, config=None):
        from wan_dit_lab.harness.config import load_config

        self.config = load_config(config)
        if warmup_chunks < 1 or measured_chunks < 1:
            raise ValueError("Positive warmup and measured chunk counts required")
        self.warmup, self.count, self.prompt = warmup_chunks, measured_chunks, prompt
        self.clock, self.chunks, self.frames = clock, [], []
        self.ticks, self.events = [], []  # window ticks (begin, end, queued frames); stage and prompt events

    @property
    def complete(self):
        return len(self.chunks) >= self.warmup + self.count

    def record_chunk(self, elapsed, frame_count, prompt):
        if not self.complete:
            self.chunks.append(dict(index=len(self.chunks), delivery_s=elapsed,
                                    frames=frame_count, prompt=prompt, finished_at=self.clock()))

    def record_frame(self, at_time=None):
        if not self.complete:
            self.frames.append(self.clock() if at_time is None else at_time)

    def record_tick(self, begin, queued):
        if not self.complete:
            self.ticks.append((begin, self.clock(), queued))

    def record_event(self, kind, **fields):
        if not self.complete:
            self.events.append(dict(kind=kind, at=self.clock(), **fields))

    def display_summary(self):
        """Why shown frames lag made ones: tick work adds to the period Tk waits after it (root.after)."""
        from wan_dit_lab.harness import summarize

        steady = self.chunks[self.warmup:self.warmup + self.count]
        start = steady[0]["finished_at"] - steady[0]["delivery_s"]
        ticks = [t for t in self.ticks if t[0] >= start]
        frames = [s for s in self.frames if s >= start]
        gaps = [(b - a) * 1000 for a, b in zip(frames, frames[1:])]
        return dict(ticks=len(ticks), tick_work=summarize([(e - b) * 1000 for b, e, _ in ticks]) if ticks else None,
                    frame_interval=summarize(gaps) if gaps else None,
                    queue_empty_ticks=sum(q == 0 for *_, q in ticks),
                    queue_mean=sum(q for *_, q in ticks) / len(ticks) if ticks else None)

    def summary(self, partial=False):
        """partial: the measured chunks made before the window was closed, at least one."""
        if not self.complete and not (partial and len(self.chunks) > self.warmup):
            raise ValueError("Live comparison is incomplete")
        from wan_dit_lab.harness import stream_summary

        steady = self.chunks[self.warmup:self.warmup + self.count]
        start = steady[0]["finished_at"] - steady[0]["delivery_s"]
        end = steady[-1]["finished_at"]
        displayed = sum(start <= stamp <= end for stamp in self.frames)
        result = stream_summary(dict(chunks=self.chunks), warmup=self.warmup, config=self.config)
        summary = dict(measured_chunks=len(steady), frames=sum(row["frames"] for row in steady),
                    measured_wall_s=end - start, displayed_frames=displayed,
                    prompt_unchanged=all(row["prompt"] == self.prompt for row in self.chunks),
                    raw_chunks=self.chunks,
                    scope="generation includes DiT, decoder and pixel copy; excludes queue backpressure; display is actual frame dequeues")
        if self.config.metrics.fps:
            summary.update(generated_fps=result["steady_fps"], displayed_fps=displayed / (end - start))
        if self.config.metrics.latency:
            summary.update(p50_chunk_ms=result["steady_median_s"] * 1000, p99_chunk_ms=result["steady_p99_s"] * 1000)
        if self.config.metrics.display:
            summary.update(display=self.display_summary(), raw_ticks=self.ticks)
        if self.events:
            summary["events"] = self.events
        return summary


def instrument_live_source(source, ticks=False):
    """Insert telemetry at three verified anchors. Fail if the Live source drifts. With ticks, each tick also
    reports when it began and how many frames were queued (__live_tick, __live_clock)."""
    tree = ast.parse(source)
    counts = dict(chunk=0, frame=0, tick=0)

    class Instrument(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            targets = [ast.unparse(target) for target in node.targets]
            if "state['chunk_s']" in targets:
                counts["chunk"] += 1
                extra = ast.parse("__live_chunk(state['chunk_s'], len(pixels), state['prompt'])\n"
                                  "if __live_done():\n    return").body
                return [node, *extra]
            if "shown['frame']" in targets and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "get_nowait":
                    counts["frame"] += 1
                    return [node, ast.parse("__live_frame()").body[0]]
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == "tick":
                counts["tick"] += 1
                body = node.body
                if ticks:  # before the rescheduling call that ends the tick, so its own wait is not counted
                    body = (ast.parse("__tick_begin = __live_clock()").body + body[:-1]
                            + ast.parse("__live_tick(__tick_begin, frames.qsize())").body + body[-1:])
                node.body = ast.parse("if __live_done():\n    root.destroy()\n    return").body + body
            return node

    tree = Instrument().visit(tree)
    if counts != dict(chunk=1, frame=1, tick=1):
        raise ValueError(f"Live telemetry anchors changed: {counts}")
    return ast.fix_missing_locations(tree)
