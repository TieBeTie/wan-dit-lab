"""One isolated, fully decoded video under an explicit resource lifetime plan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import functools
import gc
import hashlib
import json
import math
import time

import imageio.v2 as imageio
import imageio_ffmpeg
import torch

from wan_dit_lab.generation import causal
from wan_dit_lab.harness import WorkloadRun
from wan_dit_lab.harness.config import located
from wan_dit_lab.harness.memory import MemorySampler, inventory
from wan_dit_lab.json_files import write_json  # noqa: E402
from wan_dit_lab.models.checkpoint import model_config, self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.positions import rope_at, stretch
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.text_encoder import TextEncoder
from wan_dit_lab.models.vae import TAEHVStream
from wan_dit_lab.offloading.offload import BlockOffloader
from wan_dit_lab.offloading.offload_trace import OffloadTrace
from wan_dit_lab.offloading.resource_cache import ResourceCache
from wan_dit_lab.offloading.resource_cost import ResourceCosts, calibration_from_events
from wan_dit_lab.offloading.resource_schedule import Node, Resource, ResourcePlan
from wan_dit_lab.offloading.resource_store import StoredSequence, TensorStore, validate_placements
from wan_dit_lab.runtime.compile_all import compile_model_edges

PROCESS_START = time.perf_counter()


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()



def run(config, case, policy, output):
    global PROCESS_START
    PROCESS_START = case.get("process_start_perf", PROCESS_START)
    benchmark = WorkloadRun("offloading", config, case, policy)
    config = benchmark.config
    output.mkdir(parents=True, exist_ok=True)
    common, device = config["common"], torch.device("cuda:0")
    placements = validate_placements(policy.get("resources", {}))
    total_vram = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        config["execution"]["resource_budgets"]["vram_gib"] * 2**30 / total_vram, device
    )
    # another model of the Wan family (LongLive 2.0 on Wan2.2-TI2V-5B): its config.json, chunk, VAE stride, TAEHV
    chunk = common.get("chunk", causal.CHUNK)
    spec = (json.loads(Path(common["model_config"]).read_text(encoding="utf-8")) if common.get("model_config")
            else model_config())
    count = case.get("latent_frames", common["latent_frames"]) // chunk
    save_frames = case.get("saved_video_frames", common["saved_video_frames"])
    # several prompts: the video goes from one to the next every `switch_every` chunks, changed the
    # `switch` way (causal.SWITCHES, joined by "+")
    prompts = case.get("prompts") or [case["prompt"]]
    every = case.get("switch_every") or -(-count // len(prompts))
    switch = dict(way=case.get("switch", "swap"), blend_chunks=case.get("blend_chunks", 4),
                  decay_mu=case.get("decay_mu", (0.1, 0.7)))
    monitor = MemorySampler(benchmark.harness.memory.sample_interval_s)
    traces = {label: OffloadTrace(label) for label in ("t5", "dit")}
    record = dict(
        case=case,
        policy=policy,
        chunks=[],
        switches=[],
        resource_events=[],
        inventory=[],
        started_wall=time.time(),
        compile_mode=common["compile"],
        resource_placements=placements,
    )
    store_options = {}
    managed = placements["mode"] == "managed"
    if managed:
        store_options = dict(
            vram_bytes=int(placements["vram_cache_gib"] * 2**30),
            workspace_bytes=int(placements["working_ram_mib"] * 2**20),
            vram_workspace_bytes=int(placements["working_vram_mib"] * 2**20),
            min_system_ram_bytes=int(placements["system_ram_reserve_gib"] * 2**30),
            rss_limit_bytes=(int(placements["process_rss_limit_gib"] * 2**30)
                             if placements["process_rss_limit_gib"] is not None else None),
            min_disk_free_bytes=int(placements["disk_reserve_gib"] * 2**30),
            prefetch=placements["prefetch"],
        )
        costs = policy.get("resource_cost_model")
        if costs:
            samples = json.loads(located(costs["calibration"]).read_text(encoding="utf-8"))
            store_options["cost_model"] = ResourceCosts(samples["samples"], costs["profile"])
    backing = (ResourceCache if managed else TensorStore)(
        output / "resource_backing",
        ram_bytes=int(placements["backing_ram_gib"] * 2**30),
        disk_bytes=int(placements["backing_disk_gib"] * 2**30),
        staging_bytes=int(placements["staging_mib"] * 2**20),
        **store_options,
    )
    record["baseline_device"] = monitor.nvml.memory()
    live = dict(written=0, generated=0, first_frame_s=None, previous_delivery=None)
    categories = {
        "t5": "weights",
        "dit": "weights",
        "decoder": "weights",
        "stream": "state",
        "position": "data",
        "context": "data",
        "latents": "data",
    }

    def resource(name, create, destroy=lambda value: None):
        def acquire(plan):
            started = time.perf_counter()
            monitor.active_stage = f"acquire_{name}"
            print(f"acquire {name}", flush=True)
            value = create(plan)
            record["resource_events"].append(
                dict(
                    event="acquire",
                    resource=name,
                    t=time.perf_counter() - PROCESS_START,
                    duration_s=time.perf_counter() - started,
                )
            )
            return value

        def release(value):
            destroy(value)
            if name == "t5":
                gc.collect()
                torch.cuda.empty_cache()
            record["resource_events"].append(
                dict(event="release", resource=name, t=time.perf_counter() - PROCESS_START)
            )

        return Resource(name, acquire, release)

    def load_t5(plan):
        window = policy["t5"]
        return TextEncoder(
            device, window=window["gpu"], ram_window=window["ram"], trace=traces["t5"],
            **(dict(resource_cache=backing) if managed else {}),
        )

    def load_dit(plan):
        keys = (
            "dim",
            "eps",
            "ffn_dim",
            "freq_dim",
            "in_dim",
            "num_heads",
            "num_layers",
            "out_dim",
            "text_len",
        )
        with torch.device("meta"):
            model = WanModel(**{key: spec[key] for key in keys})
        # policy "checkpoint": another checkpoint of the Self-Forcing format (scripts/conversion/convert_longlive.py)
        from wan_dit_lab.models.checkpoint import SELF_FORCING
        source = Path(policy["checkpoint"]) if policy.get("checkpoint") else SELF_FORCING
        model.load_state_dict(self_forcing_weights(source) if policy.get("checkpoint") else self_forcing_weights(),
                              assign=True)
        model.eval()
        if not managed:
            model.blocks.to(dtype=torch.bfloat16)
        window = policy["dit"]
        packed, slot = None, None
        if policy.get("packed_dit_weights"):
            from wan_dit_lab.quantization.weight_pack import mapped, mapped_fp8, pack_format
            if pack_format(policy["packed_dit_weights"]).startswith("block_w4a4_"):  # SVDQuant W4A4 (wan_dit_lab.quantization.w4a4)
                from wan_dit_lab.quantization import w4a4
                packed, layout = w4a4.mapped_pack(model.blocks, policy["packed_dit_weights"], source)
                slot = functools.partial(w4a4.slot, like=model.blocks[0], layout=layout)
            elif pack_format(policy["packed_dit_weights"]) == "block_fp8_v1":  # linear weights already in FP8
                from wan_dit_lab.quantization.fp8 import fp8_slot
                packed, layout = mapped_fp8(model.blocks, policy["packed_dit_weights"], source)
                slot = functools.partial(fp8_slot, like=model.blocks[0], layout=layout)
            else:
                packed = mapped(model.blocks, policy["packed_dit_weights"], source)  # the pack must be of this checkpoint
        offloader = BlockOffloader(
            model.blocks, window["gpu"], device, ram_window=window["ram"], trace=traces["dit"],
            packed=packed, slot=slot,
            **(dict(dtype=torch.uint8 if slot is not None else torch.bfloat16, resource_cache=backing,
                    resource_prefix="dit/weights") if managed else
               dict(dtype=torch.uint8) if slot is not None else {}),
        )
        model.block_offloader = offloader
        try:
            if managed:
                static = sum(p.numel() * p.element_size() for name, p in model.named_parameters()
                             if not name.startswith("blocks."))
                backing.reserve_device("dit/static_weights", static)
            model.to(device)
            if common.get("collect_act_amax"):  # calibration of W4A4 smoothing: hooks, so no compile
                from wan_dit_lab.quantization.w4a4 import ActStats
                act_stats["stats"] = ActStats(offloader, offloader.slots, len(model.blocks))
            if common.get("linear_fp8"):  # W8A8 on the slots: quantized on every call, the slot changes
                from wan_dit_lab.quantization.fp8 import to_dynamic_fp8
                for block in offloader.slots:
                    to_dynamic_fp8(block, rows=common["linear_fp8"] == "row")
            if common["compile"]:
                forward = torch.compile(type(offloader.slots[0]).forward, mode=common["compile"])
                for block in offloader.slots:
                    block.forward = functools.partial(forward, block)
                compile_model_edges(model, common["compile"])  # the layers around the blocks too
        except Exception:
            offloader.close()
            if managed:
                backing.release_device("dit/static_weights")
            raise
        return model

    def close_dit(model):
        model.block_offloader.close()
        if managed:
            backing.release_device("dit/static_weights")

    def positions(plan):
        d = spec["dim"] // spec["num_heads"]
        tables = [rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))]
        if common.get("rope_interpolate_from"):
            # Position Interpolation of height and width: the token grid of this resolution spread over the
            # positions of the grid the model is trained on (rope_interpolate_from, pixels), fractional positions;
            # time stays as it is
            cell = 2 * common.get("vae_stride", 8)
            trained = [size // cell for size in common["rope_interpolate_from"]]
            now = [common["height"] // cell, common["width"] // cell]
            scales = stretch(trained, now, common.get("rope_interpolate_strength", 1.0))  # 1: the whole stretch
            tables[1:] = [rope_at(1024, 2 * (d // 6), s) for s in scales]
        return torch.cat(tables, dim=-1).to(device, getattr(torch, common.get("rope_dtype", "float64")))  # float64
        # as the reference Wan; float32 is our speed-up, 3 ms less of a 79 ms block at 704x1280 (E23)

    def stream(plan):
        context = backing.get("data/context", device) if managed else plan["context"]["tensor"].to(device)
        return causal.Stream(
            plan["dit"],
            context,
            plan["position"],
            common["height"],
            common["width"],
            torch.Generator(device).manual_seed(case["seed"]),
            max_frames=common["context_frames"],
            sink_frames=common["sink_frames"],
            positions=common["positions"],
            attention=common["attention"],
            context_noise=common["context_noise"],
            cross_cache=common["cross_cache"],
            pc_recent=common["pc_recent"],
            shift=common["shift"],
            resource_store=backing,
            state_tier=placements["self_kv"],
            cross_tier=placements["cross_kv"],
            history_tier=placements["history"],
            # committed inside attention only after a write pass; without it the last step is committed after it
            commit_in_attention=policy.get("commit_in_attention", False) and common.get("write_pass", True),
            chunk=chunk,
            vae_stride=common.get("vae_stride", 8),
            sampler=common.get("sampler", "dmd"),
            shot_rope_offset=common.get("multi_shot_rope_offset", 8),
            steps=common.get("steps"),
            kv_fp8=common.get("kv_fp8", False),
            write_pass=common.get("write_pass", True),
            static_heads=json.loads(Path(common["forcing_kv"]).read_text(encoding="utf-8"))["static"]
            if common.get("forcing_kv") else None,
            # tokens of a frame at the trained resolution over those now (Jin et al. scaling of attention)
            attention_entropy_ratio=(math.prod(s // (2 * common.get("vae_stride", 8)) for s in common["attention_entropy_from"])
                                     / math.prod(s // (2 * common.get("vae_stride", 8)) for s in (common["height"], common["width"]))
                                     if common.get("attention_entropy_from") else None),
        )

    def decoder(plan):
        decoder = TAEHVStream("cpu" if placements["decoder_weights"] != "vram" else device,
                              **(dict(checkpoint=Path(common["taehv"])) if common.get("taehv") else {}))
        try:
            decoder.configure_offloading(
                backing, device, placements["decoder_weights"], placements["decoder_state"]
            )
        except BaseException:
            decoder.close()
            raise
        return decoder

    resources = [
        resource("t5", load_t5, lambda x: x.close()),
        resource("context", lambda p: {}, lambda x: backing.delete("data/context") if managed else None),
        resource("dit", load_dit, close_dit),
        resource("position", positions),
        resource("stream", stream, lambda x: x.close()),
        resource("latents", lambda p: StoredSequence(backing, "latents", placements["latents"])
                 if "resources" in policy else [],
                 lambda x: x.close() if isinstance(x, StoredSequence) else None),
        resource("decoder", decoder, lambda x: x.close()),
        resource(
            "writer",
            lambda p: imageio.get_writer(output / "video.mp4", fps=common["fps"], quality=8),
            lambda x: x.close(),
        ),
    ]
    node_times = []
    act_stats = {}

    def before(node, plan):
        backing.node = node.name
        monitor.active_stage = node.name
        live["node_start"] = time.perf_counter()
        for trace in traces.values():
            trace.node = node.name
        if managed and node.name.startswith("denoise_"):
            for name in ("vae/weights", "vae/state"):
                if not backing.prefetch_device(name):
                    backing.prefetch(name)

    def after(node, plan):
        node_times.append(dict(node=node.name, duration_s=time.perf_counter() - live["node_start"]))
        if case.get("calibration_probe") and managed and node.name == case.get("calibration_probe_node", "decode_0"):
            if "stream" not in plan.values:
                raise RuntimeError("Calibration must run before the stream and its KV bindings are released")
            for _ in range(case.get("calibration_probe_repeats", 1)):
                backing.calibrate_idle_bindings()
        if case.get("calibration_checkpoint") and managed:
            write_json(output / "resource_calibration.partial.json", dict(
                samples=calibration_from_events(backing.events), node=node.name))
        snapshot = monitor.snapshot(node.name)
        # Inventory is a storage audit, not a claim of tracking internal kernel workspaces.
        if node.name == "text" or node.name in ("decode_0", "decode_7", f"decode_{count - 1}"):
            roots = [
                (categories[name], name, value)
                for name, value in plan.values.items()
                if name in categories
            ]
            roots.sort(key=lambda item: ("weights", "state", "data", "activations").index(item[0]))
            for key, entry in backing.entries.items():
                if entry["tier"] == "ram":
                    category = "weights" if key == "vae/weights" else (
                        "data" if key.startswith(("history/", "latents/", "data/")) else "state"
                    )
                    roots.append((category, f"backing/{key}", entry["value"]))
            record["inventory"].append(
                dict(node=node.name, memory=snapshot, storages=inventory(roots))
            )
        write_json(
            output / "progress.json",
            dict(node=node.name, chunks=count, completed_chunks=len(record["chunks"])),
        )

    def encode(plan):
        contexts = plan["t5"](prompts)  # [len(prompts), 512, 4096]
        context = contexts[:1].clone()  # its own storage: the cache does not take a view of a larger one
        if managed:
            backing.put("data/context", context)
        else:
            plan["context"]["tensor"] = context
        # the prompts after the first stay on the CPU, outside the managed cache (no calibration for them),
        # 8 MiB each, until their switch
        plan["context"]["prompts"] = contexts
        torch.cuda.synchronize()
        record["t5_trace"] = traces["t5"].resolve()
        record["t5_copy_count"] = len(traces["t5"].copies)
        record["context_hash"] = tensor_hash(context)
        if len(prompts) > 1:
            record["context_hashes"] = [tensor_hash(contexts[j:j + 1]) for j in range(len(prompts))]
        record["t5_block_bytes"] = sum(
            p.numel() * p.element_size() for p in plan["t5"].encoder.blocks[0].parameters()
        )

    nodes = [Node("text", ("t5", "context"), encode)]
    for index in range(count):
        name = f"chunk_{index}"
        resources.append(resource(name, lambda p: {}))
        categories[name] = "activations"

        def denoise(plan, index=index, name=name):
            if index and index % every == 0 and index // every < len(prompts):
                j = index // every
                started = time.perf_counter()
                context = plan["context"]["prompts"][j:j + 1].to(device)
                plan["stream"].set_prompt(context, switch["way"], switch["blend_chunks"],
                                          tuple(switch["decay_mu"]))
                torch.cuda.synchronize()
                record["switches"].append(dict(chunk=index, prompt=j, way=switch["way"],
                                               duration_s=time.perf_counter() - started))
                print(f"prompt {j + 1} ({switch['way']}) at chunk {index}", flush=True)
            x = plan["stream"].next_chunk()
            cpu = x.detach().cpu()
            plan[name]["tensor"] = x
            plan["latents"].append(cpu)
            plan[name]["latent_hash"] = tensor_hash(cpu)
            record["dit_block_bytes"] = plan["dit"].block_offloader.block_bytes

        def decode(plan, index=index, name=name):
            started = time.perf_counter()
            decoded = plan["decoder"](plan[name]["tensor"])
            frames = (
                ((decoded.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
            )
            decoded_hash = hashlib.sha256(frames.tobytes()).hexdigest()
            for frame in frames:
                if live["written"] < save_frames:
                    plan["writer"].append_data(frame)
                    live["written"] += 1
                    if live["first_frame_s"] is None:
                        live["first_frame_s"] = time.perf_counter() - PROCESS_START
            live["generated"] += len(frames)
            now = time.perf_counter()
            record["chunks"].append(
                dict(
                    index=index,
                    latent_hash=plan[name]["latent_hash"],
                    decoded_hash=decoded_hash,
                    frames=len(frames),
                    delivery_s=None
                    if live["previous_delivery"] is None
                    else now - live["previous_delivery"],
                    decode_write_s=now - started,
                )
            )
            live["previous_delivery"] = now
            print(f"chunk {index + 1}/{count} frames={live['written']}", flush=True)

        nodes.append(
            Node(
                f"denoise_{index}",
                ("context", "dit", "position", "stream", "latents", name)
                + (("decoder",) if managed else ()),
                denoise,
            )
        )
        nodes.append(Node(f"decode_{index}", ("decoder", "writer", name), decode))

    def save(plan):
        archive = plan["latents"]
        if placements["latents"] in ("disk", "auto"):
            chunks = []
            for index, key in enumerate(archive.names):
                relative = f"latents/{index:05d}.pt"
                backing.export(key, output / relative)
                chunks.append(relative)
            write_json(output / "latents.json", dict(format="chunked_latents_v1", dim=1, chunks=chunks))
        else:
            torch.save(torch.cat(list(archive), dim=1), output / "latents.pt")

    nodes.append(Node("save", ("latents", "writer"), save))
    record["graph"] = [dict(node=node.name, uses=node.uses) for node in nodes]
    record["managed_resources"] = categories | {"writer": "code"}
    plan = ResourcePlan(resources, nodes, before=before, after=after)
    monitor.start()
    try:
        with torch.no_grad():
            plan.run()
        # Writer closes in the plan before this timestamp.
        record["full_run_s"] = time.perf_counter() - PROCESS_START
        if act_stats:
            torch.save({k: v.cpu() for k, v in act_stats["stats"].amax.items()}, common["collect_act_amax"])
            record["act_amax"] = dict(path=common["collect_act_amax"], layers=len(act_stats["stats"].amax))
        torch.cuda.synchronize()
        record["dit_trace"] = traces["dit"].resolve()
        record["dit_copy_count"] = len(traces["dit"].copies)
        gc.collect()
        torch.cuda.empty_cache()
        record.update(
            first_frame_s=live["first_frame_s"],
            written_frames=live["written"],
            generated_frames=live["generated"],
            node_times=node_times,
        )
    finally:
        try:
            backing.close()
        finally:
            monitor.close()
            record["tensor_resource_events"] = backing.events
            if managed:
                cost_samples = calibration_from_events(backing.events)
                for row in record.get("dit_trace", []):
                    cost_samples.setdefault(f"dit/block/{row['block']}", {}).setdefault(
                        "compute", []).append(row["compute_s"])
                write_json(output / "resource_calibration.json", dict(
                    format="resource_cost_v1", samples=cost_samples,
                    scope="Observed end-to-end costs; filesystem cache state unknown"))
            record["backing_peak_bytes"] = backing.peak
            if managed:
                record["managed_vram_peak_bytes"] = backing.gpu_peak
                record["weight_staging_peak_bytes"] = backing.host_staging_peak
                record["pinned_staging_peak_bytes"] = backing.pinned_peak
            record["memory_samples"] = monitor.samples
            benchmark.finish(record)
            write_json(output / "record.json", record)
    benchmark.validate(record)
    if live["written"] != min(save_frames, count * chunk * 4 - 3):
        raise RuntimeError("Wrong saved frame count")
    counted_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(output / "video.mp4"))
    if counted_frames != live["written"] or abs(duration - counted_frames / common["fps"]) > 0.02:
        raise RuntimeError("Encoded MP4 count/duration differs from delivered frames")
    write_json(output / "complete.json", dict(status="complete"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    run(read(args.config), read(args.case), read(args.policy), Path(args.output))


if __name__ == "__main__":
    main()
