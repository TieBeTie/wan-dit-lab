"""Exports Live (preset longlive2 of runtime/live_presets.py) into AOTInductor packages for the Rust host
(native/live). The host keeps the state (the KV caches, the weights in pinned RAM, the prompt, the window); every
computation of a chunk is one of these packages, with no Python in the process that runs them.

    python scripts/conversion/export_live.py                 # all of them, into artifacts/live
    python scripts/conversion/export_live.py --only block_3 decoder

What a chunk is made of, the packages named as in the manifest (live.json):

    text     T5 embeddings [1, 512, 4096] -> text_embedding [1, 512, dim]              once per prompt
    cross    row of a block, text -> keys and values of its cross-attention              once per prompt and block
    pre      latents [C, F, H, W], timestep -> tokens, e0 (6 modulations), t (time embedding)
    block_k  row of a block, tokens, e0, cross keys and values, RoPE of the chunk's frames, its KV cache
             -> tokens; writes the chunk's keys and values into the cache at k chunks: one package per k, since
             attention reads at + n tokens
    post     tokens, t, latents x, c -> x - c * flow: a step of the sampler (UniPC with 2 steps is two of them)
    decoder  one latent frame and the memory of TAEHV's MemBlocks -> 4 frames of pixels (uint8) and the memory

The weights of a block are an input (its row of the W4A4 pack), so that one package serves all of them; the host
streams the rows as StreamedRows does. The time rows of RoPE (the only part that moves with the frame) are an
input too; the host reads them from time_rope.bin. The operators of the W4A4 layers and of Sage's attention are
live::* (native/ops), which the host loads beside the packages.

Every package is checked against the eager model it is exported from before it is written, on the inputs the
previous package gives, and the check goes into the manifest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import copy
import json
import time

import torch
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from wan_dit_lab.kernels.operators import load as load_operators
from wan_dit_lab.generation import causal
from wan_dit_lab.kernels import sage_kv
from wan_dit_lab.kernels.cache_fp8 import store_kv_fp8
from wan_dit_lab.models.checkpoint import self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.text_encoder import TEXT_LEN
from wan_dit_lab.models.vae import TAEHVStream
from wan_dit_lab.quantization import w4a4
from wan_dit_lab.runtime.live_presets import KEYS, MODELS, POSITIONS, SCENE_PREFIX, rope_table


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
CROSS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]  # causal.ATTENTION["sage"], cross-attention


def relative(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


class Exporter:
    def __init__(self, out, only):
        self.out, self.only, self.report = out, only, {}

    def __call__(self, name, module, inputs, reference=None, tolerance=0.01, dynamic=None, eager_rounding=False):
        """Exports `module` on `inputs` into <out>/<name>.pt2 and checks it against `reference` (eager by default)
        and against torch.compile of it, the code Live runs now: relative L2 of every output, below `tolerance` for
        one of the two (Inductor alone moves a block 1-3% off eager). Returns the eager outputs, inputs of the next
        package. Inputs and outputs are contiguous: a package reads its inputs with the strides of the example it was
        compiled on, whatever strides the host's tensor has, and the host passes it the outputs of other packages."""
        inputs = tuple(i.contiguous() if isinstance(i, torch.Tensor) else i for i in inputs)
        module = Contiguous(module)
        with torch.no_grad():
            eager = module(*inputs) if reference is None else reference
        if self.only and name not in self.only:
            return eager
        start = time.perf_counter()
        with torch.no_grad(), sdpa_kernel(CROSS, set_priority=True):
            program = torch.export.export(module, inputs, strict=False,
                                         dynamic_shapes=None if dynamic is None else (dynamic,))  # forward(*inputs)
            # the weights stay out of the package (packaged in, they hang or fail on Windows): in <name>.safetensors,
            # loaded by the host as constants it owns (AOTIModelPackageLoader::load_constants, user managed)
            path = torch._inductor.aoti_compile_and_package(program, package_path=str(self.out / f"{name}.pt2"),
                                                            inductor_configs={"aot_inductor.package_constants_in_so": False,
                                                                              "emulate_precision_casts": eager_rounding})
        compiled = time.perf_counter() - start
        runner = torch._inductor.aoti_load_package(path)
        values = {**program.state_dict, **program.constants}
        constants = {fqn: values[fqn].detach().contiguous() for fqn in runner.get_constant_fqns()}
        save_file(constants, str(self.out / f"{name}.safetensors"))
        runner.load_constants(constants, check_full_update=True, user_managed=True)
        with torch.no_grad():
            got = runner(*inputs)
        try:
            with torch.no_grad(), sdpa_kernel(CROSS, set_priority=True):
                inductor = torch.compile(module)(*inputs)
        except Exception as e:  # TAEHV of Wan2.2 is loaded from a path, which Dynamo cannot import again
            print(f"{name}: no torch.compile to compare with ({type(e).__name__})", flush=True)
            inductor = eager
        listed = lambda outputs: outputs if isinstance(outputs, (list, tuple)) else [outputs]
        got, expected, inductor = listed(got), listed(eager), listed(inductor)
        errors = [relative(g, e) for g, e in zip(got, expected) if g.is_floating_point()]
        to_inductor = [relative(g, e) for g, e in zip(got, inductor) if g.is_floating_point()]
        # pixels: the largest difference of a value, of 255
        exact = [int((g.int() - e.int()).abs().max()) for g, e in zip(got, expected) if not g.is_floating_point()]
        self.report[name] = dict(seconds=round(compiled, 1), relative_l2=errors, relative_l2_compile=to_inductor,
                                 pixels_off=exact)
        print(f"{name}: {compiled:.0f} s, relative L2 to eager {['%.2e' % e for e in errors]}, to torch.compile "
              f"{['%.2e' % e for e in to_inductor]}, pixels off by {exact}", flush=True)
        assert all(min(a, b) < tolerance for a, b in zip(errors, to_inductor)), (name, errors, to_inductor)
        assert all(e <= 2 for e in exact), (name, exact)
        return eager


class Contiguous(torch.nn.Module):
    """`module` with its outputs made contiguous."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs):
        out = self.module(*inputs)
        return tuple(o.contiguous() for o in out) if isinstance(out, (list, tuple)) else out.contiguous()


class Bound(torch.nn.Module):
    """A block over a row of the W4A4 pack given as an input: the template block's layers are rebound to views of
    the row on every call (w4a4.slot, without copying the module)."""

    def __init__(self, template, layout, row_bytes):
        super().__init__()
        self.block = w4a4.slot(torch.empty(row_bytes, dtype=torch.uint8, device="meta"), template, layout)
        self.entries = {e["name"]: e for e in layout}
        self.params = []  # its parameters as plain attributes, rebound like the packed layers' parts
        for name, _ in list(self.block.named_parameters()):
            parent, _, attr = name.rpartition(".")
            owner = self.block.get_submodule(parent) if parent else self.block
            del owner._parameters[attr]
            self.params.append((owner, attr, name))
        self.packed = [(m, name) for name, m in self.block.named_modules() if hasattr(m, "parts")]

    def bind(self, row):
        def view(name):
            e = self.entries[name]
            return row[e["offset"]:e["offset"] + e["bytes"]].view(getattr(torch, e["dtype"])).view(e["shape"])

        for owner, attr, name in self.params:
            setattr(owner, attr, view(name))
        for m, name in self.packed:
            groups = (w4a4.FFN[0], w4a4.FFN[1]) if isinstance(m, w4a4.PackedW4A4GeluMLP) else (name,)
            m.parts = [view(f"{g}.{p}") for g in groups for p in w4a4.PARTS]
        return self.block


class Cache:
    """A full or filling FP8 KV cache as causal.KVCache keeps it for Sage, with no state of its own: the chunk's keys
    and values are written at `at` (live::store_kv_fp8 of cache_fp8.py) and attention reads [0, at) and the chunk."""

    def __init__(self, at, k, ks, v, vs):
        self.at, self.buffers = at, (k, ks, v, vs)

    def attend(self, Q, K, V):
        store_kv_fp8(K, V, *self.buffers, self.at)
        k, ks, v, vs = self.buffers
        return sage_kv.attend(Q, k, ks, v, vs, self.at, K, V)


class BlockStep(torch.nn.Module):
    def __init__(self, bound, at, grid, hw):
        super().__init__()
        self.bound, self.at, self.grid = bound, at, grid
        self.register_buffer("hw", hw)  # height and width of the RoPE table, [2, rows, pairs]

    def forward(self, row, X, e0, cross_k, cross_v, time_rope, k, ks, v, vs):
        rows = self.hw.shape[1]
        time_rope = torch.nn.functional.pad(time_rope, (0, 0, 0, rows - time_rope.shape[1]))
        freqs = torch.cat([time_rope, self.hw], dim=-1)
        with torch.autocast("cuda", dtype=DTYPE):
            return self.bound.bind(row)(X, e0, grid_sizes=(self.grid,), freqs=freqs,
                                        cache=Cache(self.at, k, ks, v, vs), context_kv=(cross_k, cross_v))


class Cross(torch.nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = bound

    def forward(self, row, text):
        with torch.autocast("cuda", dtype=DTYPE):
            return self.bound.bind(row).cross_attn.keys_values(text)


class Text(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.text_embedding

    def forward(self, context):
        with torch.autocast("cuda", dtype=DTYPE):
            return self.embedding(context)


class T5Block(torch.nn.Module):
    """A block of umT5-XXL over a row of FastTextEncoder's INT8 file (artifacts/t5_int8_fast.bin) given as an input:
    the INT8 layers' weights and scales and the bfloat16 parameters rebound to views of the row on every call. The
    prompt's tokens only, as FastTextEncoder computes them: their number is a dynamic size of the package (all 512
    with the padding masked out moved the embeddings 2-12% off FastTextEncoder's, and the videos off the prompt:
    CLIP score 28.9 against 30.8, live_quality.py)."""

    def __init__(self, encoder):
        super().__init__()
        self.layout, self.view = encoder.layout, encoder._view
        self.block = encoder.slot(torch.empty(encoder.per_block, dtype=torch.uint8, device="meta"))
        for name, _ in list(self.block.named_parameters()):
            parent, _, attr = name.rpartition(".")
            owner = self.block.get_submodule(parent) if parent else self.block
            del owner._parameters[attr]
            setattr(owner, attr, None)

    def forward(self, row, x, mask):
        for name, placed in self.layout:
            views = [self.view(row, o, shape, dtype) for o, shape, dtype in placed]
            parent, _, attr = name.rpartition(".")
            if len(views) == 2:  # parent is the Int8Linear of the weight
                linear = self.block.get_submodule(parent)
                linear.weight, linear.scale = views
            else:
                setattr(self.block.get_submodule(parent) if parent else self.block, attr, views[0])
        return self.block(x, mask)


class T5Final(torch.nn.Module):
    """FastTextEncoder after its blocks: the last norm, [1, tokens, 4096] float32; the host puts zeros after them."""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x, mask):
        return self.norm(x).float() * mask[..., None].float()


def embedding_offset(checkpoint, name, table):
    """Where the bytes of the tensor `table` (`name` in `checkpoint`, a zip of torch.save) start in the file: the
    one member of its size, stored uncompressed; checked against the tensor itself."""
    import zipfile
    size = table.numel() * table.element_size()
    with zipfile.ZipFile(checkpoint) as z:
        members = [i for i in z.infolist() if i.file_size == size and "/data/" in i.filename]
        assert len(members) == 1 and members[0].compress_type == zipfile.ZIP_STORED, (name, members)
        header = members[0].header_offset
    with open(checkpoint, "rb") as f:
        f.seek(header)
        local = f.read(30)
        offset = header + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
        f.seek(offset + 1234 * table.shape[1] * 2)
        row = torch.frombuffer(bytearray(f.read(table.shape[1] * 2)), dtype=table.dtype)
    assert torch.equal(row, table[1234]), "the token embedding is not where the zip says"
    return offset


class Pre(torch.nn.Module):
    """WanModel.forward before its blocks."""

    def __init__(self, model):
        super().__init__()
        self.patch_embedding, self.time_embedding = model.patch_embedding, model.time_embedding
        self.time_projection = model.time_projection

    def forward(self, x, t):
        with torch.autocast("cuda", dtype=DTYPE):
            X = self.patch_embedding(x[None]).flatten(2).transpose(1, 2)
            omega = 10000 ** (-torch.arange(128, dtype=torch.float64, device=t.device) / 128)
            theta = t.double()[:, None] * omega[None, :]
            t = self.time_embedding(torch.cat([torch.cos(theta), torch.sin(theta)], dim=1).float())
            return X, self.time_projection(t).unflatten(1, (6, -1)), t


class Post(torch.nn.Module):
    """WanModel.forward after its blocks, and a step of the sampler: x - c flow, flow in float32 (causal.unipc)."""

    def __init__(self, model, grid):
        super().__init__()
        self.head, self.out_dim, self.grid = model.head, model.out_dim, grid

    def forward(self, X, t, x, c):
        f, h, w = self.grid
        with torch.autocast("cuda", dtype=DTYPE):
            X = self.head(X, t)
        X = X.view(1, f, h, w, 1, 2, 2, self.out_dim).permute(0, 7, 1, 4, 2, 5, 3, 6)
        flow = X.reshape(self.out_dim, f, h * 2, w * 2).float()
        return x - c * flow


class Decoder(torch.nn.Module):
    """TAEHVStream on one latent frame: the memory of each MemBlock (its input for the frame before) in and out, the
    frames as uint8 pixels [T, H, W, 3] as Live made them."""

    def __init__(self, stream):
        super().__init__()
        self.decoder, self.memblock, self.patch = stream.decoder, stream.memblock, stream.patch_size

    def forward(self, latent, *memory):
        x = latent.permute(1, 0, 2, 3).to(torch.float16)
        new, j = [], 0
        for block in self.decoder:
            if isinstance(block, self.memblock):
                new.append(x[-1:])
                x = block(x, torch.cat([memory[j], x[:-1]]))
                j += 1
            else:
                x = block(x)
        if self.patch > 1:
            x = torch.nn.functional.pixel_shuffle(x, self.patch)
        video = x.float().mul(2).sub(1).clamp(-1, 1)  # [T, 3, H, W]
        pixels = ((video.permute(0, 2, 3, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return pixels, *new


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="artifacts/live/<height>x<width> by default")
    parser.add_argument("--height", type=int, default=None, help="pixels, a multiple of 32; the preset's by default")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--rope-interpolate", action=argparse.BooleanOptionalAction, default=None,
                        help="height and width of RoPE stretched over the trained grid (the preset's choice by default)")
    parser.add_argument("--stream-blocks", type=int, default=None,
                        help="blocks the host streams from pinned RAM (the preset's 24 by default; 30 where the cache is large)")
    parser.add_argument("--only", nargs="*", default=None, help="names of the packages to export (all by default)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = dict(MODELS["longlive2"])
    for key in ("height", "width", "rope_interpolate", "stream_blocks"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    assert spec["height"] % 32 == 0 and spec["width"] % 32 == 0, "a token is 32x32 pixels"
    if args.out is None:  # one directory per resolution: live.exe --size picks it
        args.out = ROOT / "artifacts" / "live" / f"{spec['height']}x{spec['width']}"
    args.out.mkdir(parents=True, exist_ok=True)
    load_operators()
    height, width, chunk, stride = spec["height"], spec["width"], spec["chunk"], spec["vae_stride"]
    lat_h, lat_w = height // stride, width // stride
    grid = (chunk, lat_h // 2, lat_w // 2)
    tokens = grid[1] * grid[2]  # of a latent frame
    n = chunk * tokens  # of a chunk
    max_tokens = (spec["context_frames"] - chunk) * tokens  # causal.Stream: window less the chunk
    sink_tokens = spec["sink_frames"] * tokens
    export = Exporter(args.out, set(args.only) if args.only else None)
    g = torch.Generator(DEVICE).manual_seed(args.seed)

    config = json.loads(spec["config"].read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = WanModel(**{key: config[key] for key in KEYS})
    model.load_state_dict(self_forcing_weights(spec["checkpoint"]), assign=True)
    model.eval()
    rows, layout = w4a4.mapped_pack(model.blocks, spec["pack"], spec["checkpoint"])
    template = copy.deepcopy(model.blocks[0]).to("meta")
    bound = Bound(template, layout, rows.shape[1])
    for m in (model.patch_embedding, model.text_embedding, model.time_embedding, model.time_projection, model.head):
        m.to(DEVICE)
    model.blocks = None
    heads, dim = config["num_heads"], config["dim"]
    d = dim // heads

    freqs = rope_table(spec, d, DEVICE)
    time_pairs = freqs.shape[-1] - 2 * (freqs.shape[-1] // 3)
    freqs[:, :, :time_pairs].contiguous().cpu().numpy().tofile(args.out / "time_rope.bin")
    hw = freqs[:, :max(grid), time_pairs:].contiguous()

    row = rows[0].to(DEVICE)
    context = torch.randn(1, TEXT_LEN, config.get("text_dim", 4096), device=DEVICE, generator=g) * 0.1
    text = export("text", Text(model), (context,))
    cross_k, cross_v = export("cross", Cross(bound), (row, text))
    x = torch.randn(config["out_dim"], *grid[:1], lat_h, lat_w, device=DEVICE, generator=g)
    t = torch.tensor([999.0], device=DEVICE)
    X, e0, t_emb = export("pre", Pre(model), (x, t))
    time_rope = freqs[:, 3 * chunk:4 * chunk, :time_pairs].contiguous()  # the rows of the fourth chunk
    cache = [torch.randn(1, max_tokens + n, heads, d, device=DEVICE, generator=g).to(torch.float8_e4m3fn),
             torch.rand(1, max_tokens + n, heads, 1, device=DEVICE, generator=g) * 0.05]
    cache = cache + [cache[0].clone(), cache[1].clone()]
    steps = max_tokens // n + 1  # at = 0, n, ..., max_tokens
    for k in range(steps):
        X_next = export(f"block_{k}", BlockStep(bound, k * n, grid, hw),
                        (row, X, e0, cross_k, cross_v, time_rope, *[c.clone() for c in cache]))
    c = torch.tensor([0.16674446821107836], device=DEVICE)
    export("post", Post(model, grid), (X_next, t_emb, x, c))

    stream = TAEHVStream(DEVICE, checkpoint=spec["taehv"])
    decoder = Decoder(stream)
    latent = torch.randn(config["out_dim"], 1, lat_h, lat_w, device=DEVICE, generator=g)
    memory, probe = [], latent.permute(1, 0, 2, 3).to(torch.float16)
    with torch.no_grad():  # the shape of each MemBlock's input, from a pass with zeros as memory
        for block in stream.decoder:
            if isinstance(block, stream.memblock):
                memory.append(torch.zeros_like(probe[:1]))
                probe = block(probe, torch.cat([memory[-1], probe[:-1]]))
            else:
                probe = block(probe)
    memory = [torch.randn(m.shape, device=DEVICE, generator=g).to(torch.float16) for m in memory]
    export("decoder", decoder, (latent, *memory))

    # T5: its blocks over rows of FastTextEncoder's file, then the norm; the token embedding read in place
    from wan_dit_lab.models.checkpoint import checkpoint_dir
    from wan_dit_lab.models.text_encoder import FastTextEncoder
    t5_rows = ROOT / "artifacts" / "t5_int8_fast.bin"
    encoder = FastTextEncoder(DEVICE, weights_file=t5_rows)
    ids, mask = encoder.tokenizer(["A lone oak tree on a grassy hill sways in the wind"], return_mask=True,
                                  add_special_tokens=True)
    length = int(mask.sum())
    x5 = encoder.embedding(ids[:, :length]).to(DEVICE)
    mask = mask[:, :length].to(DEVICE)
    # 512 itself is a size export specializes on elsewhere (a guard tokens != 512): prompts of 511 tokens at most
    prompt_tokens = torch.export.Dim("tokens", min=2, max=TEXT_LEN - 1)
    t5_row = encoder._map().view(encoder.blocks, encoder.per_block)[0].to(DEVICE)
    x5 = export("t5_block", T5Block(encoder), (t5_row, x5, mask), dynamic=(None, {1: prompt_tokens}, {1: prompt_tokens}),
                eager_rounding=True)
    # FastTextEncoder runs eager: the packages round to bfloat16 where eager does (Inductor keeps float32 inside a
    # fused kernel), else 24 blocks of T5's outliers move the embeddings 2-7% off it
    export("t5_final", T5Final(encoder.norm), (x5, mask), dynamic=({1: prompt_tokens}, {1: prompt_tokens}),
           eager_rounding=True)
    t5_checkpoint = checkpoint_dir() / "models_t5_umt5-xxl-enc-bf16.pth"
    table = encoder.embedding.weight
    t5 = dict(tokenizer=str(checkpoint_dir() / "google" / "umt5-xxl" / "tokenizer.json"), rows=str(t5_rows),
              row_bytes=encoder.per_block, blocks=encoder.blocks, embedding=str(t5_checkpoint),
              embedding_offset=embedding_offset(t5_checkpoint, "token_embedding.weight", table),
              vocabulary=table.shape[0])

    # the sampler: LongLive 2.0's UniPC over 2 steps is linear in x and the two flows (checked to 2e-16 in float64):
    # x1 = x - c0 f0, x0 = x1 - c1 f1; probed here from the scheduler itself
    scheduler = causal_unipc(spec["steps"], causal.SHIFT)
    manifest = dict(
        preset="longlive2", height=height, width=width, fps=spec["fps"], chunk=chunk, latent=[config["out_dim"], chunk,
                                                                                           lat_h, lat_w],
        grid=list(grid), tokens_per_frame=tokens, chunk_tokens=n, max_tokens=max_tokens, sink_tokens=sink_tokens,
        heads=heads, head_dim=d, dim=dim, text_len=TEXT_LEN, text_dim=config.get("text_dim", 4096), blocks=len(rows),
        block_steps=steps, positions=POSITIONS, time_pairs=time_pairs, shot_rope_offset=8,
        scene_prefix=SCENE_PREFIX, pack=str(spec["pack"]), row_bytes=rows.shape[1],
        stream_blocks=spec["stream_blocks"], timesteps=scheduler["timesteps"], sampler=scheduler["coefficients"],
        decoder_memory=[list(m.shape) for m in memory], decoder_trim=3, t5=t5, checks=export.report)
    previous = args.out / "live.json"
    if export.only and previous.exists():  # the checks of the packages not exported this time stay
        manifest["checks"] = {**json.loads(previous.read_text(encoding="utf-8")).get("checks", {}), **export.report}
    previous.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"written: {args.out}", flush=True)


def causal_unipc(steps, shift):
    """Timesteps and coefficients c_k of x_{k+1} = x_k - c_k flow_k of LongLive 2.0's UniPC sampler (causal.unipc)."""
    import importlib.util
    path = ROOT / "reference" / "LongLive" / "LongLive2.0" / "wan_5b" / "utils" / "fm_solvers_unipc.py"
    module_spec = importlib.util.spec_from_file_location("fm_solvers_unipc", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    def run(x, flows, every=None):
        s = module.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        s.set_timesteps(steps, device="cpu", shift=shift)
        for i, t in enumerate(s.timesteps):
            x = s.step(flows[i], t, x, return_dict=False)[0]
            if every is not None:
                every.append(x.clone())
        return x, [float(t) for t in s.timesteps]

    one, zero = torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    _, timesteps = run(zero.clone(), [zero] * steps)
    coefficients = []
    for k in range(steps):  # the change of x_{k+1} with flow_k, the others zero
        trace = []
        run(zero.clone(), [one if j == k else zero for j in range(steps)], trace)
        coefficients.append(-float(trace[k]))
    # linear, and of that form: x_k enters with 1, earlier flows not again
    x, f = torch.randn(1 + steps, 7, dtype=torch.float64).split([1, steps])
    expected = x[0] - sum(c * f[k] for k, c in enumerate(coefficients))
    assert float((run(x[0].clone(), list(f))[0] - expected).abs().max()) < 1e-12, "UniPC is not two Euler steps here"
    return dict(timesteps=timesteps, coefficients=coefficients)


if __name__ == "__main__":
    main()
