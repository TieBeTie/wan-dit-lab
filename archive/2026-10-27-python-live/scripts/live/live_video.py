"""A live video from Self-Forcing on my WanModel: chunks are generated without end and shown in a window, and a new
prompt sent over a local socket turns the video towards it, from the frames already shown.

Three threads. The main one shows the frames, at the rate they are made (16 fps at most), so that the video slows
down rather than stops. One generates a chunk, decodes it with TAEHV and queues its 12 frames. One takes prompts
from 127.0.0.1:8765 and encodes them with T5 while the video goes on; the newest replaces the prompt between two
chunks, and the cache of the frames already made carries the scene over to it.

    PYTHONPATH=src python scripts/live/live_video.py --prompt "A lone oak tree on a grassy hill sways in the wind"
    python scripts/live/live_video.py --send "Snow starts to fall on the oak tree"
    python scripts/live/live_video.py --send "A busy night market in Taipei" --scene
    python scripts/live/live_video.py --send "A red fox in a snowy forest" --new

Three ways a prompt comes in: plain text goes on with the scene (--switch, swap); "/scene ..." is a cut of
LongLive 2.0, a new shot that still sees the old one (same world, another shot); "/new ..." starts a new video, the
cache emptied.

The model is the 2-step LongLive-2.0-5B by default (--model longlive2): Wan2.2-TI2V-5B, its blocks from the SVDQuant
W4A4 pack (artifacts/longlive2_5b_s2_blocks_fused.w4a4, convert_w4a4.py --fused), all of them on the GPU, attention
by SageAttention 2 straight over the FP8 cache. --model longlive2_fp8 is the 4-step model with its linear layers in
FP8, 12 of its blocks streamed from RAM. A prompt continues the scene (--switch, swap by default); a prompt that begins with /scene (or is sent
with --scene) cuts to a new one, as LongLive 2.0 does a shot: "shot" in causal.SWITCHES, the prompt prefixed with
"The scene transitions. ". --model longlive1 is LongLive-1.3B on Wan2.1.

Below the video, a prompt box: Enter sends what is typed in it. Esc or closing the window stops it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import copy
import os
import json
import math
import queue
import socket
import threading
import time

import tkinter as tk

import numpy as np

# the cache, the streamed slots and the activations leave the allocator fragmented: with fixed segments 600 MB
# stay reserved and unused, and 480x832 runs out of memory (E17)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image, ImageDraw, ImageTk

from wan_dit_lab.generation import causal
from wan_dit_lab.models.checkpoint import LONGLIVE, ROOT, model_config, self_forcing_weights
from wan_dit_lab.quantization.fp8 import to_fp8
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice, StreamedOnDevice, StreamedRows
from wan_dit_lab.runtime.compile_all import compile_model_edges
from wan_dit_lab.models.positions import rope_at, stretch
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.text_encoder import FastTextEncoder, TextEncoder
from wan_dit_lab.models.vae import TAEHVStream

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.generation.generate import KEYS  # noqa: E402

PORT = 8765
SCENE = "/scene"  # a prompt that begins with it is a cut of LongLive 2.0 (shot): a new shot that still sees the old one
NEW = "/new"  # a prompt that begins with it starts a new video: the cache emptied
SCENE_PREFIX = "The scene transitions. "  # how LongLive 2.0 marks a cut in its prompts (utils/dataset.py)
# what differs between the models; the rest of Live is the same
MODELS = {
    # 2 steps, W4A4 blocks (2.8 GB) and an FP8 cache fit the GPU together; 480x864 (token grid 15x27, the aspect of
    # the trained 22x40) with the RoPE positions of height and width stretched over the trained grid and the shift of
    # training, the recipe of the blind tests E27 and E28
    "longlive2": dict(config=ROOT / "checkpoints" / "Wan2.2-TI2V-5B" / "config.json",
                      checkpoint=ROOT / "checkpoints" / "LongLive-2.0-5B-S2" / "model_te.pt",
                      pack=ROOT / "artifacts" / "longlive2_5b_s2_blocks_fused.w4a4", steps=2, attention="sage",
                      cross_cache=True, taehv=ROOT / "reference" / "taehv" / "taew2_2.pth", fps=24, chunk=8,
                      vae_stride=16, context_frames=32, sink_frames=8, sampler="unipc", native=(704, 1280),
                      # 24 of the 30 W4A4 blocks streamed from pinned RAM (StreamedRows) and no CUDA graph: 5.1 GB on
                      # the card instead of 7.5, 13.0 fps instead of 13.3, room for OBS and a browser (the graph gave
                      # 4% and held 0.5 GB of its own pool)
                      height=480, width=864, kv_fp8=True, stream_blocks=24, rope_dtype="float32",
                      rope_interpolate=True, area_shift=False, fast_text=True, cuda_graph=False),
    "longlive2_fp8": dict(config=ROOT / "checkpoints" / "Wan2.2-TI2V-5B" / "config.json",
                      checkpoint=ROOT / "checkpoints" / "LongLive-2.0-5B" / "model_bf16.pt",
                      taehv=ROOT / "reference" / "taehv" / "taew2_2.pth", fps=24, chunk=8, vae_stride=16,
                      context_frames=32, sink_frames=8, sampler="unipc", native=(704, 1280), height=480, width=832,
                      # 480x832: at 384x672 the old scene comes back after a cut (E16). There the 30 FP8 blocks do
                      # not all fit beside the cache: 12 stream from pinned RAM (8 run out of memory, E17), and the
                      # cache is FP8 (32 frames in bfloat16 do not fit beside the weights)
                      kv_fp8=True, stream_blocks=12, pack=None, steps=None, attention="cudnn", cross_cache=False,
                      rope_dtype="float64", rope_interpolate=False, area_shift=True, fast_text=False, cuda_graph=False),
    "longlive1": dict(config=None, checkpoint=LONGLIVE, taehv=None, fps=16, chunk=3, vae_stride=8, context_frames=12,
                      sink_frames=3, sampler="dmd", native=(480, 832), height=352, width=608, kv_fp8=False,
                      stream_blocks=0, pack=None, steps=None, attention="cudnn", cross_cache=False,
                      rope_dtype="float64", rope_interpolate=False, area_shift=True, fast_text=False, cuda_graph=False),
}
FPS = 16  # of the model chosen, set in main
POSITIONS = 1 << 16  # latent frames RoPE numbers: at 4 frames each and 16 fps, over an hour of video
TITLE = "live video (Esc to stop)"


def send(prompt, scene=False, new=False):
    prompt = f"{NEW} {prompt.strip()}" if new else f"{SCENE} {prompt.strip()}" if scene else prompt
    with socket.create_connection(("127.0.0.1", PORT), timeout=5) as connection:
        connection.sendall((prompt.strip() + "\n").encode("utf-8"))


class Prompts:
    """Prompts from the socket, encoded by T5 in a thread of their own; of those waiting, only the newest is
    encoded. take() gives (text, embeddings) once, after it is ready."""

    def __init__(self, encoder, gpu):
        self.encoder = encoder
        # the GPU between chunks: T5 streams its blocks to the GPU, and beside the model and its cache at 512x928
        # they do not fit while a chunk is being made; so a prompt is encoded between two chunks
        self.gpu = gpu
        self.waiting = threading.Event()  # set while a prompt waits for the GPU: the next chunk waits for it
        self.wanted = queue.Queue()
        self.ready, self.lock = None, threading.Lock()
        threading.Thread(target=self.listen, daemon=True).start()
        threading.Thread(target=self.encode, daemon=True).start()

    def listen(self):
        server = socket.create_server(("127.0.0.1", PORT))
        while True:
            connection, _ = server.accept()
            with connection:
                text = connection.makefile(encoding="utf-8").readline().strip()
            if text:
                print(f"prompt received: {text}", flush=True)
                self.wanted.put(text)

    def encode(self):
        while True:
            text = self.wanted.get()
            while not self.wanted.empty():  # several came while T5 was busy: the newest one only
                text = self.wanted.get()
            scene = "new" if text.startswith(NEW) else "shot" if text.startswith(SCENE) else None
            if scene == "shot":
                text = SCENE_PREFIX + text[len(SCENE):].strip()
            elif scene == "new":
                text = text[len(NEW):].strip()
            start = time.perf_counter()
            self.waiting.set()
            with self.gpu:
                self.waiting.clear()
                torch.cuda.empty_cache()  # what the chunk left reserved goes back for T5
                context = self.encoder([text])
            print(f"prompt encoded in {time.perf_counter() - start:.1f} s"
                  f"{' (new video)' if scene == 'new' else ' (new shot)' if scene else ''}", flush=True)
            with self.lock:
                self.ready = (text, context, scene)

    def take(self):
        with self.lock:
            ready, self.ready = self.ready, None
        return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="the first prompt")
    parser.add_argument("--send", help="send this prompt to a running live video and exit")
    parser.add_argument("--scene", action="store_true", help="with --send: a cut of LongLive 2.0 (a new shot that sees the old)")
    parser.add_argument("--new", action="store_true", help="with --send: a new video, the cache emptied")
    parser.add_argument("--model", choices=list(MODELS), default="longlive2")
    parser.add_argument("--height", type=int, default=None, help="pixels; the model's default if not given")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="weights of the Self-Forcing format instead of the model's (checkpoint.SELF_FORCING: the reference)")
    # LongLive is trained on 9 recent + 3 sink latent frames (LongLive1.0), LongLive 2.0 on 32 with a sink of 8
    parser.add_argument("--context-frames", type=int, default=None, help="latent frames attention sees, chunk included")
    parser.add_argument("--stream-blocks", type=int, default=None,
                        help="blocks kept in pinned RAM and copied to the GPU ahead of use (even; 0: all on the GPU)")
    parser.add_argument("--scale", type=float, default=1.0, help="of the window over the video")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--right", action="store_true", help="put the window in the right half of the screen")
    parser.add_argument("--sink-frames", type=int, default=None, help="latent frames at the start kept in the cache")
    parser.add_argument("--positions", choices=["video", "window"], default="video",
                        help="RoPE time position of a frame: in the video, or its place in the attention window")
    parser.add_argument("--kv-fp8", action="store_true", help="keys and values of the cache in FP8, half the memory")
    parser.add_argument("--pc-recent", type=int, default=None, help="Participative Compression (generate_causal.py)")
    parser.add_argument("--cross-cache", action="store_true", help="cross-attention keys and values once per prompt")
    parser.add_argument("--switch", default="swap", help="how a new prompt comes in: " + ", ".join(causal.SWITCHES))
    parser.add_argument("--blend-chunks", type=int, default=4, help="chunks of a blend switch")
    parser.add_argument("--forcing-kv", type=Path, default=None,
                        help="Forcing-KV: a JSON of static heads (scripts/experiments/forcing_kv_heads.py); they attend to the sink, "
                             "the newest frame and the chunk only")
    parser.add_argument("--write-pass", action=argparse.BooleanOptionalAction, default=True,
                        help="a pass at t = 0 writes each chunk to the cache, as trained; --no-write-pass keeps the keys "
                             "and values of the last step instead: a third less work (13 -> 18 fps at 512x928)")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None,
                        help="replay the pass of a chunk from one CUDA graph once the cache is full (the preset's choice by default)")
    parser.add_argument("--rope-interpolate", action=argparse.BooleanOptionalAction, default=None,
                        help="stretch the RoPE positions of height and width over the trained grid (the preset's choice by default)")
    parser.add_argument("--attention", choices=["cudnn", "efficient", "sage"], default=None,
                        help="the kernel of self-attention over the cache (sage: INT8 SageAttention)")
    args = parser.parse_args()
    if args.send:
        send(args.send, args.scene, args.new)
        return
    if not args.prompt:
        parser.error("--prompt (or --send) is required")
    global FPS
    spec = MODELS[args.model]
    FPS = spec["fps"]
    for name in ("height", "width", "context_frames", "sink_frames", "checkpoint", "stream_blocks", "attention"):
        if getattr(args, name) is None:
            setattr(args, name, spec[name])
    # the shift of 5 of training, or scaled with the side of the frame (docs/resolution-schedule-2026-10-17.md); the
    # blind test E27 kept 5 for LongLive 2
    shift = causal.SHIFT * (math.sqrt(args.height * args.width / (spec["native"][0] * spec["native"][1]))
                            if spec["area_shift"] else 1.0)

    device = torch.device("cuda")
    # T5 in INT8, only the prompt's tokens: 1-2 s a prompt instead of 7.8 (FastTextEncoder); its blocks in a file
    # mapped read-only, one block pinned at a time, so that its 5.6 GB do not stay locked away from the browser, OBS
    encoder = (FastTextEncoder(device, weights_file=ROOT / "artifacts" / "t5_int8_fast.bin") if spec["fast_text"]
               else TextEncoder(device))
    prompt, context = args.prompt, encoder([args.prompt])
    gpu = threading.Lock()  # one GPU job at a time: a chunk, or the encoding of a prompt
    prompts = Prompts(encoder, gpu)

    config = json.loads(spec["config"].read_text(encoding="utf-8")) if spec["config"] else model_config()
    with torch.device("meta"):
        model = WanModel(**{key: config[key] for key in KEYS})
    model.load_state_dict(self_forcing_weights(args.checkpoint), assign=True)
    model.eval()
    if spec["pack"]:  # the blocks from the W4A4 pack, each row copied to the GPU once; the bfloat16 ones are dropped
        from wan_dit_lab.quantization import w4a4
        rows, layout = w4a4.mapped_pack(model.blocks, spec["pack"], args.checkpoint)
        like = copy.deepcopy(model.blocks[0]).to("meta")
        n, k = len(model.blocks), args.stream_blocks
        streamed = [int((j + 0.5) * n / k) for j in range(k)] if k else []  # spread over the model, distinct
        # the streamed ones stay rows in pinned RAM (StreamedRows), the others go to the GPU once
        model.blocks = torch.nn.ModuleList([copy.deepcopy(like) if i in streamed else w4a4.slot(row.to(device), like, layout)
                                            for i, row in enumerate(rows)])
    else:  # one block at a time to the GPU and into FP8: 5B in bfloat16 (10 GB) does not fit 8 GB at once
        for block in model.blocks:
            to_fp8(block.to(device, torch.bfloat16))
    n, k = len(model.blocks), args.stream_blocks
    streamed = [int((j + 0.5) * n / k) for j in range(k)] if k else []  # spread over the model, distinct
    if k and spec["pack"]:  # W4A4 rows through two GPU rows, so that Live leaves room for OBS and a browser
        from functools import partial
        model.block_offloader = StreamedRows(model.blocks, device, streamed, rows, partial(w4a4.slot, like=like, layout=layout))
    else:
        model.block_offloader = StreamedOnDevice(model.blocks, device, streamed) if k else OnDevice(model.blocks, device, dtype=None)
    model.to(device)
    modules = model.block_offloader.modules() if k else model.blocks  # what forward runs on: kept blocks and slots
    forward = torch.compile(type(model.blocks[0]).forward)  # one compiled forward shared by all the blocks
    for block in modules:
        block.forward = lambda *a, block=block, **kw: forward(block, *a, **kw)
    compile_model_edges(model)  # the layers around the blocks too (inductor)
    d = config["dim"] // config["num_heads"]
    tables = [rope(POSITIONS, d - 4 * (d // 6)), rope(POSITIONS, 2 * (d // 6)), rope(POSITIONS, 2 * (d // 6))]
    stretch_rope = spec["rope_interpolate"] if args.rope_interpolate is None else args.rope_interpolate
    if stretch_rope:  # height and width spread over the positions of the trained grid (E27, E28)
        cell = 2 * spec["vae_stride"]
        scales = stretch([s // cell for s in spec["native"]], [args.height // cell, args.width // cell])
        tables[1:] = [rope_at(POSITIONS, 2 * (d // 6), s) for s in scales]
    freqs = torch.cat(tables, dim=-1).to(device, getattr(torch, spec["rope_dtype"]))  # float64 as the reference Wan
    stream = causal.Stream(model, context.to(device), freqs, args.height, args.width,
                           torch.Generator(device).manual_seed(args.seed), max_frames=args.context_frames,
                           sink_frames=args.sink_frames, positions=args.positions, kv_fp8=args.kv_fp8 or spec["kv_fp8"],
                           pc_recent=args.pc_recent, cross_cache=args.cross_cache or spec["cross_cache"],
                           attention=args.attention, shift=shift, chunk=spec["chunk"], vae_stride=spec["vae_stride"],
                           sampler=spec["sampler"], steps=spec["steps"],
                           # the whole pass of a chunk replayed from one CUDA graph once the cache is full. At 512x928
                           # the DiT takes 2.33 s a chunk either way (2.42 s with the decoder, 2.52 s without the graph):
                           # the GPU is busy all the time, at its 100 W limit. The 1.71 s measured first was of a graph
                           # without self-attention: SageAttention's sm89 kernels went to the default stream and were not
                           # recorded; patched since to launch on the current stream.
                           cuda_graph=spec["cuda_graph"] if args.cuda_graph is None else args.cuda_graph,
                           write_pass=args.write_pass,
                           static_heads=json.loads(args.forcing_kv.read_text(encoding="utf-8"))["static"]
                           if args.forcing_kv else None)
    decoder = TAEHVStream(device, **(dict(checkpoint=spec["taehv"]) if spec["taehv"] else {}))
    print(f"{args.model}: {args.height}x{args.width}, shift {shift:.3f}, window {args.context_frames}, "
          f"sink {args.sink_frames}, chunk {spec['chunk']}, steps {stream.steps}, attention {args.attention}, "
          f"blocks {'W4A4' if spec['pack'] else 'FP8'}", flush=True)

    frames = queue.Queue(maxsize=4 * 4 * stream.chunk)  # four chunks ahead at most
    state = {"prompt": prompt, "chunk_s": None, "running": True}

    def generate():
        while state["running"]:
            new = prompts.take()
            if new is not None:
                state["prompt"], ctx, scene = new
                state["pending"] = None
                stream.set_prompt(ctx.to(device), scene or args.switch, args.blend_chunks)
            while prompts.waiting.is_set():  # a prompt waits for the GPU: it goes first (a Lock is not fair)
                time.sleep(0.01)
            start = time.perf_counter()
            with gpu:
                video = decoder(stream.next_chunk())  # [3, T, H, W] in [-1, 1]
            pixels = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            state["chunk_s"] = time.perf_counter() - start
            for frame in pixels:
                frames.put(frame)
            print(f"chunk {stream.start // stream.chunk}: {state['chunk_s']:.2f} s", flush=True)

    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    size = (int(args.width * args.scale), int(args.height * args.scale))
    root = tk.Tk()
    root.title(TITLE)
    # the window can be resized: the video fills it, scaled with Lanczos interpolation, its proportions kept
    # in the middle of the screen, or in its right half with --right
    x, y = ((root.winfo_screenwidth() // 2, 0) if args.right else
            ((root.winfo_screenwidth() - size[0]) // 2, max(0, (root.winfo_screenheight() - size[1] - 44) // 2)))
    root.geometry(f"{size[0]}x{size[1] + 44}+{x}+{y}")
    # the prompt box: Enter sends what is typed (or dictated with Win+H) to T5, as a prompt over the socket would
    entry = tk.Entry(root, font=("Segoe UI", 13))
    entry.pack(side="bottom", fill="x", padx=6, pady=6)
    label = tk.Label(root, bg="black")
    label.pack(fill="both", expand=True)
    entry.focus_set()

    def clipboard_keys(event):
        # Tk ties Ctrl+V, C, X to the Latin letters: with another keyboard layout they do nothing. The physical
        # key (its keycode) is the same whatever the layout.
        action = {86: "<<Paste>>", 67: "<<Copy>>", 88: "<<Cut>>"}.get(event.keycode)
        if action and event.keysym.lower() not in ("v", "c", "x"):
            entry.event_generate(action)
            return "break"
        if event.keycode == 65:  # Ctrl+A: all the text
            entry.select_range(0, "end")
            entry.icursor("end")
            return "break"

    entry.bind("<Control-KeyPress>", clipboard_keys)
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Paste", command=lambda: entry.event_generate("<<Paste>>"))
    menu.add_command(label="Clear", command=lambda: entry.delete(0, "end"))
    entry.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def submit(event):
        text = entry.get().strip()
        if text:
            prompts.wanted.put(text)
            state["pending"] = text
            entry.delete(0, "end")

    entry.bind("<Return>", submit)
    root.bind("<Escape>", lambda event: root.destroy())
    shown = {"frame": np.zeros((args.height, args.width, 3), np.uint8), "text": "warming up: T5, compiling the model"}

    def tick():
        # show at the rate frames are made, so that the video goes on smoothly instead of in bursts
        made = FPS if state["chunk_s"] is None else 4 * stream.chunk / state["chunk_s"]
        fps = min(FPS, 0.95 * made)
        try:
            shown["frame"] = frames.get_nowait()
            shown["text"] = f"{state['prompt'][:100]}   |   {fps:.1f} fps, chunk {state['chunk_s']:.2f} s"
            if state.get("pending"):
                shown["text"] += f"   |   next, T5 encoding: {state['pending'][:60]}"
        except queue.Empty:
            pass
        k = min(max(label.winfo_width(), 2) / args.width, max(label.winfo_height(), 2) / args.height)
        size = (max(1, int(args.width * k)), max(1, int(args.height * k)))  # the largest that fits the window
        image = Image.fromarray(shown["frame"]).resize(size, Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((10, size[1] - 26), shown["text"])
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((10, size[1] - 26), shown["text"], fill=(255, 255, 255))
        label.image = ImageTk.PhotoImage(image)  # kept on the label: Tk shows it only while Python holds it
        label.configure(image=label.image)
        # the next frame is due 1/fps after this one was due, not 1/fps after this tick's work: waiting the whole
        # period after the work showed 10.7 fps for 13.6; after a stall the schedule restarts instead of bursting
        now = time.perf_counter()
        shown["due"] = max(shown.get("due", now) + 1 / fps, now - 0.5 / fps)
        root.after(max(1, int((shown["due"] - now) * 1000)), tick)

    root.after(0, tick)
    root.mainloop()
    state["running"] = False

if __name__ == "__main__":
    main()
