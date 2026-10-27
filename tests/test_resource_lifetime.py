"""CPU checks of model ownership at the DiT/decoder phase boundary."""

import gc
import importlib
import queue
import weakref
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wan_dit_lab.offloading.offload import BlockOffloader


@pytest.mark.parametrize("decoder", ["none", "wan", "taehv"])
def test_denoiser_owners_end_before_full_vae(monkeypatch, tmp_path, decoder):
    script, args, contexts, events, owners = setup_pipeline(monkeypatch, decoder)
    latents, _, streamed = script.denoise(args, contexts, "cpu", tmp_path, "test", {}, 5, 2, 2)
    gc.collect()
    assert latents.shape == (16, 6, 1, 1)
    assert streamed == (decoder == "taehv")
    assert all(owner() is None for owner in owners)
    assert events[-1] == "offloader closed"
    if streamed:
        assert events.index("decoder loaded") > events.index("chunk 1")
        assert events.count("decoder loaded") == 1
        assert events.count("decoder closed") == 1
        assert events.count("writer closed") == 1
    else:
        assert "decoder loaded" not in events


def test_failure_also_closes_decoder_writer_and_offloader(monkeypatch, tmp_path):
    script, args, contexts, events, _ = setup_pipeline(monkeypatch, "taehv", fail=True)
    with pytest.raises(RuntimeError, match="second chunk"):
        script.denoise(args, contexts, "cpu", tmp_path, "test", {}, 5, 2, 2)
    assert events[-3:] == ["decoder closed", "writer closed", "offloader closed"]


def setup_pipeline(monkeypatch, decoder, fail=False):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/generation"))
    script = importlib.import_module("generate_causal")
    events, owners = [], []

    class Model:
        def __init__(self, **kwargs):
            self.blocks = [object()]
            owners.append(weakref.ref(self))

        def load_state_dict(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

    class Offloader:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            events.append("offloader closed")

    class Stream:
        def __init__(self, model, *args, **kwargs):
            self.model = model  # reproduces the ownership that used to outlive del model
            self.n = 0
            owners.append(weakref.ref(self))

        def next_chunk(self):
            self.n += 1
            if fail and self.n == 2:
                raise RuntimeError("second chunk")
            events.append(f"chunk {self.n}")
            return torch.zeros(16, 3, 1, 1)

    class Decoder:
        def __init__(self, device):
            events.append("decoder loaded")
            owners.append(weakref.ref(self))

        def __call__(self, chunk):
            return torch.zeros(3, 1, 8, 8)

        def close(self):
            events.append("decoder closed")

    class Writer:
        def append_data(self, frame):
            pass

        def close(self):
            events.append("writer closed")

    monkeypatch.setattr(script, "WanModel", Model)
    monkeypatch.setattr(script, "OnDevice", Offloader)
    monkeypatch.setattr(script, "self_forcing_weights", lambda: {})
    monkeypatch.setattr(script, "model_config", lambda: dict.fromkeys(script.KEYS, 1) | {"dim": 12})
    monkeypatch.setattr(script, "rope", lambda *a: torch.zeros(2, 8, 2))
    monkeypatch.setattr(script.causal, "Stream", Stream)
    monkeypatch.setattr(script, "TAEHVStream", Decoder)
    monkeypatch.setattr(script.imageio, "get_writer", lambda *a, **k: Writer())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    args = SimpleNamespace(on_gpu=True, fp8=False, compile=None, seed=0, decoder=decoder,
                           height=16, width=16, latent_frames=6, context_frames=21, sink_frames=0,
                           attention="cudnn", cuda_graph=False, positions="video", context_noise=0,
                           kv_fp8=False, pc_recent=None, cross_cache=False, prompt=["test"])
    return script, args, torch.zeros(1, 2, 4), events, owners


def test_offloader_close_drops_queue_and_slot_references(monkeypatch):
    offloader = object.__new__(BlockOffloader)
    offloader.thread, offloader.device = None, "cpu"
    offloader.gpu = torch.zeros(4)
    offloader.pinned = [torch.zeros(4)]
    offloader.slots = [torch.nn.Parameter(offloader.gpu)]
    offloader.blocks = [torch.nn.Linear(2, 2)]
    offloader.free, offloader.filled = queue.Queue(), queue.Queue()
    offloader.free.put(offloader.pinned[0])
    offloader.filled.put(offloader.pinned[0])
    offloader.loaded = deque([(0, object())])
    owners = [weakref.ref(offloader.gpu), weakref.ref(offloader.pinned[0]),
              weakref.ref(offloader.slots[0]), weakref.ref(offloader.blocks[0])]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    offloader.close()
    offloader.close()  # cleanup is safe after an earlier close
    gc.collect()
    assert all(owner() is None for owner in owners)
