"""The umT5-XXL text encoder of Wan, taken ready from the reference code.

Its 24 blocks (9.3 GB in bfloat16) go through the GPU by BlockOffloader straight from the memory-mapped
checkpoint; the token embedding table (256384 x 4096, 2.1 GB) stays there too and is read on the CPU.
"""

import copy
import sys
from contextlib import closing
from pathlib import Path

import torch

from wan_dit_lab.models.checkpoint import REFERENCE, checkpoint_dir, mapped_torch_checkpoint
from wan_dit_lab.offloading.offload import BlockOffloader, RowFile

if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
from wan.modules.t5 import umt5_xxl  # noqa: E402
from wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: E402

TEXT_LEN = 512
# sample_neg_prompt of wan/configs/shared_config.py, Wan2.1 upstream
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class TextEncoder:
    """umT5-XXL built once over its mapped checkpoint; each call streams its blocks through the GPU again. A live video
    encodes every new prompt with it, without building the model each time."""

    def __init__(self, device, window=2, ram_window=2, trace=None, resource_cache=None):
        self.tokenizer = HuggingfaceTokenizer(
            name=str(checkpoint_dir() / "google" / "umt5-xxl"), seq_len=TEXT_LEN, clean="whitespace"
        )
        self.encoder = umt5_xxl(encoder_only=True, dtype=torch.bfloat16, device="meta")
        self.encoder.load_state_dict(mapped_torch_checkpoint(checkpoint_dir() / "models_t5_umt5-xxl-enc-bf16.pth"),
                                     assign=True)
        self.encoder.eval()
        self.resource_cache = resource_cache
        if resource_cache is not None:
            resource_cache.reserve_device(
                "t5/norm", sum(p.numel() * p.element_size() for p in self.encoder.norm.parameters())
            )
        try:
            self.norm = self.encoder.norm.to(device)
        except BaseException:
            if resource_cache is not None:
                resource_cache.release_device("t5/norm")
            raise
        self.device, self.window, self.ram_window = device, window, ram_window
        self.trace = trace

    def close(self):
        self.norm = self.encoder = self.tokenizer = None
        if self.resource_cache is not None:
            self.resource_cache.release_device("t5/norm")

    @torch.no_grad()
    def __call__(self, prompts):
        """Embeddings of `prompts`: [len(prompts), 512, 4096], float32, on the CPU.

        A row holds the T5 output for the tokens of its prompt and zeros after them: the reference pipeline cuts
        the output to the prompt length and the reference model pads it back with zeros.
        """
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        offloader = BlockOffloader(self.encoder.blocks, self.window, self.device,
                                  ram_window=self.ram_window, trace=self.trace,
                                  resource_cache=self.resource_cache, resource_prefix="t5/weights")
        with closing(offloader):
            offloader.start()
            x = self.encoder.token_embedding(ids).to(self.device)
            mask = mask.to(self.device)
            for _ in range(len(self.encoder.blocks)):
                block = offloader.get_next()
                x = block(x, mask)
                offloader.release(block)
            x = self.norm(x)
        context = torch.zeros(len(prompts), TEXT_LEN, x.shape[-1])
        for i, length in enumerate(mask.sum(dim=1).tolist()):
            context[i, :length] = x[i, :length].float().cpu()
        return context


class FastTextEncoder:
    """The same umT5-XXL for a live video, where a new prompt stops the generation while it is encoded. Measured at
    480x864: 7.8 s a prompt, of which 5.8 s copying the 8.6 GB of blocks from the mapped checkpoint, 0.4 s computing.

    So: every linear weight but ffn.fc2 in INT8 with a scale per output row (FP8 E4M3 turned some tokens' embeddings
    to a cosine near 0 against bfloat16, and so did INT8 of ffn.fc2), made once at start, all blocks in RAM
    (4.4 GB); per prompt each block is copied to one GPU block in FP8 and unpacked to bfloat16 there; only the tokens
    of the prompt are computed (the reference masks the padding out, so their outputs are the same); the embeddings of
    the last prompts are kept. Output as TextEncoder.

    The blocks go through the GPU by BlockOffloader as TextEncoder's, each a row of bytes in a GPU slot that slot()
    makes a block of: its linear layers unpack their INT8 weights to bfloat16 when called."""

    def __init__(self, device, cache=16, truncate=True, weights_file=None):
        """weights_file: the INT8 blocks kept in this file instead of all of them in RAM (5.6 GB that Live, OBS and
        the tutor need): made from the checkpoint the first time, then read row by row while a prompt is encoded, on
        Windows without the page cache (RowFile), elsewhere mapped. The cost is the read of the file per prompt."""
        self.truncate = truncate  # compute only the prompt's tokens (else all TEXT_LEN, as TextEncoder)
        base = TextEncoder(device)
        self.tokenizer, self.norm, self.device = base.tokenizer, base.norm, device
        self.embedding = base.encoder.token_embedding
        blocks = base.encoder.blocks
        # the structure of a block; its weights take GPU memory (0.37 GB) only while a prompt is encoded: beside a
        # live video's model, cache and CUDA graph at 512x928 there is no room to keep them
        self.template = copy.deepcopy(blocks[0]).to("meta").requires_grad_(False)
        self.linears = [n for n, m in self.template.named_modules() if isinstance(m, torch.nn.Linear)]
        # per block: {name: (int8 weight, scale)} for linears, {name: bf16 tensor} for the rest; laid out in one row of
        # bytes per block (16-byte aligned pieces). ffn.fc2 (wo of T5) stays in bfloat16: its inputs carry the outliers
        # of T5 (Hugging Face keeps "wo" in float32); quantized, a token here and there loses its embedding (a cosine
        # of 0.24 against bfloat16)
        weights = {n + ".weight" for n in self.linears if not n.endswith("ffn.fc2")}
        shapes = [(name, tuple(p.shape)) for name, p in blocks[0].named_parameters()]
        rows = {name: shape[0] for name, shape in shapes if name in weights}
        size = lambda shape, item: -(-int(torch.Size(shape).numel()) * item // 16) * 16
        self.layout, offset = [], 0  # (name, [(offset, shape, dtype), ...]) within a block's row
        for name, shape in shapes:
            parts = [(shape, torch.int8), ((shape[0], 1), torch.bfloat16)] if name in weights else [(shape, torch.bfloat16)]
            placed = []
            for part_shape, dtype in parts:
                placed.append((offset, part_shape, dtype))
                offset += size(part_shape, torch.empty((), dtype=dtype).element_size())
            self.layout.append((name, placed))
        self.per_block, self.blocks = offset, len(blocks)
        total = self.per_block * self.blocks
        path = None if weights_file is None else Path(weights_file)
        self.mapped = path is not None
        self.path = path
        if self.mapped and path.exists() and path.stat().st_size == total:
            self.buffer = None  # mapped for each prompt only (_map): between prompts its pages are the OS's cache
        else:
            self.buffer = torch.empty(total, dtype=torch.uint8)  # the whole of it, kept in RAM unless a file holds it
            with torch.no_grad():
                for b, block in enumerate(blocks):
                    params = dict(block.named_parameters())
                    for name, placed in self.layout:
                        views = [self._view(self.buffer, b * self.per_block + o, sh, dt) for o, sh, dt in placed]
                        if len(views) == 2:
                            w = params[name].to(device, torch.float32)
                            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
                            views[0].copy_(torch.round(w / scale).clamp(-127, 127).to(torch.int8).cpu())
                            views[1].copy_(scale.to(torch.bfloat16).cpu())
                        else:
                            views[0].copy_(params[name].detach().to(torch.bfloat16))
            if self.mapped:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                self.buffer.numpy().tofile(tmp)
                tmp.replace(path)
                self.buffer = None
        del base, blocks
        self.cache, self.capacity = {}, cache

    def _map(self):
        """The file mapped read-only, for one prompt: unmapped after it, its pages leave this process's working set
        and stay in the file cache, the first memory Windows gives to other programs."""
        import numpy as np
        return torch.from_numpy(np.memmap(self.path, dtype=np.uint8, mode="r"))

    @staticmethod
    def _view(buffer, offset, shape, dtype):
        item = torch.empty((), dtype=dtype).element_size()
        return buffer[offset:offset + int(torch.Size(shape).numel()) * item].view(dtype).view(shape)

    def slot(self, row):
        """A block over `row` (a uint8 row of the layout, on the GPU): the INT8 linear layers as Int8Linear, the rest
        bfloat16 parameters, all views of the row."""
        block = copy.deepcopy(self.template)
        for name, placed in self.layout:
            views = [self._view(row, o, sh, dt) for o, sh, dt in placed]
            parent, _, attr = name.rpartition(".")
            owner = block.get_submodule(parent) if parent else block
            if len(views) == 2:  # parent is the linear layer, attr its weight
                path, _, layer = parent.rpartition(".")
                setattr(block.get_submodule(path) if path else block, layer, Int8Linear(*views))
            else:
                setattr(owner, attr, torch.nn.Parameter(views[0], requires_grad=False))
        return block

    @torch.no_grad()
    def __call__(self, prompts):
        key = tuple(prompts)
        if key in self.cache:
            return self.cache[key].clone()
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        n = int(mask.sum(dim=1).max()) if self.truncate else ids.shape[1]
        ids, mask = ids[:, :n], mask[:, :n].to(self.device)
        x = self.embedding(ids).to(self.device)
        if self.mapped and RowFile.available():
            source = dict(rows=RowFile(self.path, self.per_block))
        else:  # the rows mapped, or the buffer in RAM: BlockOffloader's thread copies each into a pinned buffer
            rows = self._map() if self.mapped else self.buffer
            source = dict(packed=rows.view(self.blocks, self.per_block))
        offloader = BlockOffloader([self.template] * self.blocks, 2, self.device, dtype=torch.uint8, slot=self.slot,
                                   once=True, **source)
        with closing(offloader):
            offloader.start()
            for _ in range(self.blocks):
                block = offloader.get_next()
                x = block(x, mask)
                offloader.release(block)
        x = self.norm(x)
        del offloader, source  # the mapping, if any, ends here
        context = torch.zeros(len(prompts), TEXT_LEN, x.shape[-1])
        for i, length in enumerate(mask.sum(dim=1).tolist()):
            context[i, :length] = x[i, :length].float().cpu()
        if len(self.cache) >= self.capacity:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = context.clone()
        torch.cuda.empty_cache()  # the slots back to the GPU's free memory
        return context


class Int8Linear(torch.nn.Module):
    """A linear layer over an INT8 weight with a bfloat16 scale per output row, unpacked to bfloat16 for each call:
    the same numbers as the bfloat16 weight FastTextEncoder used to unpack into."""

    def __init__(self, weight, scale):
        super().__init__()
        self.weight, self.scale = weight, scale

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(torch.bfloat16) * self.scale)


def encode(prompts, device, window=2, ram_window=2):
    """Embeddings of `prompts` by a TextEncoder built for this one call."""
    return TextEncoder(device, window, ram_window)(prompts)
