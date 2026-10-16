"""The umT5-XXL text encoder of Wan, taken ready from the reference code.

Its 24 blocks (9.3 GB in bfloat16) go through the GPU by BlockOffloader straight from the memory-mapped
checkpoint; the token embedding table (256384 x 4096, 2.1 GB) stays there too and is read on the CPU.
"""

import sys

import torch

from wan_dit_lab.checkpoint import REFERENCE, checkpoint_dir, mapped_torch_checkpoint
from wan_dit_lab.offload import BlockOffloader

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


def encode(prompts, device, window=2, ram_window=2):
    """Embeddings of `prompts`: [len(prompts), 512, 4096], float32, on the CPU.

    A row holds the T5 output for the tokens of its prompt and zeros after them: the reference pipeline cuts
    the output to the prompt length and the reference model pads it back with zeros.
    """
    tokenizer = HuggingfaceTokenizer(
        name=str(checkpoint_dir() / "google" / "umt5-xxl"), seq_len=TEXT_LEN, clean="whitespace"
    )
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    encoder = umt5_xxl(encoder_only=True, dtype=torch.bfloat16, device="meta")
    weights = mapped_torch_checkpoint(checkpoint_dir() / "models_t5_umt5-xxl-enc-bf16.pth")
    encoder.load_state_dict(weights, assign=True)
    encoder.eval()
    offloader = BlockOffloader(encoder.blocks, window, device, ram_window=ram_window)
    offloader.start()
    with torch.no_grad():
        x = encoder.token_embedding(ids).to(device)
        mask = mask.to(device)
        for _ in range(len(encoder.blocks)):
            block = offloader.get_next()
            x = block(x, mask)
            offloader.release(block)
        x = encoder.norm.to(device)(x)
    offloader.stop()
    context = torch.zeros(len(prompts), TEXT_LEN, x.shape[-1])
    for i, length in enumerate(mask.sum(dim=1).tolist()):
        context[i, :length] = x[i, :length].float().cpu()
    return context
