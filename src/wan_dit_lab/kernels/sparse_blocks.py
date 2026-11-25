"""Which key blocks each query block of the joint Sage/RoPE attention runs: the block prediction of SpargeAttn (Zhang
et al. 2025, its cumulative threshold on a compressed attention map), without its self-similarity filter and its
second, in-kernel stage, plus a static mask of allowed frames.

A query block is 128 tokens and a key block 64, the tiles of Sage's sm89 kernel. The attention a block mean of the
queries pays to a block mean of the keys, softmax over the allowed key blocks of q̄ k̄ᵀ / √d + log(tokens of the
block), estimates the share of the attention mass that falls into each key block. The weakest blocks are skipped
while their estimated shares add up to less than skip_mass: skip_mass 0 skips nothing. Forced blocks (the current
chunk, which also holds the last block of the keys, the one the kernel always runs) are never skipped.
"""
import torch

CTA_Q, CTA_K = 128, 64


def keep(q_mean, k_mean, k_tokens, allowed, forced, skip_mass, scale):
    """q_mean [B, H, query blocks, D], k_mean [B, H, key blocks, D], k_tokens [key blocks] (tokens of each block),
    allowed and forced [key blocks] bool -> [B, H, query blocks, key blocks] bool: the key blocks each query block
    runs."""
    if skip_mass == 0:  # nothing is skipped: no estimate is computed at all
        return (allowed | forced).expand(*q_mean.shape[:3], allowed.shape[-1])
    logits = torch.einsum("bhqd,bhkd->bhqk", q_mean.float(), k_mean.float()) * scale + k_tokens.float().log()
    share = logits.masked_fill(~allowed, float("-inf")).softmax(-1)
    ascending = share.sort(-1)
    weakest = ascending.values.cumsum(-1) < skip_mass
    skipped = torch.zeros_like(weakest).scatter(-1, ascending.indices, weakest)
    return (allowed & ~skipped) | forced


def table(kept):
    """kept [B, H, query blocks, key blocks] bool -> (lut, valid) of live::sage_sm89_sparse: the kept key blocks
    of each query block ascending (the rest of the row padded with the last block), and how many there are."""
    blocks = kept.shape[-1]
    index = torch.arange(blocks, device=kept.device)
    order = torch.where(kept, index, blocks + index).sort(-1).values.clamp(max=blocks - 1)
    return order.to(torch.int32).contiguous(), kept.sum(-1, dtype=torch.int32).contiguous()


def block_tokens(length, block, device=None):
    """The tokens of each block of `block` tokens of a sequence of `length`: all full, the last maybe not."""
    starts = torch.arange(0, length, block, device=device)
    return (length - starts).clamp(max=block)
