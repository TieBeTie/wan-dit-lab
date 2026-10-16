import torch

from wan_dit_lab.checkpoint import load_block_weights, model_config
from wan_dit_lab.layer_norm import LayerNorm
from wan_dit_lab.multi_head_attention import MultiHeadAttention


def feed_forward(dim, ffn_dim):
    return torch.nn.Sequential(torch.nn.Linear(dim, ffn_dim), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(ffn_dim, dim))

class WanBlock(torch.nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps, attention=MultiHeadAttention, ffn=feed_forward, norm=LayerNorm):
        super().__init__()
        self.self_attn = attention(dim, num_heads)
        self.cross_attn = attention(dim, num_heads)             
        self.ffn = ffn(dim, ffn_dim)
        self.norm1 = norm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = norm(dim, eps=eps, elementwise_affine=True)
        self.norm2 = norm(dim, eps=eps, elementwise_affine=False)
        self.modulation = torch.nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, X, time_in, seq_lens=None, grid_sizes=None, freqs=None, context=None, context_lens=None,
                start_frame=0, cache=None):
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (time_in + self.modulation).chunk(6, dim=1)
        X = X + self.self_attn(self.norm1(X) * (1 + scale_attn) + shift_attn, grid_sizes=grid_sizes, freq=freqs,
                               start_frame=start_frame, cache=cache) * gate_attn
        X = X + self.cross_attn(self.norm3(X), context=context)
        X = X + self.ffn(self.norm2(X) * (1 + scale_ffn) + shift_ffn) * gate_ffn
        return X


if __name__ == '__main__':
    config = model_config()
    block = WanBlock(dim=config["dim"], ffn_dim=config["ffn_dim"], num_heads=config["num_heads"], eps=config["eps"])
    block.load_state_dict(load_block_weights(0), strict=True)
