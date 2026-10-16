from platform import release
import torch
from wan_dit_lab.block import WanBlock
from wan_dit_lab.checkpoint import load_block_weights
from wan_dit_lab.head import Head


class WanModel(torch.nn.Module):
    def __init__(self, dim, eps, ffn_dim, freq_dim, in_dim, num_heads, num_layers, out_dim, text_len, text_dim=4096, conv3d=torch.nn.Conv3d, linear=torch.nn.Linear, block_offloader=None):
        super().__init__()
        self.patch_embedding = conv3d(in_dim, dim, (1, 2, 2), (1, 2, 2), 0)
        self.text_embedding = torch.nn.Sequential(linear(text_dim, dim), torch.nn.GELU(approximate="tanh"), linear(dim, dim))
        self.time_embedding = torch.nn.Sequential(linear(freq_dim, dim), torch.nn.SiLU(), linear(dim, dim))
        self.time_projection = torch.nn.Sequential(torch.nn.SiLU(), linear(dim, dim * 6))
        self.blocks = torch.nn.ModuleList([WanBlock(dim, ffn_dim, num_heads, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, eps, linear)
        self.block_offloader = block_offloader  
        self.out_dim = out_dim
    
    def to(self, device):
        self.patch_embedding.to(device)
        self.text_embedding.to(device)
        self.time_embedding.to(device)
        self.time_projection.to(device)
        self.head.to(device)
        self.block_offloader.start()
        return self


    def forward(self, X, t, context, grid_sizes, freqs, start_frame=0, caches=None):
        # start_frame and caches (one causal.KVCache per block): X is a chunk of a video generated chunk by chunk
        context = self.text_embedding(context)
        X = self.patch_embedding(X).flatten(2).transpose(1, 2)
        omega = 10000 ** (-torch.arange(128, dtype=torch.float64, device=t.device) / 128)
        theta = t.double()[:, None] * omega[None, :]
        t = self.time_embedding(torch.cat([torch.cos(theta), torch.sin(theta)], dim=1).float())
        e0 = self.time_projection(t).unflatten(1, (6, -1))
        for i in range(len(self.blocks)):
            block = self.block_offloader.get_next()
            X = block(X, e0, grid_sizes=grid_sizes, freqs=freqs, context=context, start_frame=start_frame,
                      cache=None if caches is None else caches[i])
            self.block_offloader.release(block)
        X = self.head(X, t)
        f, h, w = grid_sizes[0].tolist() if isinstance(grid_sizes, torch.Tensor) else grid_sizes[0]
        X = X.view(X.shape[0], f, h, w, 1, 2, 2, self.out_dim)
        return X.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(X.shape[0], self.out_dim, f, h * 2, w * 2)
