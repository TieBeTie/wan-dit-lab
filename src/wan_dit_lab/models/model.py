from contextlib import nullcontext

import torch

from wan_dit_lab.models.block import WanBlock
from wan_dit_lab.models.head import Head
from wan_dit_lab.offloading.resource_store import StoredSequence


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


    def forward(self, X, t, context, grid_sizes, freqs, start_frame=0, caches=None, context_kvs=None,
                commit_next=False):
        # start_frame and caches (one causal.KVCache per block): X is a chunk of a video generated chunk by chunk;
        # context_kvs: the keys and values of cross attention per block, computed once per prompt (causal.Stream)
        context = self.text_embedding(context) if context_kvs is None else None
        X = self.patch_embedding(X).flatten(2).transpose(1, 2)
        omega = 10000 ** (-torch.arange(128, dtype=torch.float64, device=t.device) / 128)
        theta = t.double()[:, None] * omega[None, :]
        t = self.time_embedding(torch.cat([torch.cos(theta), torch.sin(theta)], dim=1).float())
        e0 = self.time_projection(t).unflatten(1, (6, -1))
        for i in range(len(self.blocks)):
            block = self.block_offloader.get_next()
            stored = isinstance(context_kvs, StoredSequence)
            lease = context_kvs.device_lease(i, X.device) if stored else nullcontext(
                None if context_kvs is None else context_kvs[i]
            )
            with lease as context_kv:
                if context_kv is not None:
                    context_kv = tuple(tensor.to(X.device) for tensor in context_kv)
                if stored:
                    store = context_kvs.store
                    if getattr(store, "cost_model", None) is not None:
                        upcoming, lead = [], 0.
                        for j in range(i, len(self.blocks)):
                            lead += store.cost_model.seconds(f"dit/block/{j}", "compute")
                            if j + 1 < len(self.blocks):
                                upcoming.append((context_kvs.names[j + 1], lead))
                                binding = getattr(caches[j + 1], "_residency", None) if caches else None
                                if binding is not None:
                                    upcoming.append((binding.name, lead))
                        if commit_next and caches:
                            # Commit follows the final write pass. Its independent
                            # KV reads must be loaded before that phase, too.
                            upcoming.extend((cache._residency.name, lead) for cache in caches
                                            if getattr(cache, "_residency", None) is not None)
                        store.prefetch_schedule(upcoming, next_boundary_s=store.cost_model.seconds(
                            f"dit/block/{i}", "compute"))
                    else:
                        context_kvs.prefetch(i + 1)
                if caches is not None and i + 1 < len(caches):
                    binding = getattr(caches[i + 1], "_residency", None)
                    if binding is not None and binding.tier == "auto" and binding.store.cost_model is None:
                        if not binding.store.prefetch_device(binding.name):
                            binding.store.prefetch(binding.name)
                X = block(X, e0, grid_sizes=grid_sizes, freqs=freqs, context=context,
                          start_frame=start_frame, cache=None if caches is None else caches[i],
                          context_kv=context_kv)
            del context_kv
            self.block_offloader.release(block)
        X = self.head(X, t)
        f, h, w = grid_sizes[0].tolist() if isinstance(grid_sizes, torch.Tensor) else grid_sizes[0]
        X = X.view(X.shape[0], f, h, w, 1, 2, 2, self.out_dim)
        return X.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(X.shape[0], self.out_dim, f, h * 2, w * 2)
