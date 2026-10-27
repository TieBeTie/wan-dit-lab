
import torch

from wan_dit_lab.models.rms_norm import RMSNorm
from wan_dit_lab.models.rope import rope_apply


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()      
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.o = torch.nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
    
    def keys_values(self, C):
        # (K, V) [batch, token, heads, head_dim] of C: for cross attention they depend on the prompt alone,
        # and causal.Stream computes them once per prompt (context_kv)
        K = self.norm_k(self.k(C)).reshape(C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
        V = self.v(C).reshape(C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
        return K, V

    def forward(self, X, grid_sizes=None, freq=None, context=None, start_frame=0, cache=None, context_kv=None):
        # softmax((QK^T)/sqrt(d)) V; keys and values come from context in cross attention (or context_kv, computed
        # beforehand), and from the cached frames before this chunk too when a cache is given (causal.KVCache)
        cross = context is not None or context_kv is not None
        Q = self.norm_q(self.q(X)).reshape(X.shape[0], X.shape[-2], self.num_heads, self.head_dim)
        K, V = self.keys_values(X if context is None else context) if context_kv is None else context_kv
        if not cross:
            Q = rope_apply(Q, grid_sizes, freq, start_frame)
            K = rope_apply(K, grid_sizes, freq, start_frame)
        if cache is not None:  # attention over the cached frames too, outside a compiled graph (causal.KVCache)
            return self.o(cache.attend(Q, K, V).reshape(X.shape[0], X.shape[-2], self.dim))
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        A = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        A = A.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[-2], self.dim)
        return self.o(A)
