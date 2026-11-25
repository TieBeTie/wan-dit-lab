
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
        if hasattr(self, "sage_cross_tokens"):
            from wan_dit_lab.kernels import sage_cross
            if C.shape[-2] != self.sage_cross_tokens:
                raise ValueError("Cross context must match the exported text length")
            shape = (C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
            return sage_cross.prepare_kv(self.k(C).reshape(shape), self.v(C).reshape(shape),
                                          self.norm_k.weight, self.norm_k.eps)
        K = self.norm_k(self.k(C).to(dtype=getattr(self, "rope_input_dtype", None))).reshape(C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
        V = self.v(C).reshape(C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
        return K, V

    def forward(self, X, grid_sizes=None, freq=None, context=None, start_frame=0, cache=None, context_kv=None):
        # softmax((QK^T)/sqrt(d))
        cross = context is not None or context_kv is not None
        if cross and hasattr(self, "sage_cross_tokens"):
            from wan_dit_lab.kernels import sage_cross
            Q = self.q(X).reshape(X.shape[0], X.shape[-2], self.num_heads, self.head_dim)
            key, value = self.keys_values(context) if context_kv is None else context_kv
            A = sage_cross.attend(Q, self.norm_q.weight, self.norm_q.eps, key, value, self.sage_cross_tokens)
            return self.o(A.reshape(*X.shape[:-1], self.dim))
        if not cross and getattr(self, "sage_rope", False):
            shape = (X.shape[0], X.shape[-2], self.num_heads, self.head_dim)
            Q, K, V = (layer(X).reshape(shape) for layer in (self.q, self.k, self.v))
            A = cache.attend_rope(Q, K, V, self.norm_q, self.norm_k, grid_sizes, freq, start_frame)
            return self.o(A.reshape(*X.shape[:-1], self.dim))
        Q = self.norm_q(self.q(X).to(dtype=getattr(self, "rope_input_dtype", None))).reshape(X.shape[0], X.shape[-2], self.num_heads, self.head_dim)
        K, V = self.keys_values(X if context is None else context) if context_kv is None else context_kv
        if not cross:
            apply_rope = getattr(self, "rope_kernel", rope_apply)
            # Q and K determine position-sensitive attention weights.
            # V supplies the content those weights combine.
            Q = apply_rope(Q, grid_sizes, freq, start_frame)
            K = apply_rope(K, grid_sizes, freq, start_frame)
        if cache is not None:  # attention over the cached frames too, outside a compiled graph (causal.KVCache)
            return self.o(cache.attend(Q, K, V).reshape(X.shape[0], X.shape[-2], self.dim))
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        A = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        A = A.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[-2], self.dim)
        return self.o(A)
