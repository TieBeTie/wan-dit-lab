import math

import torch

from wan_dit_lab.rms_norm import RMSNorm
from wan_dit_lab.rope import rope_apply


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
    
    def forward(self, X, grid_sizes=None, freq=None, context=None):
        # softmax((QK^T)/sqrt(d)) V; keys and values come from context in cross attention
        C = X if context is None else context
        Q, K, V = self.norm_q(self.q(X)), self.norm_k(self.k(C)), self.v(C)
        Q = Q.reshape(X.shape[0], X.shape[-2], self.num_heads, self.head_dim)
        K = K.reshape(C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
        if context is None:
            Q = rope_apply(Q, grid_sizes, freq)
            K = rope_apply(K, grid_sizes, freq)
        V = V.reshape(C.shape[0], C.shape[-2], self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        A = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        A = A.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[-2], self.dim)
        return self.o(A)
