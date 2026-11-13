import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseWindowAttention(nn.Module):
    """Local-window self attention with a small set of global tokens."""

    def __init__(self, hidden_size: int, num_heads: int, window_size: int, num_global: int = 8):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.num_global = num_global
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.head_dim ** 0.5
        positions = torch.arange(seq_len, device=hidden_states.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        local = distance <= self.window_size
        allowed = local | (positions[None, :] < self.num_global)
        scores = scores.masked_fill(~allowed, float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", weights, v)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return self.out(context)
