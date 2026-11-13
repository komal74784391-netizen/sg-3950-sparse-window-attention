import torch
import torch.nn as nn

from .attention import SparseWindowAttention


class SparseWindowTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_layers: int, window_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": SparseWindowAttention(hidden_size, num_heads, window_size),
                        "norm1": nn.LayerNorm(hidden_size),
                        "norm2": nn.LayerNorm(hidden_size),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size)
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block["norm1"](hidden + block["attn"](hidden, attention_mask))
            hidden = block["norm2"](hidden + block["ffn"](hidden))
        return self.head(hidden)
