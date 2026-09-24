"""A small decoder-only character-level Transformer language model."""

from __future__ import annotations

import torch
from torch import nn


class CharTokenizer:
    """Character vocabulary with a stable, serializable mapping."""

    def __init__(self, characters: str):
        self.itos = sorted(set(characters))
        if not self.itos:
            raise ValueError("The training text must contain at least one character")
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"Text contains characters absent from training data: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class TinyGPT(nn.Module):
    """GPT-style causal language model using PyTorch's TransformerEncoder layers."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int = 128,
        width: int = 128,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_size, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.context_size:
            raise ValueError(f"Sequence length {length} exceeds context size {self.context_size}")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        # True entries are masked: each position can attend only to itself and earlier tokens.
        causal_mask = torch.triu(torch.ones(length, length, device=token_ids.device, dtype=torch.bool), diagonal=1)
        return self.output(self.norm(self.blocks(x, mask=causal_mask)))

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.eval()
        for _ in range(new_tokens):
            context = token_ids[:, -self.context_size :]
            logits = self(context)[:, -1, :] / temperature
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        return token_ids
