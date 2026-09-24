"""Train a small character-level language model from a plain text corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from model import CharTokenizer, TinyGPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="UTF-8 text file to learn from")
    parser.add_argument("--out", type=Path, default=Path("model.pt"), help="checkpoint path")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(data) < args.context + 2:
        raise ValueError(f"Corpus has {len(data)} characters; need at least context + 2 ({args.context + 2})")
    device = torch.device(args.device)
    model = TinyGPT(len(tokenizer.itos), args.context, args.width, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(data) - args.context - 1, (args.batch_size,))
        x = torch.stack([data[i : i + args.context] for i in starts]).to(device)
        y = torch.stack([data[i + 1 : i + args.context + 1] for i in starts]).to(device)
        return x, y

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "vocabulary": tokenizer.itos,
            "config": {
                "context_size": args.context,
                "width": args.width,
                "heads": args.heads,
                "layers": args.layers,
            },
        },
        args.out,
    )
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
