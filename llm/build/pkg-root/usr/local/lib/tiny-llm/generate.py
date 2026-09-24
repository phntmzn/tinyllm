"""Generate text from a checkpoint produced by train.py."""

from __future__ import annotations

import argparse
import torch

from model import CharTokenizer, TinyGPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="checkpoint saved by train.py")
    parser.add_argument("--prompt", default="", help="starting text (characters must occur in training corpus)")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tokenizer = CharTokenizer("".join(checkpoint["vocabulary"]))
    tokenizer.itos = checkpoint["vocabulary"]
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.itos)}
    model = TinyGPT(len(tokenizer.itos), **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    prompt = args.prompt or tokenizer.itos[0]
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    result = model.generate(ids, args.tokens, args.temperature)[0].tolist()
    print(tokenizer.decode(result))


if __name__ == "__main__":
    main()
