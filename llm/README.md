# Tiny LLM from scratch

A compact, educational character-level language model. The model is a decoder-only Transformer with learned token and position embeddings, causal self-attention, a feed-forward network, and a language-modeling training loop. It starts with random weights and learns only from the corpus you provide.

This is a small-scale learning project, not a ChatGPT-sized assistant. Character tokenization keeps the implementation easy to inspect; it is less efficient and less capable than modern subword tokenizers.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Put a UTF-8 text corpus in `corpus.txt` (more varied text and more training steps generally help), then run:

```bash
python train.py corpus.txt --steps 3000 --out model.pt
```

The default model uses context 128, width 128, four attention heads, and four Transformer layers. Training needs at least `context + 2` characters. Use `--device cpu` to force CPU or `--device cuda` when a CUDA GPU is available.

## Generate

```bash
python generate.py model.pt --prompt "Once upon a time" --tokens 500
```

Prompts can only contain characters present in the training corpus. Lower temperatures produce more conservative samples; higher temperatures produce more varied ones.

## Files

- `model.py`: tokenizer and causal Transformer implementation
- `train.py`: next-character training loop and checkpoint writer
- `generate.py`: checkpoint loader and autoregressive sampler

The checkpoint includes the vocabulary and model dimensions needed for generation. This project has not been trained on a corpus by default; you choose the data and run the training command.
