# Tiny LLM from Scratch

A compact, educational character-level language model built with PyTorch. It uses a decoder-only Transformer with learned token and position embeddings, causal self-attention, and a feed-forward network. The model starts with random weights and learns from a plain-text corpus that you provide.

This is a small-scale learning project, not a ChatGPT-sized assistant. Character tokenization keeps the implementation easy to inspect, though it is less efficient than modern subword tokenizers.

## Project structure

```text
.
├── model.py          # Character tokenizer and causal Transformer model
├── train.py          # Training loop and checkpoint creation
├── generate.py       # Load a checkpoint and generate text
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

Training also requires a UTF-8 text corpus of your choice. The corpus and generated checkpoint are not included in the repository by default.

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Save your training text as `corpus.txt`, then run:

```bash
python train.py corpus.txt --steps 3000 --out model.pt
```

The default model uses a context length of 128, embedding width of 128, four attention heads, and four Transformer layers. The corpus must contain at least `context + 2` characters. Training prints the loss every 100 steps.

Use `--device cpu` to force CPU training or `--device cuda` when a CUDA GPU is available. Model size and training settings can be adjusted with command-line options; run `python train.py --help` to see them.

## Generate text

```bash
python generate.py model.pt --prompt "Once upon a time" --tokens 500
```

The prompt may only use characters that appeared in the training corpus. Lower temperatures generally produce more conservative samples; higher temperatures produce more varied samples. Run `python generate.py --help` for all options.

## Install on macOS with a `.pkg`

Build the installer package from the project directory:

```bash
./build_pkg.sh
```

This creates `dist/TinyLLM-0.1.0.pkg`. Open the package in Finder to install the `tiny-llm` command and project files under `/usr/local`. The installer package is unsigned, so macOS may require you to approve opening it.

After installation, install Python 3.10 or newer if needed, then set up the isolated runtime:

```bash
tiny-llm setup
```

Setup downloads PyTorch and installs it into a virtual environment under `/usr/local/lib/tiny-llm/.venv`. You can then train and generate locally:

```bash
tiny-llm train corpus.txt --steps 3000 --out model.pt
tiny-llm generate model.pt --prompt "Once upon a time" --tokens 500
```

The `.pkg` installs the model software, not pretrained weights. Training requires your own text corpus. Python and PyTorch are installed separately from the package so the installer stays small and avoids bundling a Python runtime tied to one macOS machine.

## How it works

1. The tokenizer maps every character in the corpus to an integer ID.
2. During training, the Transformer learns to predict the next character from the preceding context.
3. The checkpoint stores the model weights, vocabulary, and architecture settings.
4. Generation repeatedly samples a next character and appends it to the context.

The project does not include pretrained weights or a default corpus. You supply the data and train the model yourself.
