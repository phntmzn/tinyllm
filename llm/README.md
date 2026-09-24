# Tiny LLM from Scratch

A compact, educational character-level language model built with PyTorch. It uses a decoder-only Transformer with learned token and position embeddings, causal self-attention, and a feed-forward network. The model starts with random weights and learns from a plain-text corpus that you provide.

This is a small learning project, not a pretrained general-purpose assistant. Character tokenization keeps the implementation easy to inspect, but is less efficient than modern subword tokenizers.

## Project structure

```text
.
├── model.py             # Character tokenizer and causal Transformer
├── train.py             # Training loop and checkpoint writer
├── generate.py          # Checkpoint loader and text sampler
├── requirements.txt     # Python dependencies
├── data/                # Put UTF-8 training text files here
├── build_pkg.sh         # Builds a macOS installer package
├── packaging/
│   └── tiny-llm         # Installed command-line launcher
├── dist/                # Built .pkg installer (generated)
└── README.md
```

Put UTF-8 `.txt` training corpora in `data/`. `data/example.txt` is a small sample corpus for trying out the training pipeline; larger and more varied text will produce better results. Trained model checkpoints are not included.

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer

## Install from source

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Train

Save your training text as `data/corpus.txt`, then run:

```bash
python train.py data/example.txt --steps 3000 --out model.pt
```

The default model has context length 128, embedding width 128, four attention heads, and four Transformer layers. The corpus must contain at least `context + 2` characters. Training prints its loss every 100 steps.

Use `--device cpu` to force CPU training or `--device cuda` on a CUDA-capable machine. Adjust model and training settings with command-line options; run `python train.py --help` to see them.

## Generate text

```bash
python generate.py model.pt --prompt "Once upon a time" --tokens 500
```

The prompt can only contain characters present in the training corpus. Lower temperatures generally produce more conservative samples; higher temperatures produce more varied samples. Run `python generate.py --help` for all options.

## Install on macOS with a `.pkg`

Build the installer on a Mac with the Xcode Command Line Tools installed:

```bash
./build_pkg.sh
```

This creates `dist/TinyLLM-0.1.0.pkg`. Open it in Finder and follow the installer. It installs the `tiny-llm` command and model source under `/usr/local`. The package is unsigned, so macOS may ask you to approve opening it.

Install Python 3.10 or newer if you do not already have it, then initialize the local runtime:

```bash
tiny-llm setup
```

This downloads PyTorch and creates a virtual environment in `~/Library/Application Support/TinyLLM/`. After setup, train and generate locally:

```bash
tiny-llm train data/example.txt --steps 3000 --out model.pt
tiny-llm generate model.pt --prompt "Once upon a time" --tokens 500
```

The `.pkg` installs the software, not pretrained weights. You supply a corpus and train your own checkpoint. Python and PyTorch are installed separately from the package to keep it small and avoid bundling a machine-specific Python runtime.

## How it works

1. The tokenizer maps each corpus character to an integer ID.
2. During training, the Transformer learns to predict the next character from the preceding context.
3. The checkpoint stores the model weights, vocabulary, and architecture settings.
4. Generation repeatedly samples a character and adds it to the context.
