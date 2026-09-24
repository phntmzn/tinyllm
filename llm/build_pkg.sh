#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VERSION="${1:-0.1.0}"
STAGE="$ROOT/build/pkg-root"
OUT="$ROOT/dist/TinyLLM-$VERSION.pkg"

rm -rf "$STAGE"
mkdir -p "$STAGE/usr/local/lib/tiny-llm" "$STAGE/usr/local/bin" "$ROOT/dist"
install -m 644 "$ROOT/model.py" "$ROOT/train.py" "$ROOT/generate.py" "$ROOT/requirements.txt" "$STAGE/usr/local/lib/tiny-llm/"
install -m 755 "$ROOT/packaging/tiny-llm" "$STAGE/usr/local/bin/tiny-llm"

pkgbuild \
  --root "$STAGE" \
  --identifier "org.tinyllm.local" \
  --version "$VERSION" \
  --install-location / \
  "$OUT"

echo "Created $OUT"
