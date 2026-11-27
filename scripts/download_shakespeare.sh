#!/usr/bin/env bash
# Download Tiny Shakespeare (~1 MB of text, Karpathy's char-rnn dataset) to
# data/tinyshakespeare/input.txt, for bin/gpt.ml.
set -euo pipefail

out="data/tinyshakespeare"
if [ -f "$out/input.txt" ]; then
  echo "$out/input.txt already exists"
  exit 0
fi
mkdir -p "$out"
curl -fsSL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
  -o "$out/input.txt"
echo "done: $out/input.txt"
