#!/usr/bin/env bash
# Download MNIST (default) or Fashion-MNIST into data/<name>/ and unzip.
#
#   ./scripts/download_mnist.sh            # -> data/mnist
#   ./scripts/download_mnist.sh fashion    # -> data/fashion-mnist
set -euo pipefail

case "${1:-mnist}" in
  mnist)
    base="https://ossci-datasets.s3.amazonaws.com/mnist"
    out="data/mnist" ;;
  fashion)
    base="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
    out="data/fashion-mnist" ;;
  *) echo "usage: $0 [mnist|fashion]" >&2; exit 1 ;;
esac

mkdir -p "$out"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ -f "$out/$f" ]; then
    echo "$out/$f already exists"
    continue
  fi
  echo "downloading $f"
  curl -fsSL "$base/$f.gz" -o "$out/$f.gz"
  gunzip -f "$out/$f.gz"
done
echo "done: $out"
