#!/usr/bin/env bash
# Download CIFAR-10 (the binary version, ~160 MB) into data/cifar-10/.
#
#   ./scripts/download_cifar10.sh
#
# Afterwards data/cifar-10/ holds data_batch_1.bin ... data_batch_5.bin
# (50,000 training images) and test_batch.bin (10,000), the files
# Flockml.Cifar10.load reads.
set -euo pipefail

out="data/cifar-10"
if [ -f "$out/test_batch.bin" ]; then
  echo "$out already exists"
  exit 0
fi
mkdir -p "$out"
echo "downloading cifar-10-binary.tar.gz"
curl -fL "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" -o "$out/cifar-10-binary.tar.gz"
tar -xzf "$out/cifar-10-binary.tar.gz" -C "$out" --strip-components 1
rm "$out/cifar-10-binary.tar.gz"
echo "done: $out"
