#!/usr/bin/env bash
set -euo pipefail

# 1. Download and extract T2I + MMU data (BAGEL example)
echo "[download_example_data] Downloading T2I + MMU data..."
wget -O bagel_example.zip https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
unzip -o bagel_example.zip -d example_data
mv example_data/bagel_example/vlm example_data/bagel_example/mmu
mv example_data/bagel_example/* example_data/
rm -rf example_data/bagel_example
rm -f bagel_example.zip
rm -rf example_data/editing

# 2. Download LM data (LLaVA-Instruct-150K subset)
echo "[download_example_data] Downloading LM data..."
mkdir -p example_data/lm
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json \
  -O example_data/lm/conversation_58k.json

echo "[download_example_data] Done. All example data prepared in example_data/"







