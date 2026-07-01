#!/usr/bin/env python3
"""
Expand "Ours" Stage 1 checkpoint (3 routed + 1 shared expert) to Stage 2
(12 routed: 3 text + 3 vit + 6 vae, plus 1 shared) for modality-specific routing.

Expert mapping per MoE layer
─────────────────────────────
Stage 1   →  Stage 2
experts.0    experts.0  (text-0)
experts.1    experts.1  (text-1)
experts.2    experts.2  (text-2)
experts.0    experts.3  (vit-0,  copy of text-0)
experts.1    experts.4  (vit-1,  copy of text-1)
experts.2    experts.5  (vit-2,  copy of text-2)
experts.0    experts.6  (vae-0,  copy of text-0)
experts.1    experts.7  (vae-1,  copy of text-1)
experts.2    experts.8  (vae-2,  copy of text-2)
experts.0    experts.9  (vae-3,  2nd copy of text-0)
experts.1    experts.10 (vae-4,  2nd copy of text-1)
experts.2    experts.11 (vae-5,  2nd copy of text-2)
shared_mlp   shared_mlp (unchanged)

Router mapping per MoE layer
─────────────────────────────
Stage 1 gate.wg.weight  [3, H]
  → Stage 2 gate.wg_text.weight [3, H]   (copy)
  → Stage 2 gate.wg_vit.weight  [3, H]   (copy)
  → Stage 2 gate.wg_vae.weight  [6, H]:
      cat([wg, wg])  → logits [a,b,c,a,b,c], top-2 picks "aa" at init

Rationale for VAE router initialization ("aa" via cat([wg,wg])):
  VAE is a new modality unseen in Stage 1; preserving Stage 1's exact "ab"
  routing output for VAE tokens is not a priority.

  The key goal is fast expert diversification across all 6 VAE experts.
  cat([wg,wg]) produces a perfectly symmetric router: all 6 experts have
  equal softmax probability, so balance loss gives equal gradient to all 6
  from step 1. Symmetry breaks quickly and all 6 experts specialise in parallel.

  Alternatives like wg_vae[3:] = wg - C suppress group-2's softmax probability
  by e^-C (exponentially weak gradient). With C=10, experts 9-11 are effectively
  dormant for thousands of steps, wasting half the VAE expert capacity.

All attention / norm / embedding / shared_mlp tensors are copied unchanged.
"""

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import torch
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Safetensors load / save
# ---------------------------------------------------------------------------

def _load_safetensors(src_path: str) -> dict:
    """Load the current repo's checkpoint format: model.safetensors or HF shards."""
    single = os.path.join(src_path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single, device="cpu")

    index_file = os.path.join(src_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        state_dict = {}
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(os.path.join(src_path, shard), device="cpu"))
        return state_dict

    shards = sorted(
        f for f in os.listdir(src_path)
        if f.endswith(".safetensors") and f != "model.safetensors"
    )
    if shards:
        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(os.path.join(src_path, shard), device="cpu"))
        return state_dict

    raise FileNotFoundError(f"No safetensors checkpoint found in {src_path}")


def _save_safetensors(dst_path: str, state_dict: dict, shard_size_gb: float = 5.0) -> None:
    """Save HF-style sharded safetensors that FSDPCheckpoint.load_model_weights can read."""
    os.makedirs(dst_path, exist_ok=True)

    shard_size_bytes = int(shard_size_gb * 1024 ** 3)
    shards = []
    current_shard = {}
    current_size = 0
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    total_shards = len(shards)
    weight_map = {}
    total_size = 0
    for shard_idx, shard in enumerate(shards, 1):
        filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, os.path.join(dst_path, filename))
        for key, tensor in shard.items():
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()

    with open(os.path.join(dst_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)


# ---------------------------------------------------------------------------
# Key / tensor expansion
# ---------------------------------------------------------------------------

# Regex for expert tensors:  <prefix>.experts.<idx>.<suffix>
_EXPERT_RE = re.compile(
    r'^(.*\.mlp\.experts\.)(\d+)(\.(?:down_proj|gate_proj|up_proj)\.weight)$'
)

# Regex for the old single gate router:  <prefix>.mlp.gate.wg.weight
_GATE_RE = re.compile(
    r'^(.*\.mlp\.gate)\.wg\.weight$'
)


def expand_state_dict(src_sd: dict) -> dict:
    """
    Return a new state_dict with Stage 2 structure.
    Source has 3 experts per layer; destination has 12 experts per layer.
    """
    dst: dict = {}

    for key, tensor in src_sd.items():
        tensor_cpu = tensor.cpu()  # ensure CPU

        # ── Expert weights ──────────────────────────────────────────────────
        m = _EXPERT_RE.match(key)
        if m:
            prefix, idx_str, suffix = m.group(1), m.group(2), m.group(3)
            src_idx = int(idx_str)
            # text experts (0-2): keep original index
            dst[f'{prefix}{src_idx}{suffix}']      = tensor_cpu.clone()
            # vit  experts (3-5): +3
            dst[f'{prefix}{src_idx + 3}{suffix}']  = tensor_cpu.clone()
            # vae  experts (6-8): +6  (first copy)
            dst[f'{prefix}{src_idx + 6}{suffix}']  = tensor_cpu.clone()
            # vae  experts (9-11): +9 (second copy)
            dst[f'{prefix}{src_idx + 9}{suffix}']  = tensor_cpu.clone()
            continue

        # ── Gate (router) weight ─────────────────────────────────────────────
        m2 = _GATE_RE.match(key)
        if m2:
            gate_prefix = m2.group(1)
            wg = tensor_cpu                                         # [3, H]
            dst[f'{gate_prefix}.wg_text.weight'] = wg.clone()      # [3, H]
            dst[f'{gate_prefix}.wg_vit.weight']  = wg.clone()      # [3, H]
            # wg_vae initialization: cat([wg, wg]) → symmetric "aa" init.
            #
            # With logits [a,b,c,a,b,c], top-2 selects two copies of the top expert
            # ("aa"). While this differs from Stage 1's "ab" output, it is the right
            # choice for VAE convergence because:
            #
            #   1. Perfect symmetry → balance loss gives equal gradient to all 6 experts
            #      → all 6 start differentiating immediately from step 1.
            #
            #   2. Alternatives like wg_vae[3:] = wg - C suppress group-2's softmax
            #      probability by e^-C, making their gradient signal exponentially weak.
            #      With C=10, experts 9-11 are effectively dormant for thousands of steps,
            #      wasting half the VAE expert capacity.
            #
            #   3. VAE tokens are a new modality unseen in Stage 1; preserving Stage 1's
            #      exact "ab" output for VAE is not a goal. Fast expert diversification is.
            dst[f'{gate_prefix}.wg_vae.weight'] = torch.cat([wg, wg], dim=0)  # [6, H]
            continue

        # ── embed_tokens.weight: copy as-is (151936 rows from Stage 1) ──────────
        # Stage 1 (LM-only) has vocab_size=151936; Stage 2 (multimodal) has vocab_size=157420.
        # We intentionally do NOT expand the embedding here.
        # The checkpoint_manager's _VocabExpandLoadPlanner (triggered by load_from_path
        # with strict=False) handles the mismatch at load time:
        #   - Loads the first 151936 rows in-place from this checkpoint.
        #   - Leaves the new 5484 rows at the model's small init_std values (≈ 0.02).
        # This matches exactly what qwen3-06b-upcycling-moe-mm does and gives
        # lm_text_loss ≈ 1.9 at step 1 (not ≈ 12 from large random rows dominating softmax).

        # ── All other tensors (attn, norms, shared_mlp, lm_head …) ──────────
        dst[key] = tensor_cpu.clone()

    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Expand "Ours" Stage 1 checkpoint (3 experts) to Stage 2 (12 experts).',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        '--src', required=True,
        help=(
            'Path to Stage 1 safetensors checkpoint dir.\n'
            'Usually: <output_path>/ckpt/XXXXXXX'
        ),
    )
    p.add_argument(
        '--dst', required=True,
        help=(
            'Output path for Stage 2 init safetensors checkpoint dir.\n'
            'Example: outputs/stage2_init'
        ),
    )
    p.add_argument('--shard-size-gb', type=float, default=5.0,
                   help='Shard size in GB for output safetensors files')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f'\n{"=" * 60}')
    print(f'Source  : {args.src}')
    print(f'Target  : {args.dst}')
    print(f'{"=" * 60}\n')

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print('Loading Stage 1 checkpoint (safetensors) ...')
    sd = _load_safetensors(args.src)

    # Diagnostics
    expert_keys = [k for k in sd if _EXPERT_RE.match(k)]
    gate_keys   = [k for k in sd if _GATE_RE.match(k)]
    print(f'  Total keys loaded    : {len(sd)}')
    print(f'  Expert param tensors : {len(expert_keys)}  (expected 28 layers × 3 experts × 3 = 252)')
    print(f'  Gate router tensors  : {len(gate_keys)}   (expected 28 layers = 28)')

    if not gate_keys:
        sample = list(sd.keys())[:5]
        print(f'\n  Sample keys (first 5): {sample}')
        print('\nERROR: No gate.wg.weight tensors found. Is --src pointing to a Stage 1 checkpoint dir?')
        sys.exit(1)

    # Verify expert indices are {0,1,2}
    indices_found = sorted({int(_EXPERT_RE.match(k).group(2)) for k in expert_keys})
    print(f'  Expert indices found : {indices_found}  (expected [0, 1, 2])')

    # ── 2. Expand ─────────────────────────────────────────────────────────────
    print('\nExpanding ...')
    dst_sd = expand_state_dict(sd)

    new_expert_keys = [k for k in dst_sd if _EXPERT_RE.match(k)]
    new_gate_keys   = [k for k in dst_sd if re.search(r'\.gate\.wg_', k)]
    print(f'  Total keys in Stage 2: {len(dst_sd)}')
    print(f'  Expert param tensors : {len(new_expert_keys)}  (expected 28 × 12 × 3 = 1008)')
    print(f'  Gate router tensors  : {len(new_gate_keys)}  (expected 28 × 3 = 84, for wg_text/vit/vae)')

    # Spot-check a gate tensor shape
    sample_gate = next(k for k in new_gate_keys if 'wg_vae' in k)
    print(f'  Sample wg_vae shape  : {dst_sd[sample_gate].shape}  (expected [6, 1024])')

    # Spot-check embed_tokens (should remain 151936 rows, NOT expanded here)
    emb_key = next((k for k in dst_sd if k.endswith('embed_tokens.weight')), None)
    if emb_key:
        print(f'  embed_tokens shape   : {dst_sd[emb_key].shape}  (expected [151936, 1024]; _VocabExpandLoadPlanner handles 151936→157420 at load time)')

    # ── 3. Save ───────────────────────────────────────────────────────────────
    print(f'\nSaving Stage 2 checkpoint to: {args.dst}')
    if os.path.exists(args.dst):
        print(f'  WARNING: {args.dst} already exists, overwriting.')
    _save_safetensors(args.dst, dst_sd, shard_size_gb=args.shard_size_gb)

    print('\nDone.')
    print('\nNext steps:')
    print(f'  1. Set CKPT_DIR="{args.dst}"')
    print( '  2. Run Stage 2 training with scripts/run/run_stage2_1_projector.sh')


if __name__ == '__main__':
    main()
