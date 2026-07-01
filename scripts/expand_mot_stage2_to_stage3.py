#!/usr/bin/env python3
"""
Expand a Stage 2 VLM checkpoint into a Stage 3 MoT init checkpoint.

Intended training flow:
  Stage 1 LM / Stage 2 projector / Stage 2 MMU are trained as single-stream
  7-routed-expert checkpoints (use_mot=False).  Before Stage 3 T2I training,
  this script copies the trained single-stream VLM weights into the MoT gen
  stream so Stage 3 can run with use_mot=True.

Source checkpoint (Stage 2 VLM, use_mot=False):
  model.layers.N.self_attn.{q,k,v,o}_proj
  model.layers.N.self_attn.{query,key}_layernorm
  model.layers.N.{input,post_attention}_layernorm
  model.layers.N.mlp.experts.{0-6}.*       7 routed und experts
  model.layers.N.mlp.shared_mlp.*
  model.layers.N.mlp.gate.wg.weight        [7, H]

Destination checkpoint (Stage 3 MoT init, use_mot=True):
  Keeps all source weights unchanged for the und stream, and adds:
  self_attn.*_mot_gen                       copied from und attention
  *_layernorm_mot_gen                       copied from und norms
  mlp_mot_gen.experts.{0-5}.*               copied from und experts 0..5
  mlp_mot_gen.shared_mlp.*                  copied from und shared_mlp
  mlp_mot_gen.gate.wg.weight                zeros [6, H]

Input and output are HF-style safetensors directories.
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import torch
from safetensors.torch import load_file, save_file


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


_UND_EXPERT_RE = re.compile(
    r"^(.*\.mlp\.experts\.)(\d+)(\.(?:down_proj|gate_proj|up_proj)\.weight)$"
)
_UND_GATE_RE = re.compile(r"^(.*\.mlp\.gate)\.wg\.weight$")
_ATTN_Q_RE = re.compile(r"^(.*\.self_attn)\.q_proj(\..*)$")
_ATTN_K_RE = re.compile(r"^(.*\.self_attn)\.k_proj(\..*)$")
_ATTN_V_RE = re.compile(r"^(.*\.self_attn)\.v_proj(\..*)$")
_ATTN_QKV_RE = re.compile(r"^(.*\.self_attn)\.qkv_proj(\..*)$")
_ATTN_O_RE = re.compile(r"^(.*\.self_attn)\.o_proj(\..*)$")
_QUERY_LN_RE = re.compile(r"^(.*\.self_attn)\.query_layernorm(\..*)$")
_KEY_LN_RE = re.compile(r"^(.*\.self_attn)\.key_layernorm(\..*)$")
_INPUT_LN_RE = re.compile(r"^(.*\.\d+)\.input_layernorm(\..*)$")
_POST_LN_RE = re.compile(r"^(.*\.\d+)\.post_attention_layernorm(\..*)$")
_SHARED_MLP_RE = re.compile(r"^(.*\.mlp\.shared_mlp)(\..*)$")


def _is_mot_gen_key(key: str) -> bool:
    return "mlp_mot_gen" in key or "_mot_gen" in key


def expand_state_dict(src_sd: dict, num_gen_experts: int) -> dict:
    dst = {}
    unexpected_gen = [k for k in src_sd if _is_mot_gen_key(k)]
    if unexpected_gen:
        raise ValueError(
            f"Source checkpoint already contains gen-stream keys "
            f"(e.g. {unexpected_gen[:3]}). Is this already a Stage 3 checkpoint?"
        )

    for key, tensor in src_sd.items():
        t = tensor.cpu()
        dst[key] = t.clone()

        m = _ATTN_Q_RE.match(key)
        if m:
            dst[f"{m.group(1)}.q_proj_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _ATTN_K_RE.match(key)
        if m:
            dst[f"{m.group(1)}.k_proj_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _ATTN_V_RE.match(key)
        if m:
            dst[f"{m.group(1)}.v_proj_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _ATTN_QKV_RE.match(key)
        if m:
            dst[f"{m.group(1)}.qkv_proj_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _ATTN_O_RE.match(key)
        if m:
            dst[f"{m.group(1)}.o_proj_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _QUERY_LN_RE.match(key)
        if m:
            dst[f"{m.group(1)}.query_layernorm_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _KEY_LN_RE.match(key)
        if m:
            dst[f"{m.group(1)}.key_layernorm_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _INPUT_LN_RE.match(key)
        if m and ".mlp." not in key and ".self_attn." not in key:
            dst[f"{m.group(1)}.input_layernorm_mot_gen{m.group(2)}"] = t.clone()
            continue
        m = _POST_LN_RE.match(key)
        if m and ".mlp." not in key and ".self_attn." not in key:
            dst[f"{m.group(1)}.post_attention_layernorm_mot_gen{m.group(2)}"] = t.clone()
            continue

        m = _UND_EXPERT_RE.match(key)
        if m:
            src_idx = int(m.group(2))
            if src_idx < num_gen_experts:
                gen_prefix = m.group(1).replace(".mlp.experts.", ".mlp_mot_gen.experts.")
                dst[f"{gen_prefix}{src_idx}{m.group(3)}"] = t.clone()
            continue

        m = _SHARED_MLP_RE.match(key)
        if m:
            dst[key.replace(".mlp.shared_mlp", ".mlp_mot_gen.shared_mlp")] = t.clone()
            continue

        m = _UND_GATE_RE.match(key)
        if m:
            hidden_size = t.shape[1]
            gen_gate_key = (
                m.group(1).replace(".mlp.gate", ".mlp_mot_gen.gate")
                + ".wg.weight"
            )
            dst[gen_gate_key] = torch.zeros(num_gen_experts, hidden_size, dtype=t.dtype)
            continue

    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Stage 2 single-stream VLM weights to Stage 3 MoT init.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src",
        required=True,
        help=(
            "Path to Stage 2 VLM safetensors checkpoint dir.\n"
            "This should be a 7-routed-expert use_mot=False checkpoint."
        ),
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output path for Stage 3 MoT init safetensors checkpoint dir.",
    )
    parser.add_argument(
        "--num-gen-experts",
        type=int,
        default=6,
        help="Number of routed gen-stream experts to create (default: 6).",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=5.0,
        help="Shard size in GiB for output safetensors files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f'\n{"=" * 60}')
    print(f"Source Stage 2 VLM : {args.src}")
    print(f"Target Stage 3 MoT : {args.dst}")
    print(f"Gen experts        : {args.num_gen_experts} routed + 1 shared")
    print(f'{"=" * 60}\n')

    print("Loading Stage 2 VLM checkpoint (safetensors) ...")
    sd = _load_safetensors(args.src)

    expert_keys = [k for k in sd if _UND_EXPERT_RE.match(k)]
    gate_keys = [k for k in sd if _UND_GATE_RE.match(k)]
    shared_keys = [k for k in sd if _SHARED_MLP_RE.match(k)]
    q_keys = [k for k in sd if _ATTN_Q_RE.match(k)]
    qkv_keys = [k for k in sd if _ATTN_QKV_RE.match(k)]
    qln_keys = [k for k in sd if _QUERY_LN_RE.match(k)]

    print(f"  Total keys loaded        : {len(sd)}")
    print(f"  Attn q_proj (split)      : {len(q_keys)}")
    print(f"  Attn qkv_proj (fused)    : {len(qkv_keys)}")
    print(f"  QK norm query_layernorm  : {len(qln_keys)}")
    print(f"  Und expert param tensors : {len(expert_keys)}")
    print(f"  Und router tensors       : {len(gate_keys)}")
    print(f"  Und shared_mlp tensors   : {len(shared_keys)}")

    if not gate_keys:
        sample = list(sd.keys())[:5]
        print(f"\n  Sample keys (first 5): {sample}")
        raise SystemExit("ERROR: no mlp.gate.wg.weight tensors found.")
    if not q_keys and not qkv_keys:
        raise SystemExit("ERROR: no attention projection keys found.")

    gate_shapes = sorted({tuple(sd[k].shape) for k in gate_keys})
    print(f"  Und router shapes        : {gate_shapes}")
    if any(shape[0] != 7 for shape in gate_shapes):
        raise SystemExit(
            "ERROR: source is not a 7-expert MoT/VLM checkpoint. "
            f"Found router shapes {gate_shapes}. Do not use an ours/12-expert checkpoint here."
        )

    indices_found = sorted({int(_UND_EXPERT_RE.match(k).group(2)) for k in expert_keys})
    print(f"  Und expert indices found : {indices_found}  (expected [0..6])")
    if indices_found != list(range(7)):
        raise SystemExit(f"ERROR: expected und expert indices [0..6], got {indices_found}.")
    if max(indices_found) < args.num_gen_experts - 1:
        raise SystemExit(
            f"ERROR: need at least {args.num_gen_experts} und experts, "
            f"but only found max index {max(indices_found)}."
        )

    print("\nExpanding: adding Stage 3 MoT gen stream ...")
    dst_sd = expand_state_dict(sd, num_gen_experts=args.num_gen_experts)

    gen_expert_keys = [k for k in dst_sd if "mlp_mot_gen.experts" in k]
    gen_gate_keys = [k for k in dst_sd if "mlp_mot_gen.gate.wg.weight" in k]
    gen_shared_keys = [k for k in dst_sd if "mlp_mot_gen.shared_mlp" in k]
    gen_q_keys = [k for k in dst_sd if "q_proj_mot_gen" in k]
    gen_qln_keys = [k for k in dst_sd if "query_layernorm_mot_gen" in k]

    print(f"  Total keys in Stage 3 init : {len(dst_sd)}")
    print(f"  Gen expert param tensors   : {len(gen_expert_keys)}")
    print(f"  Gen router tensors         : {len(gen_gate_keys)}")
    print(f"  Gen shared_mlp tensors     : {len(gen_shared_keys)}")
    print(f"  Gen q_proj_mot_gen         : {len(gen_q_keys)}")
    print(f"  Gen query_layernorm_mot_gen: {len(gen_qln_keys)}")

    if gen_gate_keys:
        sample_gate = gen_gate_keys[0]
        print(f"\n  Sample gen gate shape      : {dst_sd[sample_gate].shape}")
        if not torch.all(dst_sd[sample_gate] == 0):
            raise SystemExit("ERROR: gen router is not all-zero.")

    print(f"\nSaving Stage 3 MoT init checkpoint to: {args.dst}")
    if os.path.exists(args.dst):
        print(f"  WARNING: {args.dst} already exists, overwriting.")
    _save_safetensors(args.dst, dst_sd, shard_size_gb=args.shard_size_gb)

    print("\nDone.")
    print("Next steps:")
    print(f'  1. Set CKPT_DIR="{args.dst}"')
    print("  2. Run Stage 3 training with train/configs/stage3_mm_mot.yaml (use_mot=True)")


if __name__ == "__main__":
    main()
