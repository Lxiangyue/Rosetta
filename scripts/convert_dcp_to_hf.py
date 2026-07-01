#!/usr/bin/env python3
"""Convert an FSDP/DCP checkpoint to HuggingFace safetensors format.

This script is standalone and does not depend on the training runtime.
It reads the DCP metadata to discover all tensor shapes, allocates empty
tensors, and uses torch.distributed.checkpoint.load to fill them from the
shard files.  The result is saved as sharded safetensors files compatible
with the project's HF loading path.

Typical usage (single GPU is sufficient):
    torchrun --nproc_per_node=1 \\
        scripts/convert_dcp_to_hf.py \\
        --ckpt  /path/to/dcp_weights \\
        --output /path/to/hf_output

The output directory will contain:
    model-00001-of-NNNNN.safetensors
    ...
    model.safetensors.index.json
    config.json               (empty; marks the dir as HF format)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint import FileSystemReader

try:
    from safetensors.torch import save_file
except ImportError:
    sys.exit("safetensors is required: pip install safetensors")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint → HF safetensors")
    parser.add_argument("--ckpt", required=True,
                        help="Input DCP checkpoint directory (the 'weights' sub-dir).")
    parser.add_argument("--output", required=True,
                        help="Output directory for HF safetensors files.")
    parser.add_argument("--shard-size-gb", type=float, default=5.0,
                        help="Max size (GiB) per safetensors shard. Default: 5.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load tensors onto ('cpu' or 'cuda'). Default: cpu.")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Expert key conversion  (fused → individual)
# ──────────────────────────────────────────────
_EXPERT_PROJ_NAMES = [
    ("experts.gate_proj_weights", "gate_proj"),
    ("experts.up_proj_weights",   "up_proj"),
    ("experts.down_proj_weights", "down_proj"),
]


def _unfuse_expert_keys(state_dict: dict) -> dict:
    """Expand fused expert weight tensors into per-expert keys.

    Some MoE checkpoints store expert weights as a stacked 3-D tensor:
        key.experts.gate_proj_weights  → shape [num_experts, out, in]
    The HF format expects individual keys:
        key.experts.0.gate_proj.weight, key.experts.1.gate_proj.weight, ...

    If no fused keys are present this is a no-op.
    """
    out = {}
    for key, value in state_dict.items():
        matched = False
        for fused_suffix, proj_name in _EXPERT_PROJ_NAMES:
            if key.endswith(fused_suffix):
                num_experts = value.shape[0]
                prefix = key[: -len(fused_suffix)]
                for i in range(num_experts):
                    new_key = f"{prefix}experts.{i}.{proj_name}.weight"
                    out[new_key] = value[i]
                matched = True
                break
        if not matched:
            out[key] = value
    return out


# ──────────────────────────────────────────────
# Shard counting (mirrors state_dict_to_hf)
# ──────────────────────────────────────────────
def _count_shards(sorted_keys, state_dict, shard_threshold):
    shard_id, cur = 0, 0
    for k in sorted_keys:
        nb = state_dict[k].numel() * state_dict[k].element_size()
        if cur > 0 and cur + nb > shard_threshold:
            shard_id += 1
            cur = 0
        cur += nb
    return max(shard_id + (1 if cur > 0 else 0), 1)


# ──────────────────────────────────────────────
# Main conversion logic
# ──────────────────────────────────────────────
def convert(args):
    rank = dist.get_rank()

    ckpt_path = args.ckpt
    reader = FileSystemReader(ckpt_path)
    metadata = reader.read_metadata()

    if rank == 0:
        print(f"[convert] DCP checkpoint : {ckpt_path}", flush=True)
        print(f"[convert] DCP metadata entries: {len(metadata.state_dict_metadata)}", flush=True)

    # ── Step 1: allocate tensors from DCP metadata ──────────────────────────
    # DCP was saved as  torch.save({"model": model_state_dict, ...})
    # so every real parameter key has the form  "model.<actual_param_key>".
    # The "size" field on TensorStorageMetadata is the GLOBAL tensor shape,
    # which is exactly what we want for a full, unsharded conversion.
    inner_state_dict = {}  # keys without the outer "model." wrapper
    skipped = []

    for full_key, storage_meta in metadata.state_dict_metadata.items():
        if not isinstance(storage_meta, DCP.metadata.TensorStorageMetadata):
            skipped.append(full_key)
            continue
        if not full_key.startswith("model."):
            skipped.append(full_key)
            continue
        inner_key = full_key[len("model."):]           # strip outer wrapper
        global_shape = storage_meta.size               # torch.Size – global shape
        dtype = storage_meta.properties.dtype
        inner_state_dict[inner_key] = torch.empty(global_shape, dtype=dtype,
                                                   device=args.device)

    if rank == 0:
        print(f"[convert] Allocated {len(inner_state_dict)} tensors "
              f"(skipped {len(skipped)} non-tensor entries).", flush=True)
        if skipped:
            print(f"[convert] Skipped keys: {skipped[:5]}", flush=True)

    # ── Step 2: load from DCP ───────────────────────────────────────────────
    wrapped = {"model": inner_state_dict}

    try:
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        planner = DefaultLoadPlanner(flatten_sharded_tensors=True)
    except (ImportError, TypeError):
        planner = None

    DCP.load(
        state_dict=wrapped,
        storage_reader=reader,
        planner=planner,
    )

    if rank == 0:
        print(f"[convert] DCP load complete.  Sample tensors:", flush=True)
        for k in list(inner_state_dict.keys())[:5]:
            v = inner_state_dict[k]
            print(f"  {k:<80s} {v.shape}  {v.dtype}", flush=True)

    # ── Step 3: convert fused expert keys → individual (if present) ─────────
    hf_state_dict = _unfuse_expert_keys(inner_state_dict)
    if rank == 0 and len(hf_state_dict) != len(inner_state_dict):
        print(f"[convert] Expert key expansion: "
              f"{len(inner_state_dict)} → {len(hf_state_dict)} tensors.", flush=True)

    # ── Step 4: save safetensors (rank-0 only) ───────────────────────────────
    if rank != 0:
        return  # only rank-0 writes

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_threshold = int(args.shard_size_gb * (1024 ** 3))
    sorted_keys = sorted(hf_state_dict.keys())
    total_shards = _count_shards(sorted_keys, hf_state_dict, shard_threshold)

    index: dict = {"metadata": {}, "weight_map": {}}
    total_bytes = 0
    cur_bytes = 0
    cur_shard: dict = {}
    shard_id = 1

    for key in sorted_keys:
        tensor = hf_state_dict[key]
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.contiguous()
        nb = tensor.numel() * tensor.element_size()

        if cur_bytes > 0 and cur_bytes + nb > shard_threshold:
            fname = f"model-{shard_id:05d}-of-{total_shards:05d}.safetensors"
            save_file(cur_shard, output_dir / fname)
            for k in cur_shard:
                index["weight_map"][k] = fname
            print(f"[convert] Saved {fname} ({cur_bytes / 1024**3:.2f} GiB)", flush=True)
            shard_id += 1
            cur_shard = {}
            cur_bytes = 0

        cur_shard[key] = tensor
        cur_bytes += nb
        total_bytes += nb

    if cur_shard:
        fname = f"model-{shard_id:05d}-of-{total_shards:05d}.safetensors"
        save_file(cur_shard, output_dir / fname)
        for k in cur_shard:
            index["weight_map"][k] = fname
        print(f"[convert] Saved {fname} ({cur_bytes / 1024**3:.2f} GiB)", flush=True)

    index["metadata"]["total_size"] = total_bytes

    index_path = output_dir / "model.safetensors.index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[convert] Saved {index_path}", flush=True)

    # Empty config.json: marks the directory as HF format for our loading code.
    config_path = output_dir / "config.json"
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as f:
            json.dump({}, f)

    print(f"\n[convert] Done!  Output: {output_dir}", flush=True)
    print(f"[convert] {len(hf_state_dict)} tensors | "
          f"{total_bytes / 1024**3:.2f} GiB | {total_shards} shard(s)", flush=True)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    args = get_args()

    # Initialise a process group (required by torch.distributed.checkpoint.load).
    # When launched with torchrun this is already set up via env vars.
    if not dist.is_initialized():
        backend = "nccl" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "gloo"
        dist.init_process_group(backend=backend)

    try:
        convert(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
