"""
Convert Qwen3-0.6B-Base (dense LM) to MoE format for upcycling experiment.

Architecture: 12 routed experts + 1 shared expert, top-2 routing.
Each expert is initialized with a COPY of the original dense FFN weights,
optionally scaled by --expert-scale to preserve the dense model's output
magnitude at initialization.

Why --expert-scale matters:
  With 1 shared + top-2 routed experts all initialized as copies of dense FFN:
    output = weighted_sum(routed) + shared ≈ FFN(h) + FFN(h) = 2×FFN(h)   [scale=1.0]
    output = weighted_sum(routed) + shared ≈ 0.5·FFN + 0.5·FFN = FFN(h)   [scale=0.5]
  Using scale=0.5 preserves the dense model's output magnitude exactly at init.

  IMPORTANT: Only down_proj is scaled. Scaling all three weight matrices (gate/up/down)
  would apply a cubic effect (0.5^3 = 0.125), reducing output to 1/8 and causing loss
  explosion. Only down_proj is the final linear projection and scales output linearly.

Usage:
    # Reproduce existing checkpoint (original behavior, no scaling):
    python scripts/convert_qwen3_dense_to_moe.py \
        --src  /path/to/Qwen3-0.6B-Base \
        --dst  /path/to/Qwen3-0.6B-Base-upcycling-moe \
        --expert-scale 1.0

    # Clean init: only down_proj × 0.5, output ≈ dense FFN at init:
    python scripts/convert_qwen3_dense_to_moe.py \
        --src  /path/to/Qwen3-0.6B-Base \
        --dst  /path/to/Qwen3-0.6B-Base-upcycling-moe-scale05 \
        --expert-scale 0.5
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


NUM_ROUTED_EXPERTS_DEFAULT = 12
NUM_SHARED_EXPERTS_DEFAULT = 1
ROUTER_INIT_STD = 0.01   # small std so routing starts nearly uniform, breaks symmetry gradually


def load_dense_checkpoint(src_path: Path) -> tuple[dict, dict]:
    """Load a HuggingFace dense checkpoint (single or multi-shard safetensors)."""
    state_dict = {}

    # Try single-file format (small models like 0.6B)
    single_file = src_path / "model.safetensors"
    if single_file.exists():
        print(f"Loading single-shard checkpoint: {single_file}")
        state_dict = load_file(single_file)
    else:
        # Multi-shard format
        index_file = src_path / "model.safetensors.index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"No safetensors checkpoint found in {src_path}")
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard in tqdm(shard_files, desc="Loading shards"):
            state_dict.update(load_file(src_path / shard))

    with open(src_path / "config.json") as f:
        config = json.load(f)

    print(f"Loaded {len(state_dict)} tensors from dense checkpoint.")
    return state_dict, config


def dense_to_moe(
    dense_state: dict,
    config: dict,
    num_routed_experts: int,
    num_shared_experts: int,
    router_init_std: float,
    expert_scale: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Convert a dense Qwen3 state dict to MoE format.

    For each layer's FFN (gate/up/down_proj):
      - Copy weights × expert_scale to num_routed_experts routed experts
      - Copy weights × expert_scale to the shared_mlp expert
      - Initialize a router gate weight with small random std

    All other weights (attention, norms, embeddings) are copied unchanged.
    """
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    generator = torch.Generator("cpu").manual_seed(seed)

    moe_state = {}
    ffn_keys = {"gate_proj", "up_proj", "down_proj"}

    for key, tensor in tqdm(dense_state.items(), desc="Converting weights"):
        parts = key.split(".")

        # Check if this is an FFN weight: model.layers.{l}.mlp.{gate/up/down}_proj.weight
        if (
            len(parts) >= 5
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[3] == "mlp"
            and parts[4] in ffn_keys
        ):
            layer_idx = parts[2]
            proj_name = parts[4]   # gate_proj / up_proj / down_proj
            suffix = ".".join(parts[5:])  # usually "weight"

            # Only scale down_proj: it is the final linear projection, so scaling it
            # directly controls output magnitude without the cubic effect of scaling
            # all three matrices (gate/up/down would give 0.5^3 = 0.125 × output).
            scale = expert_scale if (expert_scale != 1.0 and proj_name == "down_proj") else 1.0
            expert_tensor = tensor.clone() * scale

            # Copy to each routed expert
            for expert_idx in range(num_routed_experts):
                expert_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.{suffix}"
                moe_state[expert_key] = expert_tensor.clone()

            # Copy to shared expert(s). Current Rosetta DeepSeek-MoE names this module shared_mlp.
            if num_shared_experts == 1:
                shared_key = f"model.layers.{layer_idx}.mlp.shared_mlp.{proj_name}.{suffix}"
                moe_state[shared_key] = expert_tensor.clone()
            else:
                for si in range(num_shared_experts):
                    shared_key = f"model.layers.{layer_idx}.mlp.shared_mlp.{si}.{proj_name}.{suffix}"
                    moe_state[shared_key] = expert_tensor.clone()
        elif (
            len(parts) >= 6
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[3] == "self_attn"
            and parts[4] in {"q_norm", "k_norm"}
        ):
            # Qwen3 HF names QK norm as q_norm/k_norm; Rosetta names them query/key_layernorm.
            layer_idx = parts[2]
            norm_name = "query_layernorm" if parts[4] == "q_norm" else "key_layernorm"
            suffix = ".".join(parts[5:])
            moe_state[f"model.layers.{layer_idx}.self_attn.{norm_name}.{suffix}"] = tensor
        else:
            # All non-FFN weights: copy as-is
            moe_state[key] = tensor

    # Add router (gate) weights for each layer — shape [num_routed_experts, hidden_size]
    print(f"\nInitializing {num_layers} router gates (std={router_init_std})...")
    for layer_idx in range(num_layers):
        gate_key = f"model.layers.{layer_idx}.mlp.gate.wg.weight"
        gate_weight = torch.empty(num_routed_experts, hidden_size, dtype=torch.float32)
        torch.nn.init.normal_(gate_weight, mean=0.0, std=router_init_std, generator=generator)
        moe_state[gate_key] = gate_weight.to(tensor.dtype)

    print(f"\nConversion complete: {len(dense_state)} -> {len(moe_state)} tensors.")
    if expert_scale == 0.5:
        print("  expert_scale=0.5: init output ≈ FFN(h) (dense magnitude preserved).")
    elif expert_scale == 1.0:
        print("  expert_scale=1.0: init output ≈ 2×FFN(h) (routed + shared both full scale).")
    else:
        print(f"  expert_scale={expert_scale}")
    return moe_state


def build_moe_config(config: dict, num_routed_experts: int, num_shared_experts: int) -> dict:
    """Update config.json to reflect the MoE architecture."""
    moe_config = dict(config)

    moe_config["num_experts"] = num_routed_experts
    moe_config["num_experts_per_tok"] = 2          # top-2 routing
    moe_config["moe_intermediate_size"] = config["intermediate_size"]  # expert FFN = original dense FFN
    moe_config["shared_expert_intermediate_size"] = config["intermediate_size"]
    moe_config["num_shared_expert"] = num_shared_experts
    moe_config["use_mixed_mlp_moe"] = num_shared_experts > 0
    moe_config["moe_topk"] = 2
    # Remove dense-only fields
    moe_config.pop("intermediate_size", None)

    return moe_config


def save_hf_checkpoint(state_dict: dict, config: dict, dst_path: Path, shard_size_gb: float = 5.0):
    """Save state dict as HuggingFace sharded safetensors with index file."""
    dst_path.mkdir(parents=True, exist_ok=True)

    shard_size_bytes = int(shard_size_gb * 1024 ** 3)

    # Split into shards
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
        shard_filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        shard_path = dst_path / shard_filename
        print(f"  Saving {shard_filename} ({len(shard)} tensors)...")
        save_file(shard, shard_path)
        for key in shard:
            weight_map[key] = shard_filename
            total_size += shard[key].numel() * shard[key].element_size()

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = dst_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Saved index: {index_path}")

    # Write config
    config_path = dst_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config: {config_path}")


def copy_tokenizer_files(src_path: Path, dst_path: Path):
    """Copy tokenizer files from source to destination checkpoint directory."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]
    for fname in tokenizer_files:
        src_file = src_path / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_path / fname)
            print(f"  Copied {fname}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="Path to Qwen3-0.6B-Base HF checkpoint dir")
    parser.add_argument("--dst", required=True, type=Path, help="Output path for converted MoE checkpoint")
    parser.add_argument("--num-routed-experts", type=int, default=NUM_ROUTED_EXPERTS_DEFAULT)
    parser.add_argument("--num-shared-experts", type=int, default=NUM_SHARED_EXPERTS_DEFAULT)
    parser.add_argument("--expert-scale", type=float, default=1.0,
                        help=(
                            "Scale factor applied to all expert FFN weights (default: 1.0). "
                            "Use 0.5 so that routed_sum + shared ≈ FFN(h) at init instead of 2×FFN(h). "
                            "Set to 1.0 to reproduce the original checkpoint (no scaling)."
                        ))
    parser.add_argument("--init-std", type=float, default=ROUTER_INIT_STD,
                        help="Std for router gate initialization (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size-gb", type=float, default=5.0,
                        help="Shard size in GB for output safetensors files")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Source         : {args.src}")
    print(f"Target         : {args.dst}")
    print(f"Routed experts : {args.num_routed_experts}")
    print(f"Shared experts : {args.num_shared_experts}")
    print(f"Expert scale   : {args.expert_scale}")
    print(f"Router init std: {args.init_std}")
    print(f"{'='*60}\n")

    # 1. Load dense checkpoint
    dense_state, config = load_dense_checkpoint(args.src)

    # 2. Convert to MoE
    moe_state = dense_to_moe(
        dense_state, config,
        num_routed_experts=args.num_routed_experts,
        num_shared_experts=args.num_shared_experts,
        router_init_std=args.init_std,
        expert_scale=args.expert_scale,
        seed=args.seed,
    )

    # 3. Build updated config
    moe_config = build_moe_config(config, args.num_routed_experts, args.num_shared_experts)

    # 4. Save HF-format checkpoint
    print(f"\nSaving to {args.dst} ...")
    save_hf_checkpoint(moe_state, moe_config, args.dst, shard_size_gb=args.shard_size_gb)

    # 5. Copy tokenizer files
    print("\nCopying tokenizer files...")
    copy_tokenizer_files(args.src, args.dst)

    print(f"\nDone. Converted checkpoint saved to: {args.dst}")
    print(f"Total tensors: {len(moe_state)}")
    print("\nNext step: run training with")


if __name__ == "__main__":
    main()
