import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.checkpoint_activation import checkpoint as ac_checkpoint
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import load_file, save_file

from rosetta.modeling import MultimodalDecoderLayer, MultimodalMoTDecoderLayer

# Wrap only the main multimodal transformer layers. Conditional vision modules
# are handled outside this wrapper so ranks with different modalities still run
# the same FSDP collectives.
FSDP_WRAP_LAYERS = {MultimodalDecoderLayer, MultimodalMoTDecoderLayer}


def apply_fsdp(model, sharding_strategy="HYBRID_SHARD", num_shard=8):
    """Wrap model with FSDP2. Applies inner layers first, then root."""
    if sharding_strategy == "HYBRID_SHARD":
        world = dist.get_world_size()
        if world % num_shard != 0:
            raise ValueError(
                f"HYBRID_SHARD requires world_size divisible by num_shard, "
                f"got world_size={world}, num_shard={num_shard}"
            )
        mesh = init_device_mesh("cuda", (world // num_shard, num_shard),
                                mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for m in model.modules():
        if isinstance(m, tuple(FSDP_WRAP_LAYERS)):
            fully_shard(m, mesh=mesh, mp_policy=mp)
    # Keep root parameters gathered so mixed-modality ranks do not depend on
    # per-parameter backward hooks firing for every optional branch.
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
    return model


def apply_activation_checkpointing_to_model(model):
    for m in model.modules():
        if isinstance(m, tuple(FSDP_WRAP_LAYERS)):
            ac_checkpoint(m)


class FSDPCheckpoint:

    @staticmethod
    def _expand_vocab_tensors(sd, model, logger=None):
        """Load base-vocab checkpoints into multimodal models with extra tokens."""
        model_sd = model.state_dict()
        for key in list(sd.keys()):
            if not key.endswith(("embed_tokens.weight", "lm_head.weight")):
                continue
            if key not in model_sd:
                continue

            src = sd[key]
            dst = model_sd[key]
            if src.shape == dst.shape:
                continue
            is_embedding_matrix = src.ndim == 2 and dst.ndim == 2
            same_hidden_size = is_embedding_matrix and src.shape[1] == dst.shape[1]
            checkpoint_has_base_vocab = is_embedding_matrix and src.shape[0] < dst.shape[0]
            if same_hidden_size and checkpoint_has_base_vocab:
                with torch.no_grad():
                    dst[:src.shape[0]].copy_(src.to(dtype=dst.dtype))
                del sd[key]
                if logger:
                    logger.info(f"Expanded {key}: loaded {src.shape[0]} / {dst.shape[0]} vocab rows")

    @staticmethod
    def save(ckpt_dir, step, model, optimizer, scheduler, sharding_strategy, num_shard,
             modality_batches=None, logger=None, save_optimizer=True):
        save_path = os.path.join(ckpt_dir, f"{step:07d}")
        os.makedirs(save_path, exist_ok=True)

        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if dist.get_rank() == 0:
            save_file(sd, os.path.join(save_path, "model.safetensors"))
        del sd

        rank = dist.get_rank()
        if sharding_strategy == "HYBRID_SHARD":
            shard_idx, total_shards, should_save = rank % num_shard, num_shard, rank < num_shard
        else:
            shard_idx, total_shards, should_save = rank, dist.get_world_size(), True

        # Sharded optimizer: only first replica group saves. Demo runs can skip
        # this large state and keep only model weights for evaluation.
        if save_optimizer and should_save:
            opt = get_optimizer_state_dict(model, optimizer,
                                           options=StateDictOptions(full_state_dict=False, cpu_offload=True))
            torch.save(opt, os.path.join(save_path, f"optimizer.{shard_idx:05d}-of-{total_shards:05d}.pt"))

        if dist.get_rank() == 0:
            if save_optimizer and scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            torch.save(
                {
                    "step": step,
                    "modality_batches": modality_batches or {},
                    "weights_only": not save_optimizer,
                },
                os.path.join(save_path, "train_state.pt"),
            )
            if logger:
                suffix = " (model weights only)" if not save_optimizer else ""
                logger.info(f"Saved checkpoint to {save_path}{suffix}")

        dist.barrier()

    @staticmethod
    def load_model_weights(ckpt_dir, model, logger=None):
        """Load weights from safetensors before FSDP2 wrapping."""
        single = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(single):
            sd = load_file(single, device="cpu")
        else:
            shards = sorted(f for f in os.listdir(ckpt_dir)
                            if f.endswith(".safetensors") and f != "model.safetensors")
            if not shards:
                raise FileNotFoundError(f"No safetensors found in {ckpt_dir}")
            sd = {}
            for s in shards:
                sd.update(load_file(os.path.join(ckpt_dir, s), device="cpu"))
        FSDPCheckpoint._expand_vocab_tensors(sd, model, logger=logger)
        has_meta_params = any(param.is_meta for param in model.parameters())
        msg = model.load_state_dict(sd, strict=False, assign=has_meta_params)
        meta_params = [name for name, param in model.named_parameters() if param.is_meta]
        if meta_params:
            raise RuntimeError(
                "Model still has meta parameters after loading weights: "
                f"{meta_params[:8]}"
            )
        if logger:
            logger.info(f"Loaded weights from {ckpt_dir}: {msg}")
        del sd
        return model

    @staticmethod
    def load_train_state(ckpt_dir, model, optimizer, scheduler, sharding_strategy, num_shard, logger=None):
        """Returns (optimizer, scheduler, step, modality_batches). step=0 if no checkpoint."""
        train_state = os.path.join(ckpt_dir, "train_state.pt")
        if not os.path.exists(train_state):
            return optimizer, scheduler, 0, {}

        state = torch.load(train_state, weights_only=True)
        if state.get("weights_only", False):
            raise RuntimeError(
                f"Checkpoint {ckpt_dir} only contains model weights and cannot be used with --resume. "
                "Use it for evaluation/init only, or train with optimizer saving enabled."
            )
        # state["step"] is saved as (last_completed_micro_step + 1), i.e. the next step to run.
        step = state["step"]
        modality_batches = state.get("modality_batches", {})
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        shard_idx = rank % num_shard if sharding_strategy == "HYBRID_SHARD" else rank
        total_shards = num_shard if sharding_strategy == "HYBRID_SHARD" else world

        opt_path = os.path.join(ckpt_dir, f"optimizer.{shard_idx:05d}-of-{total_shards:05d}.pt")
        if os.path.exists(opt_path):
            set_optimizer_state_dict(model, optimizer,
                                    optim_state_dict=torch.load(opt_path, map_location="cpu", weights_only=True))
        else:
            raise FileNotFoundError(
                f"Missing optimizer shard for resume: {opt_path}. "
                "Use a full training checkpoint, not a model-only checkpoint."
            )

        sched_path = os.path.join(ckpt_dir, "scheduler.pt")
        if os.path.exists(sched_path) and scheduler is not None:
            scheduler.load_state_dict(torch.load(sched_path, weights_only=True, map_location="cpu"))

        if logger:
            logger.info(f"Resumed from step {step - 1}, modality_batches={modality_batches}")
        return optimizer, scheduler, step, modality_batches


def get_latest_checkpoint(ckpt_dir):
    """Return path of latest step directory under ckpt_dir, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    dirs = sorted(d for d in os.listdir(ckpt_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(ckpt_dir, d)))
    return os.path.join(ckpt_dir, dirs[-1]) if dirs else None
