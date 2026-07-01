import argparse
import json
import os
import re
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor

from rosetta.utils import set_reproducibility
from rosetta.autoencoder import VAE_META_INFO, load_vae as _load_vae_impl
from rosetta.visual_encoder import VISION_ENCODER_META_INFO
from rosetta.configuration import parse_argv_from_yaml
from train.dataset import (
    LMDataset, MMUDataset, T2IDataset, PackedDataset, Preprocessor,
)
from rosetta.modeling import build_model
from rosetta.tokenizer import load_tokenizer
from train.fsdp_utils import (
    apply_fsdp, apply_activation_checkpointing_to_model,
    FSDPCheckpoint, get_latest_checkpoint,
)
from train.maop import SharedExpertOrthManager


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("training")
    g.add_argument("--ckpt-dir", type=str, required=True,
                   help="Init checkpoint dir (HF safetensors)")
    g.add_argument("--output-dir", type=str, required=True)
    g.add_argument("--max-steps", type=int, default=10000)
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.95)
    g.add_argument("--min-lr", type=float, default=1e-6,
                   help="Cosine decay floor LR")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--lr-decay-steps", type=int, default=None,
                   help="Cosine decay horizon in optimizer steps; defaults to --max-steps")
    g.add_argument("--save-interval", type=int, default=2500)
    g.add_argument("--log-interval", type=int, default=10)
    g.add_argument("--gradient-accumulation-steps", type=int, default=1)
    g.add_argument("--sharding-strategy", type=str, default="HYBRID_SHARD",
                   choices=["HYBRID_SHARD", "FULL_SHARD"])
    g.add_argument("--num-shard", type=int, default=8,
                   help="GPUs per shard group for HYBRID_SHARD")
    g.add_argument("--activation-checkpointing", action="store_true")
    # MAOP
    g.add_argument("--use-orth", action="store_true",
                   help="Enable MAOP for shared experts")
    g.add_argument("--shield-step", type=int, default=0,
                   help="Warmup steps where non-text gradients to shared expert are blocked")
    # Differential LR
    g.add_argument("--use-lr-diff", action="store_true",
                   help="Use different LR for VAE-related parameters")
    g.add_argument("--vae-lr", type=float, default=1e-4,
                   help="LR for VAE-related parameters (used with --use-lr-diff)")
    g.add_argument("--lm-data", type=str,
                   default="example_data/lm/conversation_58k.json")
    g.add_argument("--mmu-data", type=str,
                   default="example_data/mmu/llava_ov_si.jsonl")
    g.add_argument("--mmu-image-dir", type=str,
                   default="example_data/mmu")
    g.add_argument("--t2i-data", type=str,
                   default="example_data/t2i")
    g.add_argument("--data-weights", type=str,
                   default='{"t2i": 0.60, "mmu": 0.25, "lm": 0.15}',
                   help='JSON dict of sampling weights, e.g. \'{"lm":0.15,"mmu":0.25,"t2i":0.60}\''
                        ' DATA_ARGS.sampling-probs')
    g.add_argument("--max-seq-len", type=int, default=2048)
    g.add_argument("--image-loss-weight", type=float, default=1.0)
    g.add_argument("--moe-aux-loss-coeff", type=float, default=0.01,
                   help="MoE auxiliary loss coefficient")
    g.add_argument("--init-save", action="store_true",
                   help="Save a checkpoint at step 0 before training starts")
    g.add_argument("--no-save-optimizer", action="store_true",
                   help="Save model weights only; checkpoints cannot be used with --resume")
    return parser


def parse_args():

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("config", nargs="?", default=None)
    known, remaining = pre.parse_known_args()

    from rosetta.configuration import add_core_args
    parser = argparse.ArgumentParser(description="Rosetta Stage 3 Training")
    parser = add_core_args(parser)
    parser = add_train_args(parser)

    argv = []
    frozen_args = {}
    if known.config:
        argv, frozen_args = parse_argv_from_yaml(known.config, allow_frozen=True)

    args, _ = parser.parse_known_args(argv + remaining)
    for k, v in frozen_args.items():
        setattr(args, k, v)

    return args


@torch.no_grad()
def vae_encode(vae, pil_image, device, scaling_factor: float):
    img = TF.to_tensor(pil_image).unsqueeze(0).to(device)
    img = img * 2.0 - 1.0
    with torch.no_grad():
        encoded = vae.encode(img)
    latent = encoded if isinstance(encoded, torch.Tensor) else encoded.latent_dist.sample()
    if scaling_factor:
        latent = latent * scaling_factor
    return latent.detach()


class RectifiedFlow:
    def __init__(self, snr_type: str = "lognorm"):
        self.snr_type = snr_type

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        if self.snr_type == "lognorm":
            u = torch.randn(batch_size, device=device)
            return torch.sigmoid(u)
        return torch.rand(batch_size, device=device)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        x_1 = torch.randn_like(x_0)
        t_bc = t.view(-1, *([1] * (x_0.ndim - 1)))
        x_t = (1.0 - t_bc) * x_0 + t_bc * x_1
        u_t = x_1 - x_0
        model_t = t * 1000.0
        return x_t, u_t, model_t

    def training_losses_fn(self, t, x0, xt, ut, model_output) -> dict:
        if isinstance(model_output, torch.Tensor) and isinstance(ut, torch.Tensor):
            return {"loss": ((model_output - ut) ** 2).mean()}

        losses = []
        preds = model_output if isinstance(model_output, list) else [model_output]
        refs = ut if isinstance(ut, list) else [ut]
        for pred_i, ref_i in zip(preds, refs):
            if isinstance(pred_i, torch.Tensor):
                losses.append(((pred_i - ref_i) ** 2).mean())
            else:
                for p, r in zip(pred_i, ref_i):
                    losses.append(((p - r) ** 2).mean())
        return {"loss": torch.stack(losses).mean()}


def _build_packed_attention_mask(batch, sample_offsets, seqlen, device):
    attention_mask = torch.zeros((1, 1, seqlen, seqlen), dtype=torch.bool, device=device)
    offsets = sample_offsets[0].tolist()
    for start, end in zip(offsets[:-1], offsets[1:]):
        if start >= end:
            continue
        attention_mask[0, 0, start:end, start:end] = torch.ones(
            (end - start, end - start), dtype=torch.bool, device=device
        ).tril()

    # Keep padded query rows finite. Real tokens cannot attend to right-side pads.
    if offsets and offsets[-1] < seqlen:
        start, end = offsets[-1], seqlen
        attention_mask[0, 0, start:end, start:end] = torch.ones(
            (end - start, end - start), dtype=torch.bool, device=device
        ).tril()

    full_attn_slices = []
    for key in ("image_slices", "cond_full_attn_slices"):
        if key in batch:
            full_attn_slices.extend(batch[key][0])
    for sli in full_attn_slices:
        start = max(0, min(int(sli.start), seqlen))
        end = max(start, min(int(sli.stop), seqlen))
        if start < end:
            attention_mask[0, 0, start:end, start:end] = True
    return attention_mask


def prepare_batch(batch, vae, flow: RectifiedFlow, device, image_loss_weight, vae_scaling_factor):
    tokens      = batch["tokens"].to(device)[:, :-1].contiguous()
    target      = batch["target_tokens"].to(device)[:, 1:].contiguous()
    text_mask   = batch["text_mask"].to(device)[:, 1:].contiguous()
    seqlen      = tokens.shape[1]
    sample_offsets = [batch["offsets"][0].clamp(0, seqlen).to(device)]
    for boundary in sample_offsets[0][1:-1].tolist():
        if 0 < boundary <= seqlen:
            target[:, boundary - 1] = -100
            text_mask[:, boundary - 1] = 0.0

    dataset_tag = batch.get("dataset_tag", [None])[0]

    kwargs = dict(
        input_ids=tokens,
        sample_offsets=sample_offsets,
        attention_mask=_build_packed_attention_mask(batch, sample_offsets, seqlen, device),
    )
    loss_inputs = dict(
        target=target,
        text_mask=text_mask,
        diffusion_loss_fn=None,
        image_loss_weight=image_loss_weight,
        dataset_tag=dataset_tag,
    )

    if "image_mask" in batch:
        kwargs["image_mask"] = batch["image_mask"].to(device)[:, :-1]
    if "cond_vit_image_mask" in batch:
        kwargs["cond_vit_image_mask"] = batch["cond_vit_image_mask"].to(device)[:, :-1]

    if "cond_vit_images" in batch:
        cvi = batch["cond_vit_images"]
        if isinstance(cvi, torch.Tensor):
            kwargs["cond_vit_images"] = cvi.to(device)
        else:
            kwargs["cond_vit_images"] = [[img.to(device) for img in imgs] for imgs in cvi]
        if "cond_vit_image_kwargs" in batch:
            ck = batch["cond_vit_image_kwargs"]
            grid = ck.get("grid_thw")
            if isinstance(grid, torch.Tensor):
                kwargs["cond_vit_image_kwargs"] = {"grid_thw": grid.to(device)}
            else:
                kwargs["cond_vit_image_kwargs"] = {
                    "grid_thw": [[g.to(device) for g in grp] for grp in grid]
                }

    if "images" in batch:
        pil_images = batch["images"][0]

        x0_list, xt_list, ut_list, t_list, mt_list = [], [], [], [], []
        for pil in pil_images:
            x0 = vae_encode(vae, pil, device, vae_scaling_factor)
            t  = flow.sample_t(1, device=device)
            xt, ut, model_t = flow.add_noise(x0.float(), t)
            x0_list.append(x0.squeeze(0))
            xt_list.append(xt.squeeze(0))
            ut_list.append(ut.squeeze(0))
            t_list.append(t)
            mt_list.append(model_t.squeeze())

        model_t_tensor = torch.stack(mt_list)
        kwargs["images"]    = [xt_list]
        kwargs["timesteps"] = [model_t_tensor]
        loss_inputs["diffusion_loss_fn"] = partial(
            flow.training_losses_fn,
            t=torch.cat(t_list),
            x0=x0_list,
            xt=xt_list,
            ut=[ut_list],
        )

    rope_combined = []
    if "rope_image_info" in batch:
        rope_combined.extend(batch["rope_image_info"][0])
    if "rope_media_info" in batch:
        rope_combined.extend(batch["rope_media_info"][0])
    if rope_combined:
        rope_combined.sort(key=lambda x: x[0].start)
        kwargs["rope_image_info"] = [rope_combined]

    return kwargs, loss_inputs


def _moe_cumsum_exclusive(t, dim=-3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return torch.nn.functional.pad(t, (*pre_padding, 1, -1)).cumsum(dim=dim)


def _moe_score(cfg, logits):
    return torch.nn.functional.softmax(logits.float(), dim=1)


def _modality_balance_loss(scores, topk_idx, num_local_experts):
    fi = torch.nn.functional.one_hot(topk_idx.view(-1), num_classes=num_local_experts).float().mean(dim=0)
    fi = fi * num_local_experts
    pi = scores.mean(dim=0)
    return (fi * pi).sum()


def _compute_moe_balance_loss(moe_layer, hidden_states, topk_idx, token_modalities=None):
    cfg = moe_layer._config
    if not moe_layer.training or not cfg.moe_aux_loss:
        return None

    bsz, seqlen = hidden_states.shape[:2]
    if bsz * seqlen == 0:
        return None
    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))

    if cfg.use_modality_routing:
        gate = moe_layer.gate
        if token_modalities is None:
            token_modalities = torch.zeros(bsz * seqlen, dtype=torch.long, device=hidden_states.device)
        flat_mod = token_modalities.reshape(-1)
        balance_losses = []
        for mod_id, wg, offset, n_local in (
            (0, gate.wg_text, 0, cfg.num_text_experts),
            (1, gate.wg_vit, cfg.num_text_experts, cfg.num_vit_experts),
            (2, gate.wg_vae, cfg.num_text_experts + cfg.num_vit_experts, cfg.num_vae_experts),
        ):
            mask = flat_mod == mod_id
            if not mask.any():
                continue
            logits_m = wg(flat_hidden[mask].to(wg.weight.dtype))
            scores_m = _moe_score(cfg, logits_m)
            local_topk_idx = topk_idx[mask] - offset
            balance_losses.append(_modality_balance_loss(scores_m, local_topk_idx, n_local))
        return (sum(balance_losses) / len(balance_losses)) if balance_losses else None

    gate = moe_layer.gate
    logits = gate.wg(flat_hidden.to(gate.wg.weight.dtype))
    scores = torch.nn.functional.softmax(logits, dim=1)
    fi = torch.nn.functional.one_hot(topk_idx.view(-1), num_classes=moe_layer.num_experts).float().mean(dim=0)
    fi = fi * moe_layer.num_experts
    pi = scores.mean(dim=0)
    return (fi * pi).sum()


@torch.no_grad()
def _compute_moe_capacity_metrics(moe_layer, hidden_states, topk_idx, token_modalities=None):
    bsz, seqlen = hidden_states.shape[:2]
    cfg = moe_layer._config
    top_k = moe_layer.top_k
    min_expert_capacity = 4
    if bsz * seqlen == 0:
        return None, None

    if cfg.use_modality_routing and token_modalities is not None:
        flat_mod = token_modalities.reshape(-1)
        modality_rates = {}
        modality_counts = {}
        specs = (
            (0, "text", 0, cfg.num_text_experts),
            (1, "vit", cfg.num_text_experts, cfg.num_vit_experts),
            (2, "vae", cfg.num_text_experts + cfg.num_vit_experts, cfg.num_vae_experts),
        )
        for mod_id, name, offset, n_local in specs:
            mask = flat_mod == mod_id
            if not mask.any():
                modality_rates[name] = torch.tensor(1.0, device=topk_idx.device)
                continue
            ti_m = topk_idx[mask] - offset
            n_m = ti_m.shape[0]
            ec_m = max(min_expert_capacity, min(n_m, int(top_k * n_m * cfg.capacity_factor / n_local)))
            oh = torch.nn.functional.one_hot(ti_m.transpose(0, 1), n_local).float()
            oh_cs = _moe_cumsum_exclusive(oh, dim=-2)
            prev = 0.
            for k in range(top_k):
                pos_k = (oh_cs[k] + prev) * oh[k]
                oh[k] *= (pos_k < float(ec_m)).float()
                prev = oh[k].sum(dim=-2, keepdim=True) + prev
            modality_rates[name] = oh.sum() / (n_m * top_k)
            modality_counts[name] = n_m

        total_routed = sum(modality_counts.values())
        if total_routed > 0:
            capacity_rate = sum(
                modality_rates[name] * (modality_counts[name] / total_routed)
                for name in modality_counts
            )
        else:
            capacity_rate = torch.tensor(1.0, device=topk_idx.device)
        return capacity_rate, modality_rates

    expert_capacity = bsz * min(seqlen, int(top_k * (seqlen * cfg.capacity_factor) / moe_layer.num_experts))
    expert_capacity = max(expert_capacity, min_expert_capacity)
    mask = torch.nn.functional.one_hot(topk_idx.transpose(0, 1), moe_layer.num_experts).float()
    mask_cumsum = _moe_cumsum_exclusive(mask, dim=-2)
    prev_expert_count = 0.
    for n in range(top_k):
        position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]
        mask[n] *= (position_in_expert < float(expert_capacity)).float()
        prev_expert_count = mask[n].sum(dim=-2, keepdim=True) + prev_expert_count
    capacity_rate = mask.sum() / (bsz * seqlen * top_k)
    return capacity_rate, None


def attach_moe_training_metric_hooks(model):
    handles = []
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("gate", "experts", "top_k", "num_experts")):
            continue
        module._training_moe_metrics = {}

        def _hook(_gate, inputs, output, moe_layer=module):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            hidden_states = inputs[0]
            token_modalities = inputs[1] if len(inputs) > 1 else None
            topk_idx = output[1]
            balance_loss = _compute_moe_balance_loss(
                moe_layer, hidden_states, topk_idx, token_modalities
            )
            capacity_rate, modality_rates = _compute_moe_capacity_metrics(
                moe_layer, hidden_states, topk_idx, token_modalities
            )
            moe_layer._training_moe_metrics = {
                "balance_loss": balance_loss,
                "capacity_rate": capacity_rate,
                "modality_capacity_rates": modality_rates,
            }

        handles.append(module.gate.register_forward_hook(_hook))
    model._moe_training_metric_hook_handles = handles
    return handles


def clear_moe_training_metrics(model):
    for module in model.modules():
        if hasattr(module, "_training_moe_metrics"):
            module._training_moe_metrics = {}


def compute_training_losses(
        model,
        output,
        target,
        text_mask,
        diffusion_loss_fn=None,
        image_loss_weight=0.0,
        dataset_tag=None,
):
    logits = output.logits
    diff_pred = output.diffusion_prediction
    losses = {}
    loss = 0.0

    if text_mask is not None and text_mask.max().item() > 0:
        discrete_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), target.view(-1), ignore_index=-100, reduction="mean"
        )
        if dataset_tag is not None:
            losses[f"{dataset_tag}_text_loss"] = discrete_loss.detach()
        else:
            losses["text_loss"] = discrete_loss.detach()
        loss = discrete_loss
    elif logits is not None:
        loss_key = f"{dataset_tag}_text_loss" if dataset_tag is not None else "text_loss"
        zero_loss = logits.sum() * 0.0
        loss_value = zero_loss
        loss = zero_loss
        losses[loss_key] = loss_value.detach()

    def _accumulate_diffusion_loss(pred, diffusion_loss_fn, loss_key, loss_weight, loss):
        if pred is None:
            return loss
        raw_diff_loss = diffusion_loss_fn(model_output=pred)["loss"]
        diff_loss = raw_diff_loss.mean()
        if loss_weight > 0:
            losses[loss_key] = diff_loss.detach()
        loss = loss + loss_weight * diff_loss
        return loss

    loss = _accumulate_diffusion_loss(
        pred=diff_pred,
        diffusion_loss_fn=diffusion_loss_fn,
        loss_key=f"{dataset_tag}_image_loss" if dataset_tag is not None else "image_loss",
        loss_weight=image_loss_weight,
        loss=loss,
    )

    if model._config.moe_aux_loss and model.moe_aux_loss_coeff > 0:
        def _metric(module, key):
            return getattr(module, "_training_moe_metrics", {}).get(key)

        moe_layers = [
            block.mlp for block in model.model["layers"]
            if hasattr(block.mlp, "_training_moe_metrics")
        ]
        moe_aux_losses = [
            value for layer in moe_layers
            if (value := _metric(layer, "balance_loss")) is not None
        ]
        assert len(moe_aux_losses) > 0, "No MoE losses found across the model layers."
        moe_aux_loss = sum(moe_aux_losses)
        loss = loss + moe_aux_loss * model.moe_aux_loss_coeff
        losses["moe_loss"] = moe_aux_loss.detach() / len(moe_aux_losses)

        capacity_rates = [
            value for layer in moe_layers
            if (value := _metric(layer, "capacity_rate")) is not None
        ]
        assert len(capacity_rates) > 0, "No capacity losses found across the model layers."
        losses["capacity_rate"] = sum(capacity_rates) / len(capacity_rates)

        if model._config.use_modality_routing:
            for mod_name in ('text', 'vit', 'vae'):
                rates = []
                for m in moe_layers:
                    mod_rates = _metric(m, "modality_capacity_rates")
                    if mod_rates is not None and mod_name in mod_rates:
                        rates.append(mod_rates[mod_name])
                if rates:
                    losses[f"capacity_rate_{mod_name}"] = sum(rates) / len(rates)

        if model._config.use_mot and image_loss_weight > 0:
            moe_layers_mot = [
                block.mlp_mot_gen for block in model.model["layers"]
                if hasattr(block.mlp_mot_gen, "_training_moe_metrics")
            ]
            moe_aux_losses_mot = [
                value for layer in moe_layers_mot
                if (value := _metric(layer, "balance_loss")) is not None
            ]
            if moe_aux_losses_mot:
                moe_aux_loss_mot = sum(moe_aux_losses_mot)
                loss = loss + moe_aux_loss_mot * model.moe_aux_loss_coeff
                losses["moe_loss_mot_gen"] = moe_aux_loss_mot.detach() / len(moe_aux_losses_mot)

            capacity_rates_mot = [
                value for layer in moe_layers_mot
                if (value := _metric(layer, "capacity_rate")) is not None
            ]
            if capacity_rates_mot:
                losses["capacity_rate_mot_gen"] = sum(capacity_rates_mot) / len(capacity_rates_mot)

    losses["loss"] = loss
    return losses


def build_optimizer(model, args, model_config=None):
    if args.use_lr_diff and model_config is not None and getattr(model_config, 'use_modality_routing', False):
        n_text = getattr(model_config, 'num_text_experts', 0)
        n_vit  = getattr(model_config, 'num_vit_experts',  0)
        vae_start = n_text + n_vit

        vae_params, default_params = [], []
        for name, p in sorted(model.named_parameters(), key=lambda t: t[0]):
            if not p.requires_grad:
                continue
            clean = name.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')
            bucket = 'default'
            if 'gate.wg_vae' in clean:
                bucket = 'vae'
            elif 'mlp.experts.' in clean:
                m = re.search(r'mlp\.experts\.(\d+)\.', clean)
                if m and int(m.group(1)) >= vae_start:
                    bucket = 'vae'
            if bucket == 'vae':
                vae_params.append(p)
            else:
                default_params.append(p)

        param_groups = [
            {"params": vae_params,     "lr": args.vae_lr, "name": "vae"},
            {"params": default_params, "lr": args.lr,     "name": "default"},
        ]
        logger.info(f"Differential LR: main={args.lr} ({len(default_params)} params), "
                    f"vae={args.vae_lr} ({len(vae_params)} params, experts>={vae_start})")
    else:
        if args.use_lr_diff:
            logger.warning("--use-lr-diff ignored: use_modality_routing=False or model_config unavailable")
        param_groups = [
            {
                "params": [p for _, p in sorted(model.named_parameters(), key=lambda t: t[0]) if p.requires_grad],
                "lr": args.lr,
                "name": "default",
            }
        ]

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay,
                             betas=(args.adam_beta1, args.adam_beta2),
                             foreach=False, fused=False)


def build_scheduler(optimizer, args):
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else args.max_steps
    decay_steps = max(int(decay_steps), args.warmup_steps + 1)
    min_ratios = [
        min(args.min_lr, group["lr"]) / group["lr"]
        for group in optimizer.param_groups
    ]

    def make_lr_lambda(min_ratio):
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = min(1.0, (step - args.warmup_steps) / max(1, decay_steps - args.warmup_steps))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_ratio, cosine)
        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [make_lr_lambda(min_ratio) for min_ratio in min_ratios],
    )


def _to_local_tensor(t: torch.Tensor) -> torch.Tensor:
    if hasattr(t, "to_local"):
        return t.to_local()
    data = getattr(t, "data", None)
    if data is not None and hasattr(data, "to_local"):
        return data.to_local()
    return t


@torch.no_grad()
def _get_fsdp_shard_pg(param):
    src = None
    if hasattr(param, 'device_mesh') and param.device_mesh is not None:
        src = param.device_mesh
    elif param.grad is not None and hasattr(param.grad, 'device_mesh'):
        src = param.grad.device_mesh
    data = getattr(param, "data", None)
    if src is None and data is not None and hasattr(data, 'device_mesh'):
        src = data.device_mesh
    if src is None:
        return None
    return src.get_group(src.ndim - 1)


@torch.no_grad()
def _clip_grad_local_shards(model, max_norm: float, eps: float = 1e-6) -> float:
    params = [p for _, p in sorted(model.named_parameters(), key=lambda t: t[0]) if p.requires_grad]
    total_sq = torch.zeros((), device="cuda", dtype=torch.float32)
    reduce_group = None
    for p in params:
        if p.grad is None:
            continue
        g = _to_local_tensor(p.grad)
        total_sq = total_sq + g.float().pow(2).sum()
        if reduce_group is None:
            reduce_group = _get_fsdp_shard_pg(p)

    if dist.is_initialized():
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=reduce_group)
    total_norm = total_sq.sqrt()
    clip_coef = float(max_norm) / (total_norm.item() + eps)
    if clip_coef < 1.0:
        scale = total_norm.new_tensor(clip_coef)
        for p in params:
            if p.grad is None:
                continue
            grad_local = _to_local_tensor(p.grad)
            grad_local.mul_(scale)
    return float(total_norm.item())


@torch.no_grad()
def _local_adamw_step(optimizer) -> None:
    for group in optimizer.param_groups:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue

            param = _to_local_tensor(p)
            grad = _to_local_tensor(p.grad)
            if grad.is_sparse:
                raise RuntimeError("Sparse gradients are not supported by local AdamW")

            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            exp_avg = _to_local_tensor(state["exp_avg"])
            exp_avg_sq = _to_local_tensor(state["exp_avg_sq"])
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq
            state["step"] += 1
            step = state["step"]

            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            grad = grad.to(param.dtype)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)


def _fixed_modality_allocation(weights, keys, rank, world_size):
    if world_size < len(keys):
        raise ValueError(
            f"fixed modality allocation requires world_size >= num modalities: "
            f"{world_size} < {len(keys)} ({keys})"
        )
    total = sum(weights)
    normalized = [w / total for w in weights]
    allocated = [1] * len(keys)
    remaining = world_size - len(keys)
    targets = [w * remaining for w in normalized]
    floors = [int(t) for t in targets]
    allocated = [a + f for a, f in zip(allocated, floors)]
    leftover = world_size - sum(allocated)
    remainders = sorted(
        enumerate([t - f for t, f in zip(targets, floors)]),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(leftover):
        allocated[remainders[i][0]] += 1

    cum = [0]
    for n in allocated:
        cum.append(cum[-1] + n)
    for idx, (lo, hi) in enumerate(zip(cum[:-1], cum[1:])):
        if lo <= rank < hi:
            return keys[idx], rank - lo, allocated[idx], allocated
    raise RuntimeError(f"rank {rank} not covered by allocation {allocated}")


def _weighted_modality_schedule(weights, keys, schedule_len):
    total = sum(weights)
    normalized = [w / total for w in weights]
    raw = [w * schedule_len for w in normalized]
    allocation = [max(1, int(v)) for v in raw]
    while sum(allocation) > schedule_len:
        reducible = [i for i, count in enumerate(allocation) if count > 1]
        idx = max(
            reducible,
            key=lambda i: (allocation[i] - raw[i], allocation[i]),
        )
        allocation[idx] -= 1
    while sum(allocation) < schedule_len:
        idx = max(
            range(len(allocation)),
            key=lambda i: raw[i] - allocation[i],
        )
        allocation[idx] += 1

    scores = [0] * len(keys)
    schedule = []
    for _ in range(schedule_len):
        scores = [score + count for score, count in zip(scores, allocation)]
        idx = max(range(len(keys)), key=lambda i: scores[i])
        schedule.append(keys[idx])
        scores[idx] -= schedule_len
    return schedule

def _build_modality_mix_schedule(weights, keys, world_size, schedule_len=None):
    if world_size < len(keys):
        schedule_len = max(len(keys), schedule_len or len(keys))
        return _weighted_modality_schedule(weights, keys, schedule_len)

    _, _, _, allocation = _fixed_modality_allocation(weights, keys, 0, world_size)
    schedule = []
    for key, count in zip(keys, allocation):
        schedule.extend([key] * count)
    return schedule

def _shard_group_info(rank: int, world: int, args):
    if args.sharding_strategy == "HYBRID_SHARD":
        group_size = min(args.num_shard, world)
        return rank // group_size, max(1, world // group_size), group_size
    return 0, 1, world


def _all_reduce_metrics(running_loss_dict, running_step_dict,
                        running_samples_dict, running_tokens_dict,
                        consumed_samples_total, consumed_tokens_total,
                        dp_group, loss_names=None, data_keys=None):
    world = dist.get_world_size(group=dp_group) if dist.is_initialized() else 1

    if loss_names is None:
        all_loss_key_sets = [None] * world
        dist.all_gather_object(all_loss_key_sets, sorted(running_loss_dict.keys()),
                               group=dp_group)
        loss_names = sorted(set(k for ks in all_loss_key_sets for k in ks))

    if data_keys is None:
        all_data_key_sets = [None] * world
        dist.all_gather_object(all_data_key_sets, sorted(running_samples_dict.keys()),
                               group=dp_group)
        data_keys = sorted(set(k for ks in all_data_key_sets for k in ks))

    if not loss_names and not data_keys:
        return {}

    loss_values      = [running_loss_dict.get(k, 0.0)  for k in loss_names]
    loss_counts      = [float(running_step_dict.get(k, 0)) for k in loss_names]
    consumed_samples = [running_samples_dict.get(k, 0) for k in data_keys]
    consumed_tokens  = [running_tokens_dict.get(k, 0)  for k in data_keys]

    sum_reduced = torch.tensor(
        loss_values + loss_counts + consumed_samples + consumed_tokens,
        device="cuda", dtype=torch.float64,
    )
    dist.all_reduce(sum_reduced, group=dp_group)
    n, m = len(loss_names), len(data_keys)
    sum_losses, sum_counts, sum_smp, sum_tks = torch.split(
        sum_reduced, [n, n, m, m]
    )
    sum_counts = sum_counts.clamp(min=1.0)  # avoid division by zero

    loss_reduced = {}
    for name, log_loss, count in zip(loss_names, sum_losses, sum_counts):
        if log_loss != 0:
            loss_reduced[name] = (log_loss / count).item()

    for name, smp_cnt, tok_cnt in zip(data_keys, sum_smp.tolist(), sum_tks.tolist()):
        consumed_samples_total[name] = consumed_samples_total.get(name, 0) + int(smp_cnt)
        consumed_tokens_total[name]  = consumed_tokens_total.get(name, 0)  + int(tok_cnt)

    return loss_reduced


def _aggregate_modality_batch_counts(local_counts: dict) -> dict:
    if not dist.is_initialized():
        return dict(local_counts)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(local_counts))
    keys = sorted({key for counts in gathered for key in counts})
    return {key: max(int(counts.get(key, 0)) for counts in gathered) for key in keys}


def _training_log(logger, step, train_iters, loss_reduced, duration,
                  lr_list, consumed_samples_total, consumed_tokens_total,
                  tb_writer=None):
    grad_norm = loss_reduced.pop("grad_norm", -1)
    log_str = (
        f"iteration {step:>8d}/{train_iters}"
        f" | grad_norm: {grad_norm:>8.3f}"
        f" | speed: {1 / duration:>6.2f} it/s, {duration:>6.2f} s/it"
    )
    for i, lr in enumerate(lr_list, start=1):
        order = "" if i == 1 else f" {i}"
        log_str += f" | learning rate{order}: {lr:.6E}"
    for name, loss in loss_reduced.items():
        log_str += f" | {name}: {loss:.6E}"
    for name, samples in consumed_samples_total.items():
        log_str += f" | {name} smp: {samples:>14,}"
    for name, tokens in consumed_tokens_total.items():
        log_str += f" | {name} tks: {tokens:>18,}"
    logger.info(log_str)

    if tb_writer is not None:
        tb_writer.add_scalar("Speed/steps_per_sec", 1 / duration, step)
        tb_writer.add_scalar("Speed/seconds_per_step", duration, step)
        tb_writer.add_scalar("Gradient/grad_norm", grad_norm, step)
        for i, lr in enumerate(lr_list, start=1):
            order = "" if i == 1 else f" {i}"
            tb_writer.add_scalar(f"Learning Rate/lr{order}", lr, step)
        for name, loss in loss_reduced.items():
            tb_writer.add_scalar(f"Loss/{name}", loss, step)
        for name, samples in consumed_samples_total.items():
            tb_writer.add_scalar(f"Consumed Samples/{name}", samples, step)
        for name, tokens in consumed_tokens_total.items():
            tb_writer.add_scalar(f"Consumed Tokens/{name}", tokens, step)
        tb_writer.flush()


def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        logger.info(f"World size: {world}, local rank: {local_rank}")

    set_reproducibility(
        enable=args.reproduce,
        global_seed=args.seed,
        benchmark=getattr(args, "benchmark", None),
    )

    if not hasattr(args, "task_id") or args.task_id is None:
        args.task_id = f"train_{int(time.time())}"
    args.rank = rank

    if rank == 0:
        logger.info("Building model...")
    args.init_device = "cpu"
    model, model_config = build_model(args, dtype=torch.float32, device="cpu",
                                      initialize_weights=False)

    if args.shield_step > 0:
        model_config.shield_step = args.shield_step
    if hasattr(model_config, "moe_aux_loss"):
        model_config.moe_aux_loss = True

    output_dir = args.output_dir
    ckpt_dir   = os.path.join(output_dir, "ckpt")
    resume_ckpt = get_latest_checkpoint(ckpt_dir) if args.resume else None
    weight_source = resume_ckpt if resume_ckpt else args.ckpt_dir
    FSDPCheckpoint.load_model_weights(weight_source, model, logger=logger if rank == 0 else None)

    _frozen_attrs = {}
    for attr in ("vit",):
        m = getattr(model, attr, None)
        if m is not None:
            _frozen_attrs[attr] = m
            setattr(model, attr, None)

    if rank == 0:
        logger.info("Applying FSDP2...")
    model = apply_fsdp(model, sharding_strategy=args.sharding_strategy,
                       num_shard=args.num_shard)
    if args.activation_checkpointing:
        apply_activation_checkpointing_to_model(model)
    model.to(device)

    for attr, m in _frozen_attrs.items():
        setattr(model, attr, m.to(device))

    model.train()
    attach_moe_training_metric_hooks(model)

    data_weights = args.data_weights if isinstance(args.data_weights, dict) \
                   else json.loads(args.data_weights)
    mod_keys = [k for k, w in data_weights.items() if w > 0]
    needs_mmu = "mmu" in mod_keys
    needs_t2i = "t2i" in mod_keys

    vae = None
    vae_scaling_factor = 1.0
    if needs_t2i:
        vae = _load_vae_impl(getattr(args, "vae_type", "16x16-128c-flux2"))
        vae.to(device)
        vae.eval()
        vae_scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None) or 1.0

    tokenizer = load_tokenizer(args.tokenizer_name, args.tokenizer_class)
    vit_processor = None
    if needs_mmu:
        vit_meta = VISION_ENCODER_META_INFO.get(getattr(args, "vit_type", "qwen3vl-vit-for-0.6b"), {})
        vit_proc_path = vit_meta.get("path", "")
        _VIT_DS = 14 * 2
        _MIN_VIT_TOKENS = 256
        _MAX_VIT_TOKENS = 1024
        vit_processor = AutoImageProcessor.from_pretrained(
            vit_proc_path,
            min_pixels=_MIN_VIT_TOKENS * _VIT_DS * _VIT_DS,
            max_pixels=_MAX_VIT_TOKENS * _VIT_DS * _VIT_DS,
        )

    preprocessor = Preprocessor(
        tokenizer=tokenizer,
        vit_processor=vit_processor,
        vae_downsample=VAE_META_INFO.get(getattr(args, "vae_type", "16x16-128c-flux2"), {}).get(
            "downsample_factor", [16, 16])[0],
        max_seq_len=args.max_seq_len,
    )

    resume_mod_batches: dict = {}
    if resume_ckpt:
        ts = os.path.join(resume_ckpt, "train_state.pt")
        if os.path.exists(ts):
            resume_mod_batches = torch.load(ts, weights_only=True).get("modality_batches", {})
            if rank == 0 and resume_mod_batches:
                logger.info(f"Data resume: skipping batches {resume_mod_batches}")

    packed_by_mod = {}
    if "lm" in mod_keys:
        packed_by_mod["lm"] = PackedDataset(
            LMDataset(args.lm_data),
            preprocessor, args.max_seq_len, skip=resume_mod_batches.get("lm", 0)
        )
    if "mmu" in mod_keys:
        packed_by_mod["mmu"] = PackedDataset(
            MMUDataset(args.mmu_data, args.mmu_image_dir),
            preprocessor, args.max_seq_len, skip=resume_mod_batches.get("mmu", 0)
        )
    if "t2i" in mod_keys:
        packed_by_mod["t2i"] = PackedDataset(
            T2IDataset(args.t2i_data),
            preprocessor, args.max_seq_len, skip=resume_mod_batches.get("t2i", 0)
        )
    mod_probs = [data_weights[k] for k in mod_keys]
    shard_group_idx, num_shard_groups, shard_group_size = _shard_group_info(
        rank, world, args
    )
    modality_mix_schedule = None
    assigned_mod = None
    use_fixed_modality = num_shard_groups >= len(mod_keys)
    if use_fixed_modality:
        assigned_mod, _, _, group_allocation = _fixed_modality_allocation(
            mod_probs, mod_keys, shard_group_idx, num_shard_groups
        )
        if rank == 0:
            logger.info(
                f"Fixed modality allocation across {num_shard_groups} shard groups "
                f"(group_size={shard_group_size}): "
                f"{dict(zip(mod_keys, group_allocation))}"
            )
    else:
        if (
            num_shard_groups < len(mod_keys)
            and args.gradient_accumulation_steps < len(mod_keys)
        ):
            raise ValueError(
                "Mixed-modality fallback requires gradient_accumulation_steps "
                f">= number of modalities ({len(mod_keys)}), got "
                f"{args.gradient_accumulation_steps}."
            )
        modality_mix_schedule = _build_modality_mix_schedule(
            mod_probs,
            mod_keys,
            num_shard_groups,
            args.gradient_accumulation_steps,
        )
        if rank == 0:
            logger.info(
                "Mixed modality schedule per optimizer step: "
                f"{modality_mix_schedule}"
            )
    mod_loaders = {
        k: DataLoader(ds, batch_size=None, num_workers=0, pin_memory=False)
        for k, ds in packed_by_mod.items()
        if k in mod_keys
    }

    optimizer  = build_optimizer(model, args, model_config=model_config)
    scheduler  = build_scheduler(optimizer, args)
    flow       = RectifiedFlow(snr_type=getattr(args, "flow_snr_type", "lognorm"))
    maop       = SharedExpertOrthManager(model, optimizer) if args.use_orth else None

    start_step = 0
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if resume_ckpt:
        optimizer, scheduler, start_step, _ = FSDPCheckpoint.load_train_state(
            resume_ckpt, model, optimizer, scheduler,
            args.sharding_strategy, args.num_shard,
            logger=logger if rank == 0 else None,
        )
        if rank == 0:
            logger.info(f"Resumed from {resume_ckpt}, step={start_step}")

    modality_batch_counts = dict(resume_mod_batches)

    if args.init_save and start_step == 0:
        save_modality_batches = _aggregate_modality_batch_counts(modality_batch_counts)
        FSDPCheckpoint.save(ckpt_dir, 0, model, optimizer, scheduler,
                            args.sharding_strategy, args.num_shard,
                            modality_batches=save_modality_batches,
                            logger=logger if rank == 0 else None,
                            save_optimizer=not args.no_save_optimizer)

    dist.barrier()

    if rank == 0:
        logger.info(f"Starting training from step {start_step} to {args.max_steps}")

    tb_writer = None
    if rank == 0:
        tb_log_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard writer initialized at {tb_log_dir}")

    grad_accum  = args.gradient_accumulation_steps
    data_iters = {k: iter(loader) for k, loader in mod_loaders.items()}
    optimizer.zero_grad()

    _SKIP_LOSS_KEYS = {"loss"}
    log_loss_names = [
        "grad_norm", "loss",
        "t2i_text_loss", "mmu_text_loss", "lm_text_loss",
        "t2i_image_loss", "mmu_image_loss", "lm_image_loss",
        "moe_loss", "capacity_rate",
        "moe_loss_mot_gen", "capacity_rate_mot_gen",
    ]
    log_data_keys = ["t2i", "mmu", "lm"]
    running_loss_dict    = {}
    running_step_dict    = {}
    running_samples_dict = {}
    running_tokens_dict  = {}
    consumed_samples_total = {}
    consumed_tokens_total  = {}
    interval_t0 = time.time()

    for opt_step in range(start_step, args.max_steps):
        if args.shield_step > 0:
            model_config._current_training_iter = opt_step

        for micro_idx in range(grad_accum):
            micro_step = opt_step * grad_accum + micro_idx

            scheduled_mod = assigned_mod if assigned_mod is not None else modality_mix_schedule[micro_step % len(modality_mix_schedule)]
            try:
                batch = next(data_iters[scheduled_mod])
            except StopIteration:
                data_iters[scheduled_mod] = iter(mod_loaders[scheduled_mod])
                batch = next(data_iters[scheduled_mod])

            tag = batch.get("dataset_tag", [None])[0]
            if tag:
                modality_batch_counts[tag] = modality_batch_counts.get(tag, 0) + 1
            for mod, cnt in batch.get("modality_samples", {}).items():
                running_samples_dict[mod] = running_samples_dict.get(mod, 0) + cnt
            for mod, tks in batch.get("modality_tokens", {}).items():
                running_tokens_dict[mod] = running_tokens_dict.get(mod, 0) + tks

            kwargs, loss_inputs = prepare_batch(batch, vae, flow, device, args.image_loss_weight, vae_scaling_factor)

            clear_moe_training_metrics(model)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(**kwargs)

            losses = compute_training_losses(model, output, **loss_inputs)
            loss   = losses["loss"] / grad_accum

            _loss_val = loss.item() * grad_accum
            _extra_vals = {}
            for k, v in losses.items():
                if k not in _SKIP_LOSS_KEYS:
                    _extra_vals[k] = v.item() if hasattr(v, "item") else float(v)

            is_last_microbatch = micro_idx == grad_accum - 1
            model.set_requires_gradient_sync(is_last_microbatch)
            loss.backward()

            running_loss_dict["loss"] = running_loss_dict.get("loss", 0.0) + _loss_val
            running_step_dict["loss"] = running_step_dict.get("loss", 0) + 1
            for k, v_item in _extra_vals.items():
                running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v_item
                running_step_dict[k] = running_step_dict.get(k, 0) + 1

        gn = _clip_grad_local_shards(model, args.grad_clip)
        running_loss_dict["grad_norm"] = running_loss_dict.get("grad_norm", 0.0) + gn
        running_step_dict["grad_norm"] = running_step_dict.get("grad_norm", 0) + 1

        if maop is not None:
            if not (args.shield_step > 0 and opt_step < args.shield_step):
                maop.apply()

        _local_adamw_step(optimizer)
        optimizer.zero_grad()
        scheduler.step()

        _opt_step = opt_step + 1
        if _opt_step % args.log_interval == 0:
            elapsed = time.time() - interval_t0
            lr_list = scheduler.get_last_lr()

            loss_reduced = _all_reduce_metrics(
                running_loss_dict, running_step_dict,
                running_samples_dict, running_tokens_dict,
                consumed_samples_total, consumed_tokens_total,
                dp_group=None,
                loss_names=log_loss_names,
                data_keys=log_data_keys,
            )

            if rank == 0:
                _training_log(logger, _opt_step, args.max_steps,
                              loss_reduced, elapsed / args.log_interval,
                              lr_list, consumed_samples_total, consumed_tokens_total,
                              tb_writer=tb_writer)

            running_loss_dict    = {}
            running_step_dict    = {}
            running_samples_dict = {}
            running_tokens_dict  = {}
            interval_t0 = time.time()

        if _opt_step % args.save_interval == 0:
            save_modality_batches = _aggregate_modality_batch_counts(modality_batch_counts)
            FSDPCheckpoint.save(
                ckpt_dir, _opt_step, model, optimizer, scheduler,
                args.sharding_strategy, args.num_shard,
                modality_batches=save_modality_batches,
                logger=logger if rank == 0 else None,
                save_optimizer=not args.no_save_optimizer,
            )

    if args.max_steps % args.save_interval != 0:
        save_modality_batches = _aggregate_modality_batch_counts(modality_batch_counts)
        FSDPCheckpoint.save(
            ckpt_dir, args.max_steps, model, optimizer, scheduler,
            args.sharding_strategy, args.num_shard,
            modality_batches=save_modality_batches,
            logger=logger if rank == 0 else None,
            save_optimizer=not args.no_save_optimizer,
        )
    if rank == 0:
        if tb_writer is not None:
            tb_writer.close()
        logger.info("Training complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
