"""Momentum-Anchored Orthogonal Projection (MAOP) for shared experts.

After backward(), before optimizer.step():
  g = param.grad          -- mixed gradient (text + ViT + VAE)
  m = exp_avg             -- Adam momentum anchor
  if dot(g, m) < 0:       -- gradient conflicts with momentum direction
      g_orth = g - (g·m / |m|²) * m
      param.grad ← g_orth

In FSDP2, param.grad and exp_avg can be DTensor shards, so dot products are
reduced within the shard group before applying the projection.
"""
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_fsdp_pg(param):
    """Extract the FSDP2 shard-group process group from a DTensor parameter."""
    src = None
    if hasattr(param, 'device_mesh') and param.device_mesh is not None:
        src = param.device_mesh
    elif param.grad is not None and hasattr(param.grad, 'device_mesh'):
        src = param.grad.device_mesh
    if src is None:
        return None
    return src.get_group(src.ndim - 1)


class SharedExpertOrthManager:
    """Applies MAOP to all shared_mlp parameters after each backward pass.

    Usage (after clip_grad_norm_, before optimizer.step()):
        maop.apply()
    """

    def __init__(self, model, optimizer):
        self._optimizer = optimizer
        self._shared_params = [
            p for name, p in model.named_parameters()
            if 'shared_mlp' in name and p.requires_grad
        ]
        if not self._shared_params:
            logger.warning(
                "[MAOP] No shared_mlp parameters found; MAOP will have no effect."
            )

    @torch.no_grad()
    def apply(self):

        id_to_exp_avg = {
            id(p): state['exp_avg']
            for group in self._optimizer.param_groups
            for p in group['params']
            if 'exp_avg' in (state := self._optimizer.state.get(p, {}))
        }

        per_param = []
        fsdp_pg = None

        for param in self._shared_params:
            if param.grad is None:
                continue
            exp_avg = id_to_exp_avg.get(id(param))
            if exp_avg is None:
                continue

            grad_local = param.grad.to_local() if hasattr(param.grad, 'to_local') else param.grad
            ea_local = exp_avg.to_local() if hasattr(exp_avg, 'to_local') else exp_avg

            g = grad_local.float().clone().reshape(-1)
            m = ea_local.float().clone().reshape(-1)
            per_param.append((param, grad_local, g, m))

            if fsdp_pg is None:
                fsdp_pg = _get_fsdp_pg(param)

        if not per_param:
            return

        # Batch local dot products: [n_params, 2], with columns dot(g, m) and dot(m, m).
        dots = torch.stack([
            torch.stack([torch.dot(g, m), torch.dot(m, m)])
            for _, _, g, m in per_param
        ])

        # All-reduce within shard group to get global dots
        if fsdp_pg is not None and dist.is_initialized():
            dist.all_reduce(dots, op=dist.ReduceOp.SUM, group=fsdp_pg)

        for i, (param, grad_local, g, m) in enumerate(per_param):
            gm_dot = dots[i, 0].item()
            m_norm_sq = dots[i, 1].item()
            if gm_dot >= 0.0 or m_norm_sq < 1e-12:
                continue
            # g_orth = g - (g·m / |m|²) * m
            g.sub_(gm_dot / m_norm_sq * m)
            grad_local.copy_(g.reshape(grad_local.shape).to(grad_local.dtype))
