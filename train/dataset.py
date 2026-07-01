"""Iterable datasets and sequence packing for Rosetta FSDP training."""
import glob
import io
import itertools
import json
import os
import random
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import IterableDataset

# T2I resolution buckets: 33 aspect-ratio buckets centered on 256x256 px.
def _build_arc33_buckets(base_size: int = 256, align: int = 16):
    """Generate arc33-style resolution buckets (pixel h, pixel w).

    With base_size=256 and align=16, buckets range from (128, 512)
    to (512, 128) while keeping roughly base_size^2 pixels.
    """
    step = base_size // align       # 16 for base_size=256, align=16
    min_size = base_size // 2       # 128
    max_size = base_size * 2        # 512

    buckets = [(base_size, base_size)]

    cur_h, cur_w = base_size, base_size
    while True:
        if cur_h >= max_size and cur_w <= min_size:
            break
        cur_h = min(cur_h + step, max_size)
        cur_w = max(cur_w - step, min_size)
        buckets.append((cur_h // align * align, cur_w // align * align))

    cur_h, cur_w = base_size, base_size
    while True:
        if cur_h <= min_size and cur_w >= max_size:
            break
        cur_h = max(cur_h - step, min_size)
        cur_w = min(cur_w + step, max_size)
        buckets.append((cur_h // align * align, cur_w // align * align))

    return sorted(set(buckets), key=lambda x: x[0] / x[1])   # sorted by aspect ratio h/w


_T2I_BUCKETS_PX    = _build_arc33_buckets()   # list of (h, w) pixel sizes
_VIT_SPATIAL_MERGE = 2                        # Qwen3-VL: 2x2 patch merge -> 1 output token
_VIT_PATCH_SIZE = 16
_VIT_TEMPORAL_PATCH_SIZE = 2


def _snap_t2i_resolution(h: int, w: int) -> tuple:
    """Select closest (pixel_h, pixel_w) bucket for a given T2I image.
    Returns pixel dimensions (not token dimensions); caller divides by vae_ds for tokens.
    """
    ratio = h / w
    return min(_T2I_BUCKETS_PX, key=lambda b: abs(b[0] / b[1] - ratio))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factor_token_shape(n_tokens: int, preferred_h: int, preferred_w: int) -> tuple[int, int]:
    """Find h*w=n_tokens close to the preferred aspect ratio."""
    if n_tokens <= 0:
        return 1, 1
    target_ratio = max(1, preferred_h) / max(1, preferred_w)
    best_h, best_score = 1, float("inf")
    for h in range(1, int(n_tokens ** 0.5) + 2):
        if n_tokens % h != 0:
            continue
        w = n_tokens // h
        score = abs((h / w) - target_ratio)
        if score < best_score:
            best_h, best_score = h, score
    return best_h, n_tokens // best_h


def _worker_rank() -> int:
    """Return a unique integer per (distributed rank, DataLoader worker).

    Used to offset shuffle seeds so each replica + worker sees a different
    data ordering.

    Reads rank from RANK env var (set by torchrun) instead of torch.distributed,
    because DataLoader workers are forked after NCCL init and calling
    torch.distributed in a forked process has undefined behavior.
    """
    rank = int(os.environ.get("RANK", "0"))
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    return rank * 100 + worker_id


# ---------------------------------------------------------------------------
# Raw IO datasets - yield raw Python dicts, no tokenization
# ---------------------------------------------------------------------------

class LMDataset(IterableDataset):
    """LLaVA-Instruct conversation_58k.json - pure text, no images."""

    # shuffle_seed
    def __init__(self, json_path: str, shuffle_seed: int = 3333):
        with open(json_path) as f:
            self._data = json.load(f)
        self._shuffle_seed = shuffle_seed

    def __iter__(self):
        rng = random.Random(self._shuffle_seed + _worker_rank())
        data = list(self._data)
        rng.shuffle(data)
        for item in itertools.cycle(data):
            if item.get("conversations"):
                yield {"task_type": "lm", "conversations": item["conversations"]}


class MMUDataset(IterableDataset):
    """LLaVA-OV JSONL - image + multi-turn conversation."""

    # shuffle_seed
    def __init__(self, jsonl_path: str, image_dir: str, shuffle_seed: int = 5555):
        with open(jsonl_path) as f:
            self._lines = f.readlines()
        self._image_dir = image_dir
        self._shuffle_seed = shuffle_seed

    def __iter__(self):
        rng = random.Random(self._shuffle_seed + _worker_rank())
        lines = list(self._lines)
        rng.shuffle(lines)
        consecutive_miss = 0
        for line in itertools.cycle(lines):
            item = json.loads(line)
            if not item.get("conversations"):
                continue
            img_path = os.path.join(self._image_dir, item["image"])
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                consecutive_miss += 1
                if consecutive_miss >= len(lines):
                    raise RuntimeError(
                        f"MMUDataset: no valid images found in {self._image_dir!r}. "
                        "Check --mmu-image-dir points to the directory containing the images."
                    )
                continue
            consecutive_miss = 0
            yield {"task_type": "mmu", "image": image, "conversations": item["conversations"]}


class T2IDataset(IterableDataset):
    """Bagel-style parquet - image bytes column + captions JSON column."""

    # shuffle_seed
    def __init__(self, parquet_dir: str, shuffle_seed: int = 1111):
        self._files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
        if not self._files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
        self._shuffle_seed = shuffle_seed

    def __iter__(self):
        rng = random.Random(self._shuffle_seed + _worker_rank())
        for path in itertools.cycle(self._files):
            df = pd.read_parquet(path)
            rows = list(df.itertuples(index=False))
            rng.shuffle(rows)
            for row in rows:
                try:
                    img_bytes = row.image["bytes"] if isinstance(row.image, dict) else row.image
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    caps = json.loads(row.captions) if isinstance(row.captions, str) else row.captions
                    caption = caps if isinstance(caps, str) else rng.choice(list(caps.values()))
                except Exception:
                    continue
                yield {"task_type": "t2i", "image": image, "caption": caption}


# ---------------------------------------------------------------------------
# Preprocessor - raw dict -> tokenized per-sample dict
# ---------------------------------------------------------------------------

class Preprocessor:
    """Converts raw IO dicts to tokenized per-sample dicts (CPU-only).

    Tokenization uses BaseMultimodalTokenizerFast.encode_general.
    MMU ViT image preprocessing uses AutoImageProcessor (pixel normalization only;
    the actual ViT forward pass happens inside model.forward()).
    T2I images remain as PIL; VAE encoding (GPU) happens in the training loop.
    """

    def __init__(self, tokenizer, vit_processor,
                 vae_downsample: int = 16, max_seq_len: int = 2048):
        self._tok = tokenizer
        self._vit_proc = vit_processor          # transformers AutoImageProcessor (dynamic resolution)
        self._vae_ds = vae_downsample
        self._max_seq_len = max_seq_len

    def __call__(self, raw: dict) -> Optional[dict]:
        try:
            task_type = raw["task_type"]
            if task_type == "lm":
                return self._lm(raw)
            elif task_type == "mmu":
                return self._mmu(raw)
            elif task_type == "t2i":
                return self._t2i(raw)
        except Exception:
            return None

    def _encode(self, sections: list) -> object:
        return self._tok.encode_general(
            sections=sections,
            add_bos=False,
            add_eos=True,
            add_pad=False,
            use_text_mask=True,
        )

    @staticmethod
    def _make_target(tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        target = tokens.clone()
        target[~text_mask.bool()] = -100
        return target

    def _lm(self, raw: dict) -> Optional[dict]:
        sections = []
        for conv in raw["conversations"]:
            role = "user" if conv["from"] == "human" else "assistant"
            prefix = f"<|im_start|>{role}\n"
            suffix = "<|im_end|>\n" if role == "user" else "<|im_end|>"
            if role == "user":
                sections.append({
                    "type": "text",
                    "text": f"{prefix}{conv['value']}{suffix}",
                    "ignore": True,
                })
            else:
                sections.extend([
                    {"type": "text", "text": prefix, "ignore": True},
                    {"type": "text", "text": f"{conv['value']}{suffix}", "ignore": False},
                ])
        out = self._encode(sections)
        tokens = out.tokens
        text_mask = out.text_mask if out.text_mask is not None else torch.zeros_like(tokens, dtype=torch.float)
        return {
            "tokens": tokens,
            "target_tokens": self._make_target(tokens, text_mask),
            "text_mask": text_mask,
            "data_type": "lm",
            "n_samples": 1,
        }

    def _mmu(self, raw: dict) -> Optional[dict]:
        # Dynamic resolution: let AutoImageProcessor resize within its min/max_pixels.
        proc = self._vit_proc(raw["image"], return_tensors="pt")
        pixel_values = proc["pixel_values"].squeeze(0)   # (C, H_resized, W_resized)
        grid_thw = proc["image_grid_thw"]                # (1, 3) - (t, h_grid, w_grid)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        t_dim, h_grid, w_grid = grid_thw[0]
        token_h = int(h_grid.item()) // _VIT_SPATIAL_MERGE
        token_w = int(w_grid.item()) // _VIT_SPATIAL_MERGE
        n_tokens = token_h * token_w
        # Match the placeholder mask to the actual Qwen3-VL ViT output length.
        # PatchEmbed first reshapes pixel_values into 3 x temporal_patch x
        # patch_size x patch_size chunks, then PatchMerger merges 2x2 spatial
        # chunks into one LLM token.
        patch_embed_unit = 3 * _VIT_TEMPORAL_PATCH_SIZE * _VIT_PATCH_SIZE * _VIT_PATCH_SIZE
        actual_tokens = pixel_values.numel() // (patch_embed_unit * (_VIT_SPATIAL_MERGE ** 2))
        if actual_tokens > 0 and actual_tokens != n_tokens:
            token_h, token_w = _factor_token_shape(int(actual_tokens), token_h, token_w)
            n_tokens = token_h * token_w
            grid_thw = grid_thw.clone()
            grid_thw[0, 0] = 1
            grid_thw[0, 1] = token_h * _VIT_SPATIAL_MERGE
            grid_thw[0, 2] = token_w * _VIT_SPATIAL_MERGE

        sections = [
            {
                "type": "cond_vit_image",
                "token_length": n_tokens,
                "token_height": token_h,
                "token_width": token_w,
            }
        ]
        for conv in raw["conversations"]:
            role = "user" if conv["from"] == "human" else "assistant"
            prefix = f"<|im_start|>{role}\n"
            suffix = "<|im_end|>\n" if role == "user" else "<|im_end|>"
            if role == "user":
                sections.append({
                    "type": "text",
                    "text": f"{prefix}{conv['value']}{suffix}",
                    "ignore": True,
                })
            else:
                sections.extend([
                    {"type": "text", "text": prefix, "ignore": True},
                    {"type": "text", "text": f"{conv['value']}{suffix}", "ignore": False},
                ])

        out = self._encode(sections)
        tokens = out.tokens
        vit_mask = (out.vit_image_mask if out.vit_image_mask is not None
                    else torch.zeros_like(tokens, dtype=torch.bool))
        text_mask = (out.text_mask if out.text_mask is not None
                     else torch.zeros_like(tokens, dtype=torch.float))
        vit_slices = out.vit_image_slices or []
        rope_media = [
            (sli, (token_h, token_w), {"image_type": "cond_vit_image"})
            for sli in vit_slices
        ]

        return {
            "tokens": tokens,
            "target_tokens": self._make_target(tokens, text_mask),
            "text_mask": text_mask,
            "cond_vit_image_mask": vit_mask,
            "cond_vit_images": pixel_values.unsqueeze(0),   # (1, C, H, W)
            "cond_vit_image_kwargs": {"grid_thw": grid_thw},# (1, 3)
            "cond_vit_image_slices": vit_slices,
            "cond_full_attn_slices": list(vit_slices),      # same slices -> full attn for ViT tokens
            "rope_media_info": rope_media,
            "data_type": "mmu",
            "n_samples": 1,
        }

    def _t2i(self, raw: dict) -> Optional[dict]:
        w, h = raw["image"].size
        pix_h, pix_w = _snap_t2i_resolution(h, w)
        token_h, token_w = pix_h // self._vae_ds, pix_w // self._vae_ds
        n_tokens = token_h * token_w

        sections = [
            {"type": "text", "text": f"<|im_start|>user\n{raw['caption']}<|im_end|>\n", "ignore": False},
            {"type": "text", "text": "<|im_start|>assistant\n", "ignore": True},
            {
                "type": "gen_image",
                "token_length": n_tokens,
                "token_height": token_h,
                "token_width": token_w,
            },
            {"type": "text", "text": "<|im_end|>", "ignore": True},
        ]

        out = self._encode(sections)
        tokens = out.tokens
        img_mask = (out.gen_image_mask if out.gen_image_mask is not None
                    else torch.zeros_like(tokens, dtype=torch.bool))
        text_mask = (out.text_mask if out.text_mask is not None
                     else torch.zeros_like(tokens, dtype=torch.float))
        gen_slices = out.gen_image_slices or []
        rope_img = [
            (sli, (token_h, token_w), {"image_type": "gen_image"})
            for sli in gen_slices
        ]

        image = raw["image"].resize((pix_w, pix_h), Image.LANCZOS)

        return {
            "tokens": tokens,
            "target_tokens": self._make_target(tokens, text_mask),
            "text_mask": text_mask,
            "image_mask": img_mask,
            "images": [image],
            "image_slices": gen_slices,
            "rope_image_info": rope_img,
            "data_type": "t2i",
            "n_samples": 1,
        }


# ---------------------------------------------------------------------------
# Bin-packing helpers
# ---------------------------------------------------------------------------

def _first_fit_pack(buf: list, max_len: int) -> tuple:
    """Take the longest prefix of buf whose total token length <= max_len."""
    total = 0
    for i, sample in enumerate(buf):
        n = sample["tokens"].shape[0]
        if total + n > max_len:
            # Must take at least one sample (caller guarantees n <= max_len)
            return buf[:max(i, 1)], buf[max(i, 1):]
        total += n
    return buf, []


def _seq_collate_fn(items: list, max_len: int, pad_token_id: int = 0) -> dict:
    """Pack tokenized samples into one sequence dict (bsz=1).

    Offsets store packed sample boundaries. Position-aware fields such as
    image slices and RoPE metadata are shifted by each sample offset.
    """
    lengths = [item["tokens"].shape[0] for item in items]
    offsets = [0] + list(itertools.accumulate(lengths))
    assert offsets[-1] <= max_len, (
        f"Packed length {offsets[-1]} exceeds max_seq_len {max_len}. "
        f"Sample lengths: {lengths}"
    )
    pad_len = max_len - offsets[-1]

    def _cat_pad(key: str, pad_val, dtype) -> Optional[torch.Tensor]:
        seqs = [item[key] for item in items if key in item]
        if not seqs:
            return None
        pad = torch.full((pad_len,), pad_val, dtype=dtype)
        return torch.cat(seqs + [pad]).unsqueeze(0)     # (1, max_len)

    data_types = [item["data_type"] for item in items]
    result = {
        # Packed sequence boundaries: (1, K+1)
        # Passed as sample_offsets to model.forward() for flex attention + MRoPE rebasing
        "offsets": torch.tensor(offsets, dtype=torch.long).unsqueeze(0),
        "tokens": _cat_pad("tokens", pad_token_id, torch.long),
        "target_tokens": _cat_pad("target_tokens", -100, torch.long),
        "text_mask": _cat_pad("text_mask", 0.0, torch.float),
        "n_samples": torch.tensor([len(items)]),
        "data_types": data_types,
        # Per-modality consumed metrics used by training logs and resume state.
        "dataset_tag": [items[0]["data_type"]],
        "modality_samples": {dt: sum(1 for t in data_types if t == dt) for dt in set(data_types)},
        "modality_tokens":  {dt: sum(lengths[i] for i, t in enumerate(data_types) if t == dt)
                             for dt in set(data_types)},
    }

    if any("image_mask" in item for item in items):
        result["image_mask"] = _cat_pad("image_mask", False, torch.bool)
    if any("cond_vit_image_mask" in item for item in items):
        result["cond_vit_image_mask"] = _cat_pad("cond_vit_image_mask", False, torch.bool)

    # Collect and offset-shift position-aware data
    img_slices, rope_img, all_images = [], [], []
    vit_slices, full_attn_slices, rope_media = [], [], []
    all_vit_imgs, all_grid_thw = [], []

    for item, off in zip(items, offsets):
        for sli in item.get("image_slices", []):
            img_slices.append(slice(sli.start + off, sli.stop + off))
        for sli, shape, meta in item.get("rope_image_info", []):
            rope_img.append((slice(sli.start + off, sli.stop + off), shape, meta))
        all_images.extend(item.get("images", []))

        for sli in item.get("cond_vit_image_slices", []):
            vit_slices.append(slice(sli.start + off, sli.stop + off))
        for sli in item.get("cond_full_attn_slices", []):
            full_attn_slices.append(slice(sli.start + off, sli.stop + off))
        for sli, shape, meta in item.get("rope_media_info", []):
            rope_media.append((slice(sli.start + off, sli.stop + off), shape, meta))

        cvi = item.get("cond_vit_images")
        if cvi is not None:
            cvi_4d = cvi.unsqueeze(0) if cvi.ndim == 3 else cvi  # (N, C, H, W)
            for img in cvi_4d:
                all_vit_imgs.append(img)
        kwargs = item.get("cond_vit_image_kwargs")
        if kwargs is not None and "grid_thw" in kwargs:
            g = kwargs["grid_thw"]
            all_grid_thw.append(g if g.ndim == 2 else g.unsqueeze(0))  # ensure (n, 3)

    if img_slices:
        result["image_slices"] = [img_slices]
    if rope_img:
        result["rope_image_info"] = [rope_img]
    if all_images:
        result["images"] = [all_images]

    if vit_slices:
        result["cond_vit_image_slices"] = [vit_slices]
    if full_attn_slices:
        result["cond_full_attn_slices"] = [full_attn_slices]
    if rope_media:
        result["rope_media_info"] = [rope_media]

    if all_vit_imgs:
        shapes = [img.shape for img in all_vit_imgs]
        if len(all_vit_imgs) == 1 and len(set(shapes)) == 1:
            # Single image: tensor path is safe and efficient.
            stacked = torch.stack(all_vit_imgs)         # (N, C, H, W)
            result["cond_vit_images"] = stacked.unsqueeze(0)   # (1, N, C, H, W)
            if all_grid_thw:
                result["cond_vit_image_kwargs"] = {
                    "grid_thw": torch.cat(all_grid_thw).unsqueeze(0)  # (1, N, 3)
                }
        else:
            # Multiple images: keep as list-of-lists. The tensor path treats N as
            # a separate image axis, while Qwen ViT returns one concatenated token
            # sequence, causing an extra factor of N in scatter shape.
            result["cond_vit_images"] = [all_vit_imgs]
            if all_grid_thw:
                result["cond_vit_image_kwargs"] = {"grid_thw": [all_grid_thw]}

    return result


# ---------------------------------------------------------------------------
# PackedDataset - accumulates and bin-packs preprocessed samples
# ---------------------------------------------------------------------------

class PackedDataset(IterableDataset):
    """Wraps a source IterableDataset, preprocesses samples, and bin-packs them.

    skip: batches to discard at __iter__ start for data resume. Works because
    sources use itertools.cycle with fixed shuffle seeds -> same order every run.
    """

    def __init__(self, source: IterableDataset, preprocessor: Callable,
                 max_seq_len: int, skip: int = 0):
        self._source = source
        self._prep = preprocessor
        self._max_seq_len = max_seq_len
        self._skip = skip
        tok = getattr(preprocessor, "_tok", None)
        self._pad_id = (getattr(tok, "pad_token_id", None) or 0) if tok else 0

    def __iter__(self):
        skipped = 0
        buf = []
        for raw in self._source:
            sample = self._prep(raw)
            if sample is None:
                continue
            n = sample["tokens"].shape[0]
            if n > self._max_seq_len:
                continue
            buf.append(sample)

            total = sum(s["tokens"].shape[0] for s in buf)
            if total >= self._max_seq_len:
                pack, buf = _first_fit_pack(buf, self._max_seq_len)
                if skipped < self._skip:
                    skipped += 1
                    continue
                yield _seq_collate_fn(pack, self._max_seq_len, self._pad_id)

        # Flush remaining samples
        while buf:
            pack, buf = _first_fit_pack(buf, self._max_seq_len)
            if skipped < self._skip:
                skipped += 1
                continue
            yield _seq_collate_fn(pack, self._max_seq_len, self._pad_id)
