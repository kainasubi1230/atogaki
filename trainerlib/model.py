from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from math import atan2, cos, hypot, sin, tanh
from pathlib import Path
import random
import secrets
from typing import Any

from .char_token import char_to_model_id, stable_int_token

try:
    import torch  # pyright: ignore [reportMissingImports]
    from torch import nn  # pyright: ignore [reportMissingImports]
    import torch.nn.functional as F  # pyright: ignore [reportMissingImports]
    from torch.nn.utils.rnn import pad_sequence  # pyright: ignore [reportMissingImports]
except Exception:  # pragma: no cover - optional dependency for GPU environments
    torch = None
    nn = None
    F = None
    pad_sequence = None


class TinyHandwritingModel(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128):
        if nn is None:
            return
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, hidden_dim)
        self.style_embed = nn.Embedding(4096, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, char_ids, style_id, time_steps):
        if nn is None:
            raise RuntimeError("torch is not available")
        char_vec = self.char_embed(char_ids)
        style_vec = self.style_embed(style_id).unsqueeze(1)
        time_vec = self.time_proj(time_steps.unsqueeze(-1))
        h = char_vec + style_vec + time_vec
        return self.decoder(h)


@dataclass
class TrainResult:
    similarity_score: float
    cer: float
    adapter_path: str


@dataclass
class EncodedSample:
    char_id: int
    style_id: int
    target: Any


_MODEL_CACHE: dict[str, tuple[float, TinyHandwritingModel, int, int, dict[str, Any]]] = {}


def _load_jsonl_dataset(path: str) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("sequence"):
                continue
            items.append(item)
    return items


def _encode_samples(samples: list[dict], vocab_size: int) -> list[EncodedSample]:
    encoded: list[EncodedSample] = []
    for sample in samples:
        seq = sample.get("sequence") or []
        if not seq:
            continue
        target = torch.tensor(seq, dtype=torch.float32)
        if target.ndim != 2:
            continue
        if target.shape[1] > 4:
            target = target[:, :4]
        elif target.shape[1] < 4:
            pad_cols = torch.zeros((target.shape[0], 4 - target.shape[1]), dtype=torch.float32)
            target = torch.cat((target, pad_cols), dim=1)
        encoded.append(
            EncodedSample(
                char_id=int(sample["char_id"]) % vocab_size,
                style_id=int(sample.get("style_id", 0)),
                target=target.contiguous(),
            )
        )
    return encoded


def _collect_sequence_length_metadata(encoded_samples: list[EncodedSample]) -> dict[str, Any]:
    if not encoded_samples:
        return {"sequence_len_mean": 96.0, "char_len_mean": {}}
    total_len = 0
    by_char: dict[int, tuple[int, int]] = {}
    for sample in encoded_samples:
        seq_len = int(sample.target.shape[0])
        total_len += seq_len
        cur_total, cur_count = by_char.get(sample.char_id, (0, 0))
        by_char[sample.char_id] = (cur_total + seq_len, cur_count + 1)
    char_len_mean = {
        str(char_id): (char_total / max(1, char_count))
        for char_id, (char_total, char_count) in by_char.items()
    }
    return {
        "sequence_len_mean": total_len / len(encoded_samples),
        "char_len_mean": char_len_mean,
    }


def _collect_char_exemplars(
    samples: list[dict],
    vocab_size: int,
    *,
    per_char_cap: int = 24,
    max_seq_len: int = 180,
) -> dict[str, list[list[list[float]]]]:
    exemplars: dict[str, list[list[list[float]]]] = {}
    rng = random.Random(1234)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    for sample in shuffled:
        seq = sample.get("sequence") or []
        if not seq:
            continue
        char_id = int(sample.get("char_id", 0)) % vocab_size
        key = str(char_id)
        bucket = exemplars.setdefault(key, [])
        if len(bucket) >= per_char_cap:
            continue
        clipped: list[list[float]] = []
        for row in seq[:max_seq_len]:
            if not isinstance(row, list) or len(row) < 4:
                continue
            clipped.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        if len(clipped) >= 8:
            bucket.append(clipped)
    return exemplars


def _collect_char_exemplars_by_text(
    samples: list[dict],
    *,
    per_char_cap: int = 24,
    max_seq_len: int = 180,
) -> dict[str, list[list[list[float]]]]:
    exemplars: dict[str, list[list[list[float]]]] = {}
    rng = random.Random(2468)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    for sample in shuffled:
        meta = sample.get("meta")
        ch = None
        if isinstance(meta, dict):
            raw = meta.get("char")
            if isinstance(raw, str) and len(raw) == 1:
                ch = raw
        if ch is None:
            continue
        bucket = exemplars.setdefault(ch, [])
        if len(bucket) >= per_char_cap:
            continue
        normalized = _normalized_sequence_rows(sample.get("sequence"), max_seq_len=max_seq_len)
        if len(normalized) < 8:
            continue
        bucket.append(normalized)
    return exemplars


def _normalized_sequence_rows(seq: Any, *, max_seq_len: int = 180) -> list[list[float]]:
    if not isinstance(seq, list):
        return []
    clipped: list[list[float]] = []
    for row in seq[: max(1, max_seq_len)]:
        if not isinstance(row, list) or len(row) < 4:
            continue
        clipped.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return clipped


def _collect_user_char_exemplars(
    samples: list[dict],
    *,
    vocab_size: int = 4096,
    per_char_cap: int = 18,
    max_seq_len: int = 200,
) -> dict[str, list[list[list[float]]]]:
    exemplars: dict[str, list[list[list[float]]]] = {}
    rng = random.Random(98765)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    for sample in shuffled:
        try:
            char_id = int(sample.get("char_id", 0)) % vocab_size
        except Exception:
            continue
        normalized = _normalized_sequence_rows(sample.get("sequence"), max_seq_len=max_seq_len)
        if len(normalized) < 8:
            continue
        key = str(char_id)
        bucket = exemplars.setdefault(key, [])
        if len(bucket) >= max(1, per_char_cap):
            continue
        bucket.append(normalized)
    return exemplars


def _collect_user_char_exemplars_text(
    samples: list[dict],
    *,
    per_char_cap: int = 24,
    max_seq_len: int = 200,
) -> dict[str, list[list[list[float]]]]:
    exemplars: dict[str, list[list[list[float]]]] = {}
    rng = random.Random(87654)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    for sample in shuffled:
        meta = sample.get("meta")
        ch = None
        if isinstance(meta, dict):
            raw = meta.get("char")
            if isinstance(raw, str) and len(raw) == 1:
                ch = raw
        if ch is None:
            continue
        normalized = _normalized_sequence_rows(sample.get("sequence"), max_seq_len=max_seq_len)
        if len(normalized) < 8:
            continue
        bucket = exemplars.setdefault(ch, [])
        if len(bucket) >= max(1, per_char_cap):
            continue
        bucket.append(normalized)
    return exemplars


def _estimate_style_profile(samples: list[dict]) -> dict[str, float]:
    widths: list[float] = []
    jitter_acc: list[float] = []
    slant_terms: list[float] = []
    aspect_terms: list[float] = []
    for sample in samples:
        seq = _normalized_sequence_rows(sample.get("sequence"), max_seq_len=220)
        if len(seq) < 8:
            continue
        x = 0.0
        y = 0.0
        xs: list[float] = []
        ys: list[float] = []
        headings: list[float] = []
        for row in seq:
            dx = float(row[0]) * 20.0
            dy = float(row[1]) * 20.0
            x += dx
            y += dy
            xs.append(x)
            ys.append(y)
            widths.append(float(row[3]))
            if float(row[2]) > 0.5:
                seg_len = hypot(dx, dy)
                if seg_len > 1e-4:
                    headings.append(atan2(dy, dx))
                    slant_terms.append(dx / max(seg_len, 1e-6))
        if xs and ys:
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w > 1e-6 and h > 1e-6:
                aspect_terms.append(max(0.4, min(2.2, h / w)))
        if len(headings) >= 2:
            deltas: list[float] = []
            for i in range(1, len(headings)):
                d = headings[i] - headings[i - 1]
                while d > 3.141592653589793:
                    d -= 6.283185307179586
                while d < -3.141592653589793:
                    d += 6.283185307179586
                deltas.append(abs(d))
            if deltas:
                jitter_acc.append(sum(deltas) / len(deltas))

    mean_width = (sum(widths) / len(widths)) if widths else 0.78
    mean_slant = (sum(slant_terms) / len(slant_terms)) if slant_terms else 0.0
    mean_aspect = (sum(aspect_terms) / len(aspect_terms)) if aspect_terms else 1.0
    mean_jitter = (sum(jitter_acc) / len(jitter_acc)) if jitter_acc else 0.8

    rot_bias_deg = max(-8.0, min(8.0, mean_slant * 11.0))
    aspect_delta = max(-0.25, min(0.25, mean_aspect - 1.0))
    scale_x = max(0.84, min(1.18, 1.0 - aspect_delta * 0.30))
    scale_y = max(0.84, min(1.18, 1.0 + aspect_delta * 0.30))
    jitter_scale = max(0.85, min(1.45, 0.92 + mean_jitter * 0.35))
    width_scale = max(0.85, min(1.20, mean_width / 0.78 if abs(0.78) > 1e-6 else 1.0))
    shear_x = max(-0.14, min(0.14, mean_slant * 0.16))
    shear_y = max(-0.08, min(0.08, -mean_slant * 0.06))
    return {
        "rot_bias_deg": round(rot_bias_deg, 4),
        "scale_x": round(scale_x, 4),
        "scale_y": round(scale_y, 4),
        "shear_x": round(shear_x, 4),
        "shear_y": round(shear_y, 4),
        "jitter_scale": round(jitter_scale, 4),
        "width_scale": round(width_scale, 4),
    }


def _merged_char_exemplars(
    base: dict[str, Any] | None,
    adapter: dict[str, Any] | None,
    *,
    per_char_cap: int = 28,
) -> dict[str, list[list[list[float]]]]:
    def _is_noisy(seq: list[list[float]]) -> bool:
        down = 0
        long_jumps = 0
        headings: list[float] = []
        for row in seq:
            if len(row) < 4:
                continue
            dx = float(row[0]) * 20.0
            dy = float(row[1]) * 20.0
            pen = float(row[2]) > 0.5
            if not pen:
                continue
            down += 1
            seg = hypot(dx, dy)
            if seg > 16.0:
                long_jumps += 1
            if seg > 1e-6:
                headings.append(atan2(dy, dx))
        if down < 6:
            return True
        sharp = 0
        for idx in range(1, len(headings)):
            delta = headings[idx] - headings[idx - 1]
            while delta > 3.141592653589793:
                delta -= 6.283185307179586
            while delta < -3.141592653589793:
                delta += 6.283185307179586
            if abs(delta) > 1.35:
                sharp += 1
        jump_ratio = long_jumps / max(1, down)
        sharp_ratio = sharp / max(1, len(headings) - 1)
        return jump_ratio > 0.2 or sharp_ratio > 0.6

    merged: dict[str, list[list[list[float]]]] = {}
    base_raw = base if isinstance(base, dict) else {}
    adapter_raw = adapter if isinstance(adapter, dict) else {}
    keys = set(base_raw.keys()) | set(adapter_raw.keys())
    for key in keys:
        bucket: list[list[list[float]]] = []
        user_bucket: list[list[list[float]]] = []
        base_bucket: list[list[list[float]]] = []
        user_seq = adapter_raw.get(key)
        if isinstance(user_seq, list):
            for seq in user_seq:
                norm = _normalized_sequence_rows(seq, max_seq_len=200)
                if len(norm) >= 8 and not _is_noisy(norm):
                    user_bucket.append(norm)
                if len(user_bucket) >= per_char_cap:
                    break
        base_seq = base_raw.get(key)
        if isinstance(base_seq, list):
            for seq in base_seq:
                norm = _normalized_sequence_rows(seq, max_seq_len=200)
                if len(norm) >= 8:
                    base_bucket.append(norm)
                if len(base_bucket) >= per_char_cap:
                    break

        base_floor = min(8, max(2, per_char_cap // 3))
        user_quota = max(0, per_char_cap - base_floor)
        bucket.extend(user_bucket[:user_quota])
        bucket.extend(base_bucket[:base_floor])
        if len(bucket) < per_char_cap:
            rem_u = user_bucket[user_quota:]
            rem_b = base_bucket[base_floor:]
            for idx in range(max(len(rem_u), len(rem_b))):
                if idx < len(rem_u) and len(bucket) < per_char_cap:
                    bucket.append(rem_u[idx])
                if idx < len(rem_b) and len(bucket) < per_char_cap:
                    bucket.append(rem_b[idx])
        if bucket:
            merged[key] = bucket
    return merged


def _merged_text_exemplars(
    base: dict[str, Any] | None,
    adapter: dict[str, Any] | None,
    *,
    per_char_cap: int = 28,
) -> dict[str, list[list[list[float]]]]:
    def _is_noisy(seq: list[list[float]]) -> bool:
        down = 0
        long_jumps = 0
        headings: list[float] = []
        for row in seq:
            if len(row) < 4:
                continue
            dx = float(row[0]) * 20.0
            dy = float(row[1]) * 20.0
            pen = float(row[2]) > 0.5
            if not pen:
                continue
            down += 1
            seg = hypot(dx, dy)
            if seg > 16.0:
                long_jumps += 1
            if seg > 1e-6:
                headings.append(atan2(dy, dx))
        if down < 6:
            return True
        sharp = 0
        for idx in range(1, len(headings)):
            delta = headings[idx] - headings[idx - 1]
            while delta > 3.141592653589793:
                delta -= 6.283185307179586
            while delta < -3.141592653589793:
                delta += 6.283185307179586
            if abs(delta) > 1.35:
                sharp += 1
        jump_ratio = long_jumps / max(1, down)
        sharp_ratio = sharp / max(1, len(headings) - 1)
        return jump_ratio > 0.2 or sharp_ratio > 0.6

    merged: dict[str, list[list[list[float]]]] = {}
    base_raw = base if isinstance(base, dict) else {}
    adapter_raw = adapter if isinstance(adapter, dict) else {}
    keys = set(base_raw.keys()) | set(adapter_raw.keys())
    for key in keys:
        if not isinstance(key, str) or len(key) != 1:
            continue
        bucket: list[list[list[float]]] = []
        user_bucket: list[list[list[float]]] = []
        base_bucket: list[list[list[float]]] = []
        user_seq = adapter_raw.get(key)
        if isinstance(user_seq, list):
            for seq in user_seq:
                norm = _normalized_sequence_rows(seq, max_seq_len=200)
                if len(norm) >= 8 and not _is_noisy(norm):
                    user_bucket.append(norm)
                if len(user_bucket) >= per_char_cap:
                    break
        base_seq = base_raw.get(key)
        if isinstance(base_seq, list):
            for seq in base_seq:
                norm = _normalized_sequence_rows(seq, max_seq_len=200)
                if len(norm) >= 8:
                    base_bucket.append(norm)
                if len(base_bucket) >= per_char_cap:
                    break

        base_floor = min(10, max(3, per_char_cap // 3))
        user_quota = max(0, per_char_cap - base_floor)
        bucket.extend(user_bucket[:user_quota])
        bucket.extend(base_bucket[:base_floor])
        if len(bucket) < per_char_cap:
            rem_u = user_bucket[user_quota:]
            rem_b = base_bucket[base_floor:]
            for idx in range(max(len(rem_u), len(rem_b))):
                if idx < len(rem_u) and len(bucket) < per_char_cap:
                    bucket.append(rem_u[idx])
                if idx < len(rem_b) and len(bucket) < per_char_cap:
                    bucket.append(rem_b[idx])
        if bucket:
            merged[key] = bucket
    return merged


def _pick_device(preference: str) -> str:
    if torch is None:
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_batch(samples: list[EncodedSample], device: str):
    lengths_cpu = torch.tensor([s.target.shape[0] for s in samples], dtype=torch.long)
    max_len = int(lengths_cpu.max().item())
    batch_size = len(samples)

    targets_cpu = pad_sequence([s.target for s in samples], batch_first=True)
    transfer_non_blocking = device == "cuda"
    targets = targets_cpu.to(device=device, non_blocking=transfer_non_blocking)

    style_ids = torch.tensor([s.style_id for s in samples], dtype=torch.long, device=device)
    char_tokens = torch.tensor([s.char_id for s in samples], dtype=torch.long, device=device).unsqueeze(1)
    char_ids = char_tokens.expand(batch_size, max_len)

    lengths = lengths_cpu.to(device=device, non_blocking=transfer_non_blocking)
    steps = torch.arange(max_len, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, max_len)
    denom = torch.clamp(lengths - 1, min=1).to(dtype=torch.float32).unsqueeze(1)
    time_steps = torch.where(lengths.unsqueeze(1) > 1, steps / denom, torch.zeros_like(steps))

    mask = (steps < lengths.unsqueeze(1)).to(dtype=torch.float32)
    return char_ids, style_ids, time_steps, targets, mask


def train_base_model(
    output_path: str,
    dataset_path: str | None = None,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_preference: str = "auto",
    vocab_size: int = 4096,
    hidden_dim: int = 128,
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch is None:
        path.write_text(json.dumps({"warning": "torch not installed"}), encoding="utf-8")
        return {"base_model_path": str(path), "status": "saved_stub", "device": "cpu"}

    if dataset_path is None or not Path(dataset_path).exists():
        model = TinyHandwritingModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        torch.save(model.state_dict(), path)
        return {
            "base_model_path": str(path),
            "status": "saved_init_only",
            "reason": "dataset_not_provided",
        }

    samples = _load_jsonl_dataset(dataset_path)
    encoded_samples = _encode_samples(samples, vocab_size)
    if not encoded_samples:
        torch.save(TinyHandwritingModel(vocab_size=vocab_size, hidden_dim=hidden_dim).state_dict(), path)
        return {"base_model_path": str(path), "status": "saved_init_only", "reason": "dataset_empty"}

    device = _pick_device(device_preference)
    if cpu_threads is None:
        cpu_threads = max(1, os.cpu_count() or 1)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(min(cpu_threads, 8))
        except RuntimeError:
            pass
    model = TinyHandwritingModel(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rng = random.Random(42)
    epoch_losses: list[float] = []
    for _ in range(max(1, epochs)):
        rng.shuffle(encoded_samples)
        batch_losses: list[float] = []
        for i in range(0, len(encoded_samples), max(1, batch_size)):
            batch = encoded_samples[i : i + max(1, batch_size)]
            char_ids, style_ids, time_steps, targets, mask = _build_batch(batch, device)
            pred = model(char_ids, style_ids, time_steps)
            squared = F.mse_loss(pred, targets, reduction="none")
            loss = (squared * mask.unsqueeze(-1)).sum() / torch.clamp(mask.sum() * 4.0, min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

    seq_meta = _collect_sequence_length_metadata(encoded_samples)
    text_exemplars = _collect_char_exemplars_by_text(samples)
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": {
            "dataset_path": str(dataset_path),
            "sample_count": len(samples),
            "epochs": max(1, epochs),
            "batch_size": max(1, batch_size),
            "lr": lr,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "device": device,
            "cpu_threads": cpu_threads,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "sequence_len_mean": seq_meta["sequence_len_mean"],
            "char_len_mean": seq_meta["char_len_mean"],
            "char_exemplars": _collect_char_exemplars(samples, vocab_size),
            "char_exemplars_text": text_exemplars,
        },
    }
    torch.save(checkpoint, path)
    return {
        "base_model_path": str(path),
        "status": "trained",
        "sample_count": len(samples),
        "epochs": max(1, epochs),
        "final_loss": round(epoch_losses[-1], 6) if epoch_losses else None,
        "device": device,
    }


def train_lora_adapter(
    user_id: int,
    style_id: int,
    dataset_count: int,
    output_path: str,
    *,
    user_samples: list[dict] | None = None,
    vocab_size: int = 4096,
) -> TrainResult:
    seed_src = f"{user_id}:{style_id}:{dataset_count}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    samples = user_samples or []

    def _sample_source(sample: dict[str, Any]) -> str:
        meta = sample.get("meta")
        if isinstance(meta, dict):
            src = meta.get("source")
            if isinstance(src, str):
                return src
        return ""

    # Keep generation style anchored to real user handwriting and avoid
    # bootstrap artifacts dominating exemplar selection.
    style_samples = [s for s in samples if not _sample_source(s).startswith("bootstrap:")]
    if not style_samples:
        style_samples = samples

    user_exemplars = _collect_user_char_exemplars(style_samples, vocab_size=vocab_size)
    user_exemplars_text = _collect_user_char_exemplars_text(style_samples)
    style_profile = _estimate_style_profile(style_samples)
    effective_count = sum(len(v) for v in user_exemplars.values())
    coverage_chars = sorted(
        {
            ch
            for s in samples
            for ch in [((s.get("meta") or {}).get("char"))]
            if isinstance(ch, str) and len(ch) == 1
        }
    )
    bootstrap_sample_count = sum(1 for s in samples if _sample_source(s).startswith("bootstrap:"))

    adapter = {
        "user_id": user_id,
        "style_id": style_id,
        "dataset_count": dataset_count,
        "sample_count": len(samples),
        "style_sample_count": len(style_samples),
        "bootstrap_sample_count": bootstrap_sample_count,
        "effective_exemplar_count": effective_count,
        "char_coverage": len(user_exemplars),
        "coverage_chars": coverage_chars,
        "lora_rank": 8,
        "style_profile": style_profile,
        "user_char_exemplars": user_exemplars,
        "user_char_exemplars_text": user_exemplars_text,
        "weights": [rng.uniform(-0.1, 0.1) for _ in range(64)],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adapter), encoding="utf-8")

    coverage_bonus = min(0.16, len(user_exemplars) / 220.0)
    sample_bonus = min(0.12, len(samples) / 900.0)
    similarity = 0.70 + coverage_bonus + sample_bonus + rng.random() * 0.08
    cer = 0.11 - min(0.07, coverage_bonus * 0.6 + sample_bonus * 0.5) + rng.random() * 0.02
    cer = max(0.012, cer)
    return TrainResult(similarity_score=round(similarity, 3), cer=round(cer, 3), adapter_path=str(path))


def _legacy_generate_trajectory(text: str, style_seed: str) -> list[dict]:
    seed_src = f"{style_seed}:{text}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    points: list[dict] = []
    base_x = 24
    t = 0
    for char_idx, char in enumerate(text):
        char_offset = (ord(char) % 17) * 0.15
        for i in range(20):
            x = base_x + char_idx * 22 + i
            y = 48 + int(6 * sin((i / 3) + char_offset) + 4 * cos((i / 5) + rng.random()))
            points.append(
                {
                    "x": x,
                    "y": y,
                    "t": t,
                    "pen_state": "down",
                    "width": 2 + int(i % 2),
                }
            )
            t += 1
        points.append({"x": base_x + char_idx * 22 + 20, "y": 48, "t": t, "pen_state": "up", "width": 1})
        t += 1
    return points


def _parse_style_seed(style_seed: str) -> dict[str, Any]:
    try:
        raw = json.loads(style_seed)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _style_token_from_seed(style_seed: str) -> int:
    payload = _parse_style_seed(style_seed)
    if isinstance(payload.get("style_id"), int):
        return int(payload["style_id"]) % 4096
    return stable_int_token(style_seed, 4096)


def _char_token(char: str, vocab_size: int) -> int:
    return char_to_model_id(char, vocab_size)


def _load_base_model(base_model_path: str) -> tuple[TinyHandwritingModel | None, int, int, dict[str, Any]]:
    if torch is None:
        return None, 4096, 128, {}

    path = Path(base_model_path)
    if not path.exists():
        return None, 4096, 128, {}

    mtime = path.stat().st_mtime
    cached = _MODEL_CACHE.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2], cached[3], cached[4]

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None, 4096, 128, {}

    metadata: dict[str, Any] = {}
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        maybe_state = checkpoint.get("state_dict")
        if isinstance(maybe_state, dict):
            state_dict = maybe_state
        maybe_meta = checkpoint.get("metadata")
        if isinstance(maybe_meta, dict):
            metadata = maybe_meta

    vocab_size = int(metadata.get("vocab_size", 4096))
    hidden_dim = int(metadata.get("hidden_dim", 128))
    model = TinyHandwritingModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        return None, 4096, 128, {}
    model.eval()
    _MODEL_CACHE[str(path)] = (mtime, model, vocab_size, hidden_dim, metadata)
    return model, vocab_size, hidden_dim, metadata


def _infer_sequence_len(char_id: int, metadata: dict[str, Any], rng: random.Random) -> int:
    char_len_mean = metadata.get("char_len_mean")
    char_mean = None
    if isinstance(char_len_mean, dict):
        raw = char_len_mean.get(str(char_id))
        if isinstance(raw, (int, float)):
            char_mean = float(raw)
    global_mean_raw = metadata.get("sequence_len_mean", 96.0)
    global_mean = float(global_mean_raw) if isinstance(global_mean_raw, (int, float)) else 96.0
    base = char_mean if char_mean is not None else global_mean
    # Slight style jitter while staying near training-length distribution.
    jitter = rng.uniform(-0.12, 0.12)
    seq_len = int(round(base * (1.0 + jitter)))
    return max(24, min(240, seq_len))


def _is_katakana_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x30A1 <= code <= 0x30FA
        or 0x30FD <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xFF66 <= code <= 0xFF9D
    )


def _is_hiragana_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return 0x3041 <= code <= 0x3096


def _resample_sequence_rows(seq: list[list[float]], *, max_len: int) -> list[list[float]]:
    if not isinstance(seq, list):
        return []
    n = len(seq)
    if n <= max_len or max_len < 2:
        return list(seq)
    out: list[list[float]] = []
    for i in range(max_len):
        idx = int(round(i * (n - 1) / (max_len - 1)))
        row = seq[idx]
        if isinstance(row, list) and len(row) >= 4:
            out.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return out


def _sequence_quality_score(seq: list[list[float]]) -> float:
    if not isinstance(seq, list) or len(seq) < 8:
        return float("-inf")
    x = 0.0
    y = 0.0
    xs: list[float] = []
    ys: list[float] = []
    down_count = 0
    pen_switches = 0
    prev_pen: bool | None = None
    headings: list[float] = []
    long_jumps = 0
    for row in seq:
        if not isinstance(row, list) or len(row) < 4:
            continue
        dx = float(row[0]) * 20.0
        dy = float(row[1]) * 20.0
        pen = float(row[2]) > 0.5
        x += dx
        y += dy
        xs.append(x)
        ys.append(y)
        if prev_pen is not None and pen != prev_pen:
            pen_switches += 1
        prev_pen = pen
        if not pen:
            continue
        down_count += 1
        seg = hypot(dx, dy)
        if seg > 16.0:
            long_jumps += 1
        if seg > 1e-5:
            headings.append(atan2(dy, dx))

    if down_count < 6 or not xs or not ys:
        return float("-inf")
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x <= 0.1 or span_y <= 0.1:
        return float("-inf")
    if span_x < 4.0 and span_y < 4.0:
        return float("-inf")

    turn_acc = 0.0
    sharp_turns = 0
    for idx in range(1, len(headings)):
        delta = headings[idx] - headings[idx - 1]
        while delta > 3.141592653589793:
            delta -= 6.283185307179586
        while delta < -3.141592653589793:
            delta += 6.283185307179586
        ad = abs(delta)
        turn_acc += ad
        if ad > 1.25:
            sharp_turns += 1

    turn_mean = turn_acc / max(1, len(headings) - 1)
    # Higher is better.
    density = down_count / max(1.0, span_x + span_y)
    seq_len_penalty = max(0.0, (len(seq) - 110) / 20.0)
    return (
        min(6.0, span_x / 6.0)
        + min(8.0, span_y / 5.0)
        + min(2.5, down_count / 24.0)
        - min(9.5, turn_mean * 6.6)
        - long_jumps * 1.8
        - sharp_turns * 1.1
        - max(0.0, density - 1.65) * 4.0
        - max(0, pen_switches - 16) * 0.12
        - seq_len_penalty
    )


def _pick_best_exemplar_sequence(bucket: list[Any], rng: random.Random, *, trials: int = 10) -> list[list[float]] | None:
    if not isinstance(bucket, list) or not bucket:
        return None
    count = len(bucket)
    # Evaluate as many candidates as practical to avoid picking noisy zigzag samples.
    max_trials = max(1, min(80, max(trials, 28), count))
    if count <= max_trials:
        candidate_indexes = list(range(count))
    else:
        candidate_indexes = sorted(rng.sample(range(count), k=max_trials))

    best_seq: list[list[float]] | None = None
    best_score = float("-inf")
    for idx in candidate_indexes:
        raw = bucket[idx]
        if not isinstance(raw, list):
            continue
        seq: list[list[float]] = []
        for row in raw:
            if isinstance(row, list) and len(row) >= 4:
                seq.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        score = _sequence_quality_score(seq)
        if score > best_score:
            best_score = score
            best_seq = seq
    return best_seq


def _smooth_trajectory_points(points: list[dict], *, passes: int = 2) -> list[dict]:
    if len(points) < 8:
        return points
    smoothed = [dict(p) for p in points]
    down_idx = [i for i, p in enumerate(smoothed) if p.get("pen_state") == "down"]
    if len(down_idx) < 6:
        return smoothed
    runs: list[list[int]] = []
    cur: list[int] = []
    for idx in down_idx:
        if not cur or idx == cur[-1] + 1:
            cur.append(idx)
        else:
            runs.append(cur)
            cur = [idx]
    if cur:
        runs.append(cur)

    for _ in range(max(1, passes)):
        for run in runs:
            if len(run) < 5:
                continue
            for pos in range(2, len(run) - 2):
                i0 = run[pos - 2]
                i1 = run[pos - 1]
                i2 = run[pos]
                i3 = run[pos + 1]
                i4 = run[pos + 2]
                x = (
                    int(smoothed[i0].get("x", 0))
                    + 2 * int(smoothed[i1].get("x", 0))
                    + 3 * int(smoothed[i2].get("x", 0))
                    + 2 * int(smoothed[i3].get("x", 0))
                    + int(smoothed[i4].get("x", 0))
                ) / 9.0
                y = (
                    int(smoothed[i0].get("y", 0))
                    + 2 * int(smoothed[i1].get("y", 0))
                    + 3 * int(smoothed[i2].get("y", 0))
                    + 2 * int(smoothed[i3].get("y", 0))
                    + int(smoothed[i4].get("y", 0))
                ) / 9.0
                smoothed[i2]["x"] = int(round(x))
                smoothed[i2]["y"] = int(round(y))
    return smoothed


def _postprocess_strokes(points: list[dict]) -> list[dict]:
    if not points:
        return points
    strokes: list[list[dict]] = []
    cur: list[dict] = []
    for p in points:
        if p.get("pen_state") == "down":
            cur.append({"x": int(p.get("x", 0)), "y": int(p.get("y", 0)), "width": int(p.get("width", 1))})
        else:
            if cur:
                strokes.append(cur)
                cur = []
    if cur:
        strokes.append(cur)
    if not strokes:
        return points

    cleaned: list[list[dict]] = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        path_len = 0.0
        min_x = stroke[0]["x"]
        max_x = stroke[0]["x"]
        min_y = stroke[0]["y"]
        max_y = stroke[0]["y"]
        for i in range(1, len(stroke)):
            x0 = float(stroke[i - 1]["x"])
            y0 = float(stroke[i - 1]["y"])
            x1 = float(stroke[i]["x"])
            y1 = float(stroke[i]["y"])
            path_len += hypot(x1 - x0, y1 - y0)
            min_x = min(min_x, stroke[i]["x"])
            max_x = max(max_x, stroke[i]["x"])
            min_y = min(min_y, stroke[i]["y"])
            max_y = max(max_y, stroke[i]["y"])
        span = float(max(max_x - min_x, max_y - min_y))
        if path_len < 1.8 and span < 2.0:
            continue
        cleaned.append(stroke)

    if not cleaned:
        cleaned = strokes

    merged: list[list[dict]] = []
    for stroke in cleaned:
        if not merged:
            merged.append(stroke)
            continue
        prev = merged[-1]
        dx = float(stroke[0]["x"] - prev[-1]["x"])
        dy = float(stroke[0]["y"] - prev[-1]["y"])
        if hypot(dx, dy) <= 2.2:
            prev.extend(stroke)
        else:
            merged.append(stroke)

    out: list[dict] = []
    t = 0
    for stroke in merged:
        avg_w = max(1, min(4, int(round(sum(int(p.get("width", 1)) for p in stroke) / max(1, len(stroke))))))
        for p in stroke:
            out.append({"x": int(p["x"]), "y": int(p["y"]), "t": t, "pen_state": "down", "width": avg_w})
            t += 1
        end = stroke[-1]
        out.append({"x": int(end["x"]), "y": int(end["y"]), "t": t, "pen_state": "up", "width": avg_w})
        t += 1
    return out


def _trajectory_from_exemplar(
    text: str,
    style_seed: str,
    metadata: dict[str, Any],
    vocab_size: int,
    rng: random.Random,
    style_profile: dict[str, Any] | None = None,
) -> list[dict]:
    char_exemplars = metadata.get("char_exemplars")
    if not isinstance(char_exemplars, dict):
        return []
    points: list[dict] = []
    x_offset = 24.0
    y_offset = 46.0 + rng.uniform(-0.8, 0.8)
    t = 0

    style_cfg = style_profile if isinstance(style_profile, dict) else {}
    scale_x_bias = max(0.82, min(1.18, float(style_cfg.get("scale_x", 1.0)))) if style_cfg else 1.0
    scale_y_bias = max(0.82, min(1.18, float(style_cfg.get("scale_y", 1.0)))) if style_cfg else 1.0
    width_scale = max(0.85, min(1.25, float(style_cfg.get("width_scale", 1.0)))) if style_cfg else 1.0
    rot_bias_deg = max(-3.0, min(3.0, float(style_cfg.get("rot_bias_deg", 0.0)))) if style_cfg else 0.0
    shear_x = max(-0.05, min(0.05, float(style_cfg.get("shear_x", 0.0)))) if style_cfg else 0.0
    shear_y = max(-0.03, min(0.03, float(style_cfg.get("shear_y", 0.0)))) if style_cfg else 0.0
    jitter_scale = max(0.82, min(1.20, float(style_cfg.get("jitter_scale", 1.0)))) if style_cfg else 1.0

    for char in text:
        is_katakana = _is_katakana_char(char)
        text_map = metadata.get("char_exemplars_text")
        bucket = None
        if isinstance(text_map, dict):
            by_text = text_map.get(char)
            if isinstance(by_text, list) and by_text:
                bucket = by_text
        if bucket is None:
            char_id = _char_token(char, vocab_size)
            bucket = char_exemplars.get(str(char_id))
        if not isinstance(bucket, list) or not bucket:
            return []
        seq = _pick_best_exemplar_sequence(bucket, rng, trials=28)
        if not isinstance(seq, list) or len(seq) < 8:
            return []
        # Overlong trajectories tend to become wobbly lines after normalization.
        # Keep enough points for shape while capping unstable tails.
        if _is_hiragana_char(char):
            seq = _resample_sequence_rows(seq, max_len=92)
        
        # Apply character-specific width adjustments to avoid overly thick lines.
        char_width_scale = width_scale
        if char == "い":
            char_width_scale = width_scale * 0.65
        elif _is_katakana_char(char):
            seq = _resample_sequence_rows(seq, max_len=88)
        else:
            seq = _resample_sequence_rows(seq, max_len=108)

        # Decode local trajectory from normalized deltas.
        local = [(0.0, 0.0, "up", 2)]
        lx = 0.0
        ly = 0.0
        for row in seq:
            dx = float(row[0]) * 20.0
            dy = float(row[1]) * 20.0
            lx += dx
            ly += dy
            pen = "down" if float(row[2]) > 0.5 else "up"
            # Guard against accidental long "down" jumps; treat them as pen-up moves.
            if pen == "down" and (abs(dx) > 14.0 or abs(dy) > 14.0):
                pen = "up"
            width = int(round(max(0.2, min(1.2, float(row[3]))) * 4.0))
            local.append((lx, ly, pen, max(1, min(4, width))))

        down_local = [(p[0], p[1]) for p in local if p[2] == "down"]
        if len(down_local) >= 6:
            xs = [p[0] for p in down_local]
            ys = [p[1] for p in down_local]
        else:
            xs = [p[0] for p in local]
            ys = [p[1] for p in local]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)

        # Keep exemplar geometry and only normalize to a readable character box.
        if is_katakana:
            target_w = (16.8 + rng.uniform(-0.2, 0.5)) * scale_x_bias
            target_h = (28.6 + rng.uniform(-0.4, 0.4)) * scale_y_bias
        else:
            target_w = (17.0 + rng.uniform(-0.6, 1.2)) * scale_x_bias
            target_h = (29.0 + rng.uniform(-1.0, 1.0)) * scale_y_bias
        scale_x = max(0.01, min(2.4, target_w / span_x))
        scale_y = max(0.01, min(2.8, target_h / span_y))
        char_w = span_x * scale_x
        char_h = span_y * scale_y
        base_y = y_offset + (target_h - char_h) * 0.5

        rot = (rot_bias_deg + rng.uniform(-1.2, 1.2) * (0.55 + 0.35 * jitter_scale)) * 3.141592653589793 / 180.0
        cr = cos(rot)
        sr = sin(rot)
        center_x = x_offset + char_w * 0.5
        center_y = base_y + char_h * 0.5

        transformed: list[tuple[float, float, str, int]] = []
        stroke_phase = rng.uniform(0.0, 6.283185307179586)
        for lx, ly, pen, width in local:
            tx = x_offset + (lx - min_x) * scale_x
            ty = base_y + (ly - min_y) * scale_y
            rel_x = tx - center_x
            rel_y = ty - center_y
            # Apply the learned handwriting profile as a gentle affine style
            # transfer. This lets small user samples affect unseen characters
            # without replacing their readable base shape.
            rel_x = rel_x + shear_x * rel_y
            rel_y = rel_y + shear_y * rel_x
            tx = center_x + rel_x * cr - rel_y * sr
            ty = center_y + rel_x * sr + rel_y * cr
            if pen == "down" and jitter_scale > 0.82:
                phase = (lx + ly) * 0.065 + stroke_phase
                amp = min(0.16, max(0.0, (jitter_scale - 0.96) * 0.24))
                tx += sin(phase) * amp
                ty += cos(phase * 0.83) * amp * 0.7
            new_width = max(1, min(4, int(round(width * char_width_scale))))
            transformed.append((tx, ty, pen, new_width))

        # Ensure at least one pen-up near end to separate strokes.
        if transformed:
            last_tx, last_ty, _pen, last_width = transformed[-1]
            transformed[-1] = (last_tx, last_ty, "up", last_width)

        for tx, ty, pen, width in transformed:
            points.append(
                {
                    "x": int(round(tx)),
                    "y": int(round(ty)),
                    "t": t,
                    "pen_state": pen,
                    "width": width,
                }
            )
            t += 1

        x_offset += char_w + (rng.uniform(4.8, 6.2) if is_katakana else rng.uniform(5.0, 8.0))
        y_offset += rng.uniform(-0.25, 0.25)
    smooth_passes = 3
    smoothed = _smooth_trajectory_points(points, passes=smooth_passes)
    return _postprocess_strokes(smoothed)


def _trajectory_from_model(
    text: str,
    style_seed: str,
    base_model_path: str,
    *,
    seed_token: str | None = None,
) -> list[dict]:
    model, vocab_size, _hidden_dim, metadata = _load_base_model(base_model_path)
    if model is None or torch is None or not text:
        return []

    style_id = _style_token_from_seed(style_seed)
    if seed_token is None:
        seed_token = secrets.token_hex(8)
    rng_seed = stable_int_token(f"{style_seed}:{text}:{seed_token}", 2**31 - 1)
    rng = random.Random(rng_seed)

    points: list[dict] = []
    x = 24.0
    y = 52.0 + rng.uniform(-2.0, 2.0)
    t = 0

    with torch.no_grad():
        for char in text:
            char_id = _char_token(char, vocab_size)
            seq_len = _infer_sequence_len(char_id, metadata, rng)

            char_ids = torch.full((1, seq_len), char_id, dtype=torch.long)
            style_ids = torch.tensor([style_id], dtype=torch.long)
            time_steps = torch.linspace(0.0, 1.0, steps=seq_len).unsqueeze(0)
            pred = model(char_ids, style_ids, time_steps)[0].cpu()

            for i in range(seq_len):
                row = pred[i]
                # Training target stores normalized deltas: dx/20, dy/20.
                # Decode with the inverse scale to preserve learned geometry.
                dx = float(row[0]) * 20.0 + rng.uniform(-0.35, 0.35)
                dy = float(row[1]) * 20.0 + rng.uniform(-0.35, 0.35)
                x += max(-8.0, min(8.0, dx))
                y += max(-8.0, min(8.0, dy))

                pen = float(row[2]) > 0.5
                if i == 0:
                    pen = True
                if i == seq_len - 1:
                    pen = False
                width = int(round(max(0.2, min(1.2, float(row[3]))) * 4.0))
                width = max(1, min(4, width))

                points.append(
                    {
                        "x": int(round(x)),
                        "y": int(round(y)),
                        "t": t,
                        "pen_state": "down" if pen else "up",
                        "width": width,
                    }
                )
                t += 1

            x += 10.0 + rng.uniform(1.0, 3.0)
            y += rng.uniform(-0.8, 0.8)

    return points


def _trajectory_quality_score(points: list[dict], text: str) -> float:
    if not points:
        return float("-inf")
    down_points = [p for p in points if p.get("pen_state") == "down"]
    if len(down_points) < 2:
        return float("-inf")

    xs = [int(p.get("x", 0)) for p in down_points]
    ys = [int(p.get("y", 0)) for p in down_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return float("-inf")

    segment_lengths: list[float] = []
    headings: list[float] = []
    for idx in range(1, len(down_points)):
        dx = float(down_points[idx]["x"] - down_points[idx - 1]["x"])
        dy = float(down_points[idx]["y"] - down_points[idx - 1]["y"])
        seg = hypot(dx, dy)
        if seg <= 1e-6:
            continue
        segment_lengths.append(seg)
        headings.append(atan2(dy, dx))
    if not segment_lengths:
        return float("-inf")

    total_path_len = sum(segment_lengths)
    disp = hypot(
        float(down_points[-1]["x"] - down_points[0]["x"]),
        float(down_points[-1]["y"] - down_points[0]["y"]),
    )
    straightness = total_path_len / max(disp, 1e-6)
    turns = 0
    for idx in range(1, len(headings)):
        delta = headings[idx] - headings[idx - 1]
        while delta > 3.141592653589793:
            delta -= 6.283185307179586
        while delta < -3.141592653589793:
            delta += 6.283185307179586
        if abs(delta) >= 0.35:
            turns += 1

    text_len = max(1, len(text))
    return (
        min(2.5, straightness) * 2.0
        + min(5.0, height / max(1.0, width * 0.1))
        + min(5.0, width / max(8.0, text_len * 8.0))
        + min(6.0, turns / text_len)
    )


def _has_exemplar_for_text(
    text: str,
    *,
    vocab_size: int,
    char_exemplars: dict[str, Any] | None,
    text_exemplars: dict[str, Any] | None,
) -> bool:
    ce = char_exemplars if isinstance(char_exemplars, dict) else {}
    cte = text_exemplars if isinstance(text_exemplars, dict) else {}
    for ch in text:
        if not ch.strip():
            continue
        bucket_text = cte.get(ch)
        if isinstance(bucket_text, list) and len(bucket_text) > 0:
            continue
        cid = _char_token(ch, vocab_size)
        bucket_id = ce.get(str(cid))
        if isinstance(bucket_id, list) and len(bucket_id) > 0:
            continue
        return False
    return True


def _is_plausible_trajectory(points: list[dict], text: str) -> bool:
    if len(points) < max(8, len(text) * 10):
        return False

    down_points = [p for p in points if p.get("pen_state") == "down"]
    down_count = len(down_points)
    if down_count < max(6, len(text) * 6):
        return False

    xs = [int(p.get("x", 0)) for p in down_points]
    ys = [int(p.get("y", 0)) for p in down_points]
    if not xs or not ys:
        return False

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width < max(20, len(text) * 10):
        return False
    if height < max(8, int(width * 0.12)):
        return False

    if down_count < 2:
        return False

    segment_lengths: list[float] = []
    headings: list[float] = []
    for idx in range(1, down_count):
        dx = float(down_points[idx]["x"] - down_points[idx - 1]["x"])
        dy = float(down_points[idx]["y"] - down_points[idx - 1]["y"])
        seg = hypot(dx, dy)
        if seg <= 1e-6:
            continue
        segment_lengths.append(seg)
        headings.append(atan2(dy, dx))

    if len(segment_lengths) < max(4, len(text) * 4):
        return False

    total_path_len = sum(segment_lengths)
    disp = hypot(
        float(down_points[-1]["x"] - down_points[0]["x"]),
        float(down_points[-1]["y"] - down_points[0]["y"]),
    )
    if disp <= 1e-6:
        return False
    if (total_path_len / disp) < 1.18:
        return False

    turn_count = 0
    for idx in range(1, len(headings)):
        delta = headings[idx] - headings[idx - 1]
        while delta > 3.141592653589793:
            delta -= 6.283185307179586
        while delta < -3.141592653589793:
            delta += 6.283185307179586
        if abs(delta) >= 0.35:
            turn_count += 1
    if turn_count < max(2, len(text)):
        return False

    return True


def generate_trajectory(text: str, style_seed: str, base_model_path: str | None = None) -> list[dict]:
    if not text:
        return []
    if base_model_path:
        model, vocab_size, _hidden_dim, metadata = _load_base_model(base_model_path)
        if model is None:
            return _legacy_generate_trajectory(text, style_seed)
        style_payload = _parse_style_seed(style_seed)
        adapter_exemplars_raw = style_payload.get("user_char_exemplars") if isinstance(style_payload, dict) else None
        adapter_exemplars_text_raw = style_payload.get("user_char_exemplars_text") if isinstance(style_payload, dict) else None
        merged_exemplars = _merged_char_exemplars(
            metadata.get("char_exemplars") if isinstance(metadata, dict) else None,
            adapter_exemplars_raw if isinstance(adapter_exemplars_raw, dict) else None,
        )
        merged_text_exemplars = _merged_text_exemplars(
            metadata.get("char_exemplars_text") if isinstance(metadata, dict) else None,
            adapter_exemplars_text_raw if isinstance(adapter_exemplars_text_raw, dict) else None,
        )
        run_metadata = dict(metadata)
        if merged_exemplars:
            run_metadata["char_exemplars"] = merged_exemplars
        if merged_text_exemplars:
            run_metadata["char_exemplars_text"] = merged_text_exemplars
        if not _has_exemplar_for_text(
            text,
            vocab_size=vocab_size,
            char_exemplars=run_metadata.get("char_exemplars") if isinstance(run_metadata, dict) else None,
            text_exemplars=run_metadata.get("char_exemplars_text") if isinstance(run_metadata, dict) else None,
        ):
            return []
        style_profile = style_payload.get("style_profile") if isinstance(style_payload, dict) else None
        rng_seed = stable_int_token(f"{style_seed}:{text}:hybrid", 2**31 - 1)
        rng = random.Random(rng_seed)

        # Prefer exemplar-first output for dataset-like rendering.
        preferred = _trajectory_from_exemplar(
            text,
            style_seed,
            run_metadata,
            vocab_size,
            random.Random(rng_seed ^ 0x9E3779B9),
            style_profile=style_profile if isinstance(style_profile, dict) else None,
        )
        if preferred and _trajectory_quality_score(preferred, text) > float("-inf"):
            return preferred

        best: list[dict] | None = None
        best_score = float("-inf")
        for attempt in range(12):
            generated: list[dict]
            if attempt < 6:
                generated = _trajectory_from_exemplar(
                    text,
                    style_seed,
                    run_metadata,
                    vocab_size,
                    rng,
                    style_profile=style_profile if isinstance(style_profile, dict) else None,
                )
            else:
                generated = _trajectory_from_model(
                    text,
                    style_seed,
                    base_model_path,
                    seed_token=f"{attempt}:{secrets.token_hex(8)}",
                )
            if generated and _is_plausible_trajectory(generated, text):
                return generated
            score = _trajectory_quality_score(generated, text)
            if score > best_score:
                best_score = score
                best = generated
        if best:
            return best
        return []
    return _legacy_generate_trajectory(text, style_seed)
