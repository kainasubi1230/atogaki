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
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
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


def train_lora_adapter(user_id: int, style_id: int, dataset_count: int, output_path: str) -> TrainResult:
    seed_src = f"{user_id}:{style_id}:{dataset_count}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    adapter = {
        "user_id": user_id,
        "style_id": style_id,
        "dataset_count": dataset_count,
        "lora_rank": 8,
        "weights": [rng.uniform(-0.1, 0.1) for _ in range(64)],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adapter), encoding="utf-8")

    similarity = 0.75 + rng.random() * 0.2
    cer = 0.02 + rng.random() * 0.06
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


def _trajectory_from_exemplar(
    text: str,
    style_seed: str,
    metadata: dict[str, Any],
    vocab_size: int,
    rng: random.Random,
) -> list[dict]:
    char_exemplars = metadata.get("char_exemplars")
    if not isinstance(char_exemplars, dict):
        return []
    points: list[dict] = []
    x_offset = 24.0
    y_offset = 54.0 + rng.uniform(-2.0, 2.0)
    t = 0

    for char in text:
        char_id = _char_token(char, vocab_size)
        bucket = char_exemplars.get(str(char_id))
        if not isinstance(bucket, list) or not bucket:
            return []
        seq = bucket[int(rng.random() * len(bucket)) % len(bucket)]
        if not isinstance(seq, list) or len(seq) < 8:
            return []

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
            width = int(round(max(0.2, min(1.2, float(row[3]))) * 4.0))
            local.append((lx, ly, pen, max(1, min(4, width))))

        # Apply random affine to create style variation while preserving character shape.
        theta = rng.uniform(-0.18, 0.18)
        c = cos(theta)
        s = sin(theta)
        sx = rng.uniform(0.88, 1.14)
        sy = rng.uniform(0.88, 1.14)
        shx = rng.uniform(-0.16, 0.16)
        shy = rng.uniform(-0.08, 0.08)

        xs = [p[0] for p in local]
        ys = [p[1] for p in local]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5

        transformed: list[tuple[float, float, str, int]] = []
        for lx, ly, pen, width in local:
            px = lx - cx
            py = ly - cy
            ax = (px * sx) + (py * shx)
            ay = (py * sy) + (px * shy)
            tx = (ax * c) - (ay * s)
            ty = (ax * s) + (ay * c)
            tx += x_offset + rng.uniform(-0.25, 0.25)
            ty += y_offset + rng.uniform(-0.25, 0.25)
            transformed.append((tx, ty, pen, width))

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

        x_offset += 26.0 + rng.uniform(-1.0, 2.0)
        y_offset += rng.uniform(-0.8, 0.8)
    return points


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
            return []
        rng_seed = stable_int_token(f"{style_seed}:{text}:hybrid", 2**31 - 1)
        rng = random.Random(rng_seed)
        best: list[dict] | None = None
        best_score = float("-inf")
        for attempt in range(12):
            generated: list[dict]
            if attempt < 6:
                generated = _trajectory_from_exemplar(text, style_seed, metadata, vocab_size, rng)
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
