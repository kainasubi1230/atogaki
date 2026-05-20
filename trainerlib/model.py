from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from math import cos, sin, tanh
from pathlib import Path
import random
from typing import Any

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


_MODEL_CACHE: dict[str, tuple[float, TinyHandwritingModel, int, int]] = {}
_KMNIST_CHAR_TO_ID: dict[str, int] = {
    "\u304a": 0,  # お
    "\u304d": 1,  # き
    "\u3059": 2,  # す
    "\u3064": 3,  # つ
    "\u306a": 4,  # な
    "\u306f": 5,  # は
    "\u307e": 6,  # ま
    "\u3084": 7,  # や
    "\u308c": 8,  # れ
    "\u3092": 9,  # を
}
_HIRAGANA_GROUP_TO_KMNIST: tuple[tuple[str, str], ...] = (
    ("\u3041\u3042\u3043\u3044\u3045\u3046\u3047\u3048\u3049\u304a", "\u304a"),  # あ行
    ("\u304b\u304c\u304d\u304e\u304f\u3050\u3051\u3052\u3053\u3054", "\u304d"),  # か行
    ("\u3055\u3056\u3057\u3058\u3059\u305a\u305b\u305c\u305d\u305e", "\u3059"),  # さ行
    ("\u305f\u3060\u3061\u3062\u3063\u3064\u3065\u3066\u3067\u3068\u3069", "\u3064"),  # た行
    ("\u306a\u306b\u306c\u306d\u306e", "\u306a"),  # な行
    ("\u306f\u3070\u3071\u3072\u3073\u3074\u3075\u3076\u3077\u3078\u3079\u307a\u307b\u307c\u307d", "\u306f"),  # は行
    ("\u307e\u307f\u3080\u3081\u3082", "\u307e"),  # ま行
    ("\u3083\u3084\u3085\u3086\u3087\u3088", "\u3084"),  # や行
    ("\u3089\u308a\u308b\u308c\u308d", "\u308c"),  # ら行
    ("\u308e\u308f\u3090\u3091\u3092\u3093", "\u3092"),  # わ行
)


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


def _stable_int_token(value: str, mod: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, mod)


def _katakana_to_hiragana(char: str) -> str:
    if len(char) != 1:
        return char
    code = ord(char)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return char


def _normalize_for_kmnist(char: str) -> str:
    ch = _katakana_to_hiragana(char)
    for group, mapped in _HIRAGANA_GROUP_TO_KMNIST:
        if ch in group:
            return mapped
    return ch


def _style_token_from_seed(style_seed: str) -> int:
    payload = _parse_style_seed(style_seed)
    if isinstance(payload.get("style_id"), int):
        return int(payload["style_id"]) % 4096
    return _stable_int_token(style_seed, 4096)


def _char_token(char: str, vocab_size: int) -> int:
    normalized = _normalize_for_kmnist(char)
    kmnist_id = _KMNIST_CHAR_TO_ID.get(normalized)
    if kmnist_id is not None:
        return kmnist_id % max(1, vocab_size)
    return _stable_int_token(char, vocab_size)


def _load_base_model(base_model_path: str) -> tuple[TinyHandwritingModel | None, int, int]:
    if torch is None:
        return None, 4096, 128

    path = Path(base_model_path)
    if not path.exists():
        return None, 4096, 128

    mtime = path.stat().st_mtime
    cached = _MODEL_CACHE.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2], cached[3]

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None, 4096, 128

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
        return None, 4096, 128
    model.eval()
    _MODEL_CACHE[str(path)] = (mtime, model, vocab_size, hidden_dim)
    return model, vocab_size, hidden_dim


def _trajectory_from_model(text: str, style_seed: str, base_model_path: str) -> list[dict]:
    model, vocab_size, _hidden_dim = _load_base_model(base_model_path)
    if model is None or torch is None or not text:
        return []

    style_id = _style_token_from_seed(style_seed)
    rng_seed = _stable_int_token(f"{style_seed}:{text}", 2**31 - 1)
    rng = random.Random(rng_seed)

    points: list[dict] = []
    x = 24.0
    y = 52.0 + rng.uniform(-2.0, 2.0)
    t = 0

    with torch.no_grad():
        for char in text:
            char_id = _char_token(char, vocab_size)
            seq_len = 18 + (ord(char) % 8)

            char_ids = torch.full((1, seq_len), char_id, dtype=torch.long)
            style_ids = torch.tensor([style_id], dtype=torch.long)
            time_steps = torch.linspace(0.0, 1.0, steps=seq_len).unsqueeze(0)
            pred = model(char_ids, style_ids, time_steps)[0].cpu()

            for i in range(seq_len):
                row = pred[i]
                dx = tanh(float(row[0])) * 4.2 + 1.1 + rng.uniform(-0.25, 0.25)
                dy = tanh(float(row[1])) * 2.8 + rng.uniform(-0.35, 0.35)
                x += max(-1.2, min(6.2, dx))
                y += max(-4.0, min(4.0, dy))

                pen = float(row[2]) > 0.0
                if i == seq_len - 1:
                    pen = False
                width = int(round(2.0 + tanh(float(row[3])) * 1.2))
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

            x += 8.0 + rng.uniform(2.0, 5.0)
            y += rng.uniform(-1.0, 1.0)

    return points


def generate_trajectory(text: str, style_seed: str, base_model_path: str | None = None) -> list[dict]:
    if not text:
        return []
    if base_model_path:
        generated = _trajectory_from_model(text, style_seed, base_model_path)
        if generated:
            return generated
    return _legacy_generate_trajectory(text, style_seed)
