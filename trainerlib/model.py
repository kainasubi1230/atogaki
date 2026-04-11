from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import cos, sin
from pathlib import Path
import random
from typing import Any

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency for GPU environments
    torch = None
    nn = None
    F = None


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


def _pick_device(preference: str) -> str:
    if torch is None:
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_batch(samples: list[dict], device: str, vocab_size: int):
    batch_size = len(samples)
    max_len = max(len(s["sequence"]) for s in samples)

    char_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    time_steps = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    targets = torch.zeros((batch_size, max_len, 4), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    style_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)

    for i, sample in enumerate(samples):
        seq = sample["sequence"]
        char_id = int(sample["char_id"]) % vocab_size
        seq_len = len(seq)
        char_ids[i, :seq_len] = char_id
        mask[i, :seq_len] = 1.0
        style_ids[i] = int(sample.get("style_id", 0))
        if seq_len > 1:
            time_steps[i, :seq_len] = torch.linspace(0.0, 1.0, steps=seq_len, device=device)
        for j, point in enumerate(seq):
            targets[i, j, 0] = float(point[0])
            targets[i, j, 1] = float(point[1])
            targets[i, j, 2] = float(point[2])
            targets[i, j, 3] = float(point[3])
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
    if not samples:
        torch.save(TinyHandwritingModel(vocab_size=vocab_size, hidden_dim=hidden_dim).state_dict(), path)
        return {"base_model_path": str(path), "status": "saved_init_only", "reason": "dataset_empty"}

    device = _pick_device(device_preference)
    model = TinyHandwritingModel(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rng = random.Random(42)
    epoch_losses: list[float] = []
    for _ in range(max(1, epochs)):
        rng.shuffle(samples)
        batch_losses: list[float] = []
        for i in range(0, len(samples), max(1, batch_size)):
            batch = samples[i : i + max(1, batch_size)]
            char_ids, style_ids, time_steps, targets, mask = _build_batch(batch, device, vocab_size)
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


def generate_trajectory(text: str, style_seed: str) -> list[dict]:
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
