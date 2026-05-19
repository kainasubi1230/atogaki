from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
from urllib.request import urlretrieve

import numpy as np


K49_FILES = {
    "train_images": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
    "train_labels": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
    "test_images": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
    "test_labels": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz",
}

MNIST_FILES = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


@dataclass
class ImportStats:
    source: str
    requested_count: int
    imported_count: int
    skipped_count: int
    output_path: str
    image_shape: list[int]


def _download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urlretrieve(url, str(path))


def _load_npz(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if "arr_0" in arr:
        return arr["arr_0"]
    # Fallback for custom npz naming.
    first_key = next(iter(arr.keys()))
    return arr[first_key]


def _sample_indices(total_count: int, target_count: int, seed: int) -> list[int]:
    capped = min(total_count, max(0, target_count))
    rng = random.Random(seed)
    return rng.sample(range(total_count), capped)


def _char_id_from_label(source_name: str, label: int) -> int:
    token = f"{source_name}:{label}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 4096


def _image_to_sequence(image: np.ndarray, max_points: int, threshold: int) -> list[list[float]]:
    # image is expected to be uint8 grayscale.
    # Some datasets store ink as dark pixels, others as bright pixels.
    # Pick the sparser side as foreground to avoid tracing the background.
    dark_mask = image < threshold
    bright_mask = image > threshold
    dark_count = int(dark_mask.sum())
    bright_count = int(bright_mask.sum())
    mask = dark_mask if dark_count <= bright_count else bright_mask

    coords = np.argwhere(mask)
    if coords.shape[0] < 2:
        return []

    # Use snake-like scan (row by row with alternating direction) to reduce big jumps
    # while keeping conversion fast for large datasets.
    coords_sorted = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    ordered: list[tuple[int, int]] = []
    unique_rows = np.unique(coords_sorted[:, 0])
    reverse = False
    for y in unique_rows:
        row = coords_sorted[coords_sorted[:, 0] == y]
        xs = row[:, 1]
        if reverse:
            xs = xs[::-1]
        for x in xs:
            ordered.append((int(y), int(x)))
        reverse = not reverse

    stride = max(1, len(ordered) // max(2, max_points))
    sampled = ordered[::stride]
    if len(sampled) < 2:
        return []

    # coords are (y, x); convert to trajectory-like feature sequence.
    prev_y, prev_x = sampled[0]
    seq: list[list[float]] = []
    for y, x in sampled[1:]:
        dx = float(x - prev_x) / 20.0
        dy = float(y - prev_y) / 20.0
        seq.append([dx, dy, 1.0, 0.5])
        prev_y, prev_x = y, x
    if seq:
        seq[-1][2] = 0.0  # mark pen-up at end
    return seq


def _resolve_split(images: np.ndarray, labels: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels length mismatch")
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of train/test/all")
    return images, labels


def _iter_pairs(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        return _resolve_split(train_images, train_labels, split)
    if split == "test":
        return _resolve_split(test_images, test_labels, split)
    # all
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    return _resolve_split(all_images, all_labels, split)


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"invalid image magic: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != count * rows * cols:
        raise ValueError("image payload length mismatch")
    return arr.reshape(count, rows, cols)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid label magic: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != count:
        raise ValueError("label payload length mismatch")
    return arr


def import_k49_to_base_dataset(
    output_path: str,
    *,
    target_count: int = 40000,
    split: str = "train",
    seed: int = 42,
    cache_dir: str = "storage/public_cache/k49",
    append: bool = True,
    threshold: int = 200,
    max_points: int = 120,
    min_points: int = 8,
) -> ImportStats:
    cache = Path(cache_dir)
    train_imgs_path = cache / "k49-train-imgs.npz"
    train_labels_path = cache / "k49-train-labels.npz"
    test_imgs_path = cache / "k49-test-imgs.npz"
    test_labels_path = cache / "k49-test-labels.npz"

    _download_if_missing(K49_FILES["train_images"], train_imgs_path)
    _download_if_missing(K49_FILES["train_labels"], train_labels_path)
    _download_if_missing(K49_FILES["test_images"], test_imgs_path)
    _download_if_missing(K49_FILES["test_labels"], test_labels_path)

    train_images = _load_npz(train_imgs_path)
    train_labels = _load_npz(train_labels_path)
    test_images = _load_npz(test_imgs_path)
    test_labels = _load_npz(test_labels_path)

    images, labels = _iter_pairs(train_images, train_labels, test_images, test_labels, split)
    indices = _sample_indices(images.shape[0], target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    with out.open(mode, encoding="utf-8") as w:
        for idx in indices:
            image = images[idx]
            label = int(labels[idx])
            seq = _image_to_sequence(image, max_points=max_points, threshold=threshold)
            if len(seq) < min_points:
                skipped += 1
                continue
            sample = {
                "char_id": _char_id_from_label("k49", label),
                "style_id": 0,
                "dataset_id": -49,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": "k49",
                    "split": split,
                    "source_index": int(idx),
                    "label": label,
                },
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            imported += 1

    shape = list(images.shape[1:]) if images.ndim >= 2 else []
    return ImportStats(
        source="k49",
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=shape,
    )


def import_mnist_to_base_dataset(
    output_path: str,
    *,
    target_count: int = 40000,
    split: str = "train",
    seed: int = 42,
    cache_dir: str = "storage/public_cache/mnist",
    append: bool = True,
    threshold: int = 200,
    max_points: int = 120,
    min_points: int = 8,
) -> ImportStats:
    cache = Path(cache_dir)
    train_imgs_path = cache / "train-images-idx3-ubyte.gz"
    train_labels_path = cache / "train-labels-idx1-ubyte.gz"
    test_imgs_path = cache / "t10k-images-idx3-ubyte.gz"
    test_labels_path = cache / "t10k-labels-idx1-ubyte.gz"

    _download_if_missing(MNIST_FILES["train_images"], train_imgs_path)
    _download_if_missing(MNIST_FILES["train_labels"], train_labels_path)
    _download_if_missing(MNIST_FILES["test_images"], test_imgs_path)
    _download_if_missing(MNIST_FILES["test_labels"], test_labels_path)

    train_images = _read_idx_images_gz(train_imgs_path)
    train_labels = _read_idx_labels_gz(train_labels_path)
    test_images = _read_idx_images_gz(test_imgs_path)
    test_labels = _read_idx_labels_gz(test_labels_path)

    images, labels = _iter_pairs(train_images, train_labels, test_images, test_labels, split)
    indices = _sample_indices(images.shape[0], target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    with out.open(mode, encoding="utf-8") as w:
        for idx in indices:
            image = images[idx]
            label = int(labels[idx])
            seq = _image_to_sequence(image, max_points=max_points, threshold=threshold)
            if len(seq) < min_points:
                skipped += 1
                continue
            sample = {
                "char_id": _char_id_from_label("mnist", label),
                "style_id": 0,
                "dataset_id": -1,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": "mnist",
                    "split": split,
                    "source_index": int(idx),
                    "label": label,
                },
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            imported += 1

    shape = list(images.shape[1:]) if images.ndim >= 2 else []
    return ImportStats(
        source="mnist",
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=shape,
    )
