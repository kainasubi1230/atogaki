import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.base_dataset import build_base_dataset
from trainerlib.model import train_base_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainer utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-base-dataset")
    build.add_argument("--output", required=True, help="Output JSONL path")
    build.add_argument("--min-points", type=int, default=8)
    build.add_argument("--max-segments-per-dataset", type=int, default=500)

    base = sub.add_parser("train-base")
    base.add_argument("--output", required=True)
    base.add_argument("--dataset", default=None, help="Base dataset JSONL path")
    base.add_argument("--epochs", type=int, default=5)
    base.add_argument("--batch-size", type=int, default=32)
    base.add_argument("--lr", type=float, default=1e-3)
    base.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()
    if args.cmd == "build-base-dataset":
        result = build_base_dataset(
            output_path=args.output,
            min_points=args.min_points,
            max_segments_per_dataset=args.max_segments_per_dataset,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "train-base":
        result = train_base_model(
            args.output,
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_preference=args.device,
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
