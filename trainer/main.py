import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainer utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-base-dataset")
    build.add_argument("--output", required=True, help="Output JSONL path")
    build.add_argument("--min-points", type=int, default=8)
    build.add_argument("--max-segments-per-dataset", type=int, default=500)
    build.add_argument(
        "--object-key-prefix",
        default=None,
        help="Optional prefix filter for Dataset.object_key (e.g. trajectories/)",
    )
    build.add_argument(
        "--require-label",
        action="store_true",
        help="Skip unlabeled segments to avoid pseudo char ids",
    )

    base = sub.add_parser("train-base")
    base.add_argument("--output", required=True)
    base.add_argument("--dataset", default=None, help="Base dataset JSONL path")
    base.add_argument("--epochs", type=int, default=5)
    base.add_argument("--batch-size", type=int, default=32)
    base.add_argument("--lr", type=float, default=1e-3)
    base.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    base.add_argument("--cpu-threads", type=int, default=0, help="0: auto (all cores), >0: explicit")

    import_public = sub.add_parser("import-public-k49")
    import_public.add_argument("--output", default="storage/base/base_dataset.jsonl")
    import_public.add_argument("--count", type=int, default=40000)
    import_public.add_argument("--split", choices=["train", "test", "all"], default="train")
    import_public.add_argument("--seed", type=int, default=42)
    import_public.add_argument("--cache-dir", default="storage/public_cache/k49")
    import_public.add_argument("--append", action="store_true")
    import_public.add_argument("--replace", action="store_true")
    import_public.add_argument("--threshold", type=int, default=200)
    import_public.add_argument("--max-points", type=int, default=120)
    import_public.add_argument("--min-points", type=int, default=8)

    import_mnist = sub.add_parser("import-public-mnist")
    import_mnist.add_argument("--output", default="storage/base/base_dataset.jsonl")
    import_mnist.add_argument("--count", type=int, default=40000)
    import_mnist.add_argument("--split", choices=["train", "test", "all"], default="train")
    import_mnist.add_argument("--seed", type=int, default=42)
    import_mnist.add_argument("--cache-dir", default="storage/public_cache/mnist")
    import_mnist.add_argument("--append", action="store_true")
    import_mnist.add_argument("--replace", action="store_true")
    import_mnist.add_argument("--threshold", type=int, default=200)
    import_mnist.add_argument("--max-points", type=int, default=120)
    import_mnist.add_argument("--min-points", type=int, default=8)

    import_hiragana = sub.add_parser("import-hiragana-images")
    import_hiragana.add_argument("--output", default="storage/base/base_dataset_hiragana_images.jsonl")
    import_hiragana.add_argument("--input-dir", default="storage/public_cache/net_datasets/hiragana-dataset/hiragana_images")
    import_hiragana.add_argument("--count", type=int, default=1000)
    import_hiragana.add_argument("--seed", type=int, default=42)
    import_hiragana.add_argument("--append", action="store_true")
    import_hiragana.add_argument("--replace", action="store_true")
    import_hiragana.add_argument("--threshold", type=int, default=200)
    import_hiragana.add_argument("--max-points", type=int, default=120)
    import_hiragana.add_argument("--min-points", type=int, default=8)

    import_chars = sub.add_parser("import-character-images")
    import_chars.add_argument("--output", default="storage/base/base_dataset_character_images.jsonl")
    import_chars.add_argument("--input-dir", required=True)
    import_chars.add_argument("--count", type=int, default=2000)
    import_chars.add_argument("--seed", type=int, default=42)
    import_chars.add_argument("--append", action="store_true")
    import_chars.add_argument("--replace", action="store_true")
    import_chars.add_argument(
        "--script-filter",
        choices=["all", "hiragana", "katakana", "kana", "kanji"],
        default="all",
        help="keep only labels in the selected script",
    )
    import_chars.add_argument("--normalize-size", type=int, default=96, help="resize/pad image before sequence extraction")
    import_chars.add_argument(
        "--sequence-mode",
        choices=["centerline", "contour"],
        default="centerline",
        help="centerline avoids double-stroke bolding; contour traces outline",
    )
    import_chars.add_argument(
        "--quality-profile",
        choices=["default", "kanji_fine"],
        default="default",
        help="kanji_fine enables per-character thresholding and finer smoothing",
    )
    import_chars.add_argument("--max-points", type=int, default=120)
    import_chars.add_argument("--min-points", type=int, default=8)
    import_chars.add_argument("--min-shape-iou", type=float, default=0.26, help="drop samples that do not match source shape")
    import_chars.add_argument("--source-name", default="labeled-images")

    import_kkanji = sub.add_parser("import-public-kkanji")
    import_kkanji.add_argument("--output", default="storage/base/base_dataset_kkanji.jsonl")
    import_kkanji.add_argument("--count", type=int, default=20000)
    import_kkanji.add_argument("--seed", type=int, default=42)
    import_kkanji.add_argument("--cache-dir", default="storage/public_cache/kkanji")
    import_kkanji.add_argument("--append", action="store_true")
    import_kkanji.add_argument("--replace", action="store_true")
    import_kkanji.add_argument("--sequence-mode", choices=["centerline", "contour"], default="centerline")
    import_kkanji.add_argument("--quality-profile", choices=["default", "kanji_fine"], default="kanji_fine")
    import_kkanji.add_argument("--normalize-size", type=int, default=128)
    import_kkanji.add_argument("--max-points", type=int, default=240)
    import_kkanji.add_argument("--min-points", type=int, default=24)
    import_kkanji.add_argument("--min-shape-iou", type=float, default=0.12)
    import_kkanji.add_argument("--source-name", default="kkanji")

    import_kanjivg = sub.add_parser("import-kanjivg-joyo")
    import_kanjivg.add_argument("--output", default="storage/base/base_dataset_kanjivg_joyo.jsonl")
    import_kanjivg.add_argument("--repo-dir", default="storage/public_cache/net_datasets/kanjivg/kanji")
    import_kanjivg.add_argument("--chars-limit", type=int, default=0, help="0 means all joyo")
    import_kanjivg.add_argument("--variants-per-char", type=int, default=3)
    import_kanjivg.add_argument("--compact-scale", type=float, default=0.90, help="global size scale; smaller -> more compact")
    import_kanjivg.add_argument("--seed", type=int, default=42)
    import_kanjivg.add_argument("--append", action="store_true")
    import_kanjivg.add_argument("--replace", action="store_true")
    import_kanjivg.add_argument("--source-name", default="kanjivg-joyo")

    image_gen = sub.add_parser("generate-kana-images")
    image_gen.add_argument("--text", required=True, help="Characters to generate, e.g. あいうえお")
    image_gen.add_argument("--output-dir", default="storage/generated_kana")
    image_gen.add_argument("--count-per-char", type=int, default=4)
    image_gen.add_argument("--seed-prefix", default="stable")
    image_gen.add_argument("--input-dir", default="storage/public_cache/net_datasets/hiragana-dataset/hiragana_images")
    image_gen.add_argument("--size", type=int, default=512)
    image_gen.add_argument("--trials", type=int, default=12, help="candidate count per output (higher=more stable)")
    image_gen.add_argument("--max-quality-score", type=float, default=2500.0, help="lower is stricter")
    image_gen.add_argument("--retry-per-sample", type=int, default=4)

    args = parser.parse_args()
    if args.cmd == "build-base-dataset":
        from trainer.base_dataset import build_base_dataset

        result = build_base_dataset(
            output_path=args.output,
            min_points=args.min_points,
            max_segments_per_dataset=args.max_segments_per_dataset,
            object_key_prefix=args.object_key_prefix,
            require_label=args.require_label,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "train-base":
        from trainerlib.model import train_base_model

        result = train_base_model(
            args.output,
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_preference=args.device,
            cpu_threads=None if args.cpu_threads <= 0 else args.cpu_threads,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    if args.cmd == "import-public-k49":
        from trainer.public_dataset import import_k49_to_base_dataset

        if args.append and args.replace:
            raise ValueError("--append and --replace cannot be used together")
        append = True
        if args.replace:
            append = False
        if args.append:
            append = True
        result = import_k49_to_base_dataset(
            output_path=args.output,
            target_count=args.count,
            split=args.split,
            seed=args.seed,
            cache_dir=args.cache_dir,
            append=append,
            threshold=args.threshold,
            max_points=args.max_points,
            min_points=args.min_points,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "import-public-mnist":
        from trainer.public_dataset import import_mnist_to_base_dataset

        if args.append and args.replace:
            raise ValueError("--append and --replace cannot be used together")
        append = True
        if args.replace:
            append = False
        if args.append:
            append = True
        result = import_mnist_to_base_dataset(
            output_path=args.output,
            target_count=args.count,
            split=args.split,
            seed=args.seed,
            cache_dir=args.cache_dir,
            append=append,
            threshold=args.threshold,
            max_points=args.max_points,
            min_points=args.min_points,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "import-hiragana-images":
        from trainer.public_dataset import import_hiragana_images_to_base_dataset

        if args.append and args.replace:
            raise ValueError("--append and --replace cannot be used together")
        append = True
        if args.replace:
            append = False
        if args.append:
            append = True
        result = import_hiragana_images_to_base_dataset(
            output_path=args.output,
            input_dir=args.input_dir,
            target_count=args.count,
            seed=args.seed,
            append=append,
            threshold=args.threshold,
            max_points=args.max_points,
            min_points=args.min_points,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "import-character-images":
        from trainer.public_dataset import import_character_images_to_base_dataset

        if args.append and args.replace:
            raise ValueError("--append and --replace cannot be used together")
        append = True
        if args.replace:
            append = False
        if args.append:
            append = True
        result = import_character_images_to_base_dataset(
            output_path=args.output,
            input_dir=args.input_dir,
            target_count=args.count,
            seed=args.seed,
            append=append,
            script_filter=args.script_filter,
            normalize_size=args.normalize_size,
            sequence_mode=args.sequence_mode,
            quality_profile=args.quality_profile,
            min_shape_iou=args.min_shape_iou,
            max_points=args.max_points,
            min_points=args.min_points,
            source_name=args.source_name,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "import-public-kkanji":
        from trainer.public_dataset import import_kkanji_to_base_dataset

        if args.append and args.replace:
            raise ValueError("--append and --replace cannot be used together")
        append = True
        if args.replace:
            append = False
        if args.append:
            append = True
        result = import_kkanji_to_base_dataset(
            output_path=args.output,
            target_count=args.count,
            seed=args.seed,
            cache_dir=args.cache_dir,
            append=append,
            sequence_mode=args.sequence_mode,
            quality_profile=args.quality_profile,
            normalize_size=args.normalize_size,
            max_points=args.max_points,
            min_points=args.min_points,
            min_shape_iou=args.min_shape_iou,
            source_name=args.source_name,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "import-kanjivg-joyo":
        from trainer.kanjivg_dataset import import_kanjivg_joyo_to_base_dataset

        if args.append and args.replace:
            raise ValueError("--append and --replace cannot be used together")
        append = True
        if args.replace:
            append = False
        if args.append:
            append = True
        result = import_kanjivg_joyo_to_base_dataset(
            output_path=args.output,
            repo_dir=args.repo_dir,
            chars_limit=args.chars_limit,
            variants_per_char=args.variants_per_char,
            compact_scale=args.compact_scale,
            seed=args.seed,
            append=append,
            source_name=args.source_name,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
        return

    if args.cmd == "generate-kana-images":
        from trainerlib.kana_image import generate_kana_image_best, kana_image_quality_score

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for ch in args.text:
            if ch.isspace():
                continue
            for idx in range(max(1, args.count_per_char)):
                image = None
                best_image = None
                best_q = float("inf")
                for retry in range(max(1, int(args.retry_per_sample))):
                    seed = f"{args.seed_prefix}:{ch}:{idx}:retry:{retry}"
                    candidate = generate_kana_image_best(
                        ch,
                        style_seed=seed,
                        input_dir=args.input_dir,
                        size=max(128, int(args.size)),
                        trials=max(1, int(args.trials)),
                    )
                    q = kana_image_quality_score(candidate)
                    if q < best_q:
                        best_q = q
                        best_image = candidate
                    if q <= float(args.max_quality_score):
                        image = candidate
                        break
                if image is None:
                    image = best_image
                path = output_dir / f"{ch}_{idx:02d}.png"
                image.save(path)
                written.append(str(path))
        print(json.dumps({"output_count": len(written), "output_dir": str(output_dir), "files": written[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
