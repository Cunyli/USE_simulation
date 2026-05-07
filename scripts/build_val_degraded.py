import argparse
import json
import random
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import yaml

from simulate_degradation import apply_degradation_with_wind, random_select_and_order
from scripts.quality_filtering import (
    QualityFilterConfig,
    filter_clean_paths,
    load_quality_scores,
    load_whitelist_patterns,
    quality_filter_enabled,
)


def load_json_list(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return data


def load_audio(path: str, target_sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.reshape(1, -1)


def make_output_path(clean_path: str, output_root: Path) -> Path:
    return output_root / Path(clean_path).with_suffix(".wav").name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline degraded validation audio from val_clean.json."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--clean-json",
        type=Path,
        default=Path("data/val_clean.json"),
        help="Validation clean JSON file.",
    )
    parser.add_argument(
        "--noise-json",
        type=Path,
        default=Path("data/train_noise.json"),
        help="Noise JSON file.",
    )
    parser.add_argument(
        "--rir-json",
        type=Path,
        default=Path("data/train_rir.json"),
        help="RIR JSON file.",
    )
    parser.add_argument(
        "--wind-noise-json",
        type=Path,
        default=None,
        help="Optional wind noise JSON file. Required when wind_noise can be selected.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/val_degraded"),
        help="Directory to store degraded validation audio.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/val_degraded.json"),
        help="Output degraded JSON file.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("data/val_degraded_manifest.json"),
        help="Output manifest for reproducibility.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for partial generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing degraded files.",
    )
    parser.add_argument(
        "--quality-scores",
        type=Path,
        default=None,
        help="Optional precomputed clean-speech quality scores for filtering.",
    )
    parser.add_argument(
        "--use-dnsmos-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable DNSMOS filtering. Overrides quality_filter.use_dnsmos_filter.",
    )
    parser.add_argument(
        "--dnsmos-threshold",
        type=float,
        default=None,
        help="DNSMOS threshold used when DNSMOS filtering is enabled.",
    )
    parser.add_argument(
        "--vqscore-threshold",
        type=float,
        default=None,
        help="Optional VQScore threshold for clean target filtering.",
    )
    parser.add_argument(
        "--quality-whitelist",
        type=Path,
        default=None,
        help="Optional JSON/text list of path or source substrings to keep regardless of scores.",
    )
    parser.add_argument(
        "--quality-manifest-json",
        type=Path,
        default=None,
        help="Optional manifest for quality-filter summary and rejected paths.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    clean_paths = load_json_list(args.clean_json)
    noise_paths = load_json_list(args.noise_json)
    rir_paths = load_json_list(args.rir_json)
    wind_noise_paths = load_json_list(args.wind_noise_json) if args.wind_noise_json else []

    if not clean_paths:
        raise ValueError(f"No clean validation files found in {args.clean_json}.")
    if not noise_paths:
        raise ValueError(f"No noise files found in {args.noise_json}.")
    if not rir_paths:
        raise ValueError(f"No RIR files found in {args.rir_json}.")
    if cfg["degradation_cfg"].get("wind_noise_prob", 0.0) > 0.0 and not wind_noise_paths:
        raise ValueError("--wind-noise-json is required when wind_noise_prob > 0.")

    quality_cfg = cfg.get("quality_filter", {})
    whitelist_patterns = list(quality_cfg.get("whitelist_patterns", []))
    if quality_cfg.get("whitelist"):
        whitelist_patterns.extend(load_whitelist_patterns(Path(quality_cfg["whitelist"])))
    whitelist_patterns.extend(load_whitelist_patterns(args.quality_whitelist))

    filter_config = QualityFilterConfig(
        use_dnsmos=(
            quality_cfg.get("use_dnsmos_filter", False)
            if args.use_dnsmos_filter is None
            else args.use_dnsmos_filter
        ),
        dnsmos_threshold=(
            quality_cfg.get("dnsmos_threshold", 3.0)
            if args.dnsmos_threshold is None
            else args.dnsmos_threshold
        ),
        dnsmos_fields=tuple(quality_cfg.get("dnsmos_fields", ["ovrl", "sig", "bak", "p808"])),
        vqscore_threshold=(
            quality_cfg.get("vqscore_threshold")
            if args.vqscore_threshold is None
            else args.vqscore_threshold
        ),
        whitelist_patterns=whitelist_patterns,
        missing_scores=quality_cfg.get("missing_scores", "fail"),
    )
    if quality_filter_enabled(filter_config):
        quality_scores = args.quality_scores or (
            Path(quality_cfg["scores"]) if quality_cfg.get("scores") else None
        )
        if quality_scores is None:
            raise ValueError(
                "Quality filtering is enabled, but no score file was provided. "
                "Use --quality-scores or quality_filter.scores."
            )
        filter_result = filter_clean_paths(
            clean_paths,
            load_quality_scores(quality_scores),
            filter_config,
        )
        clean_paths = filter_result.kept_paths
        print(f"Quality filter summary: {json.dumps(filter_result.summary(), sort_keys=True)}")
        if not clean_paths:
            raise ValueError("Quality filtering removed every clean file.")
        if args.quality_manifest_json:
            args.quality_manifest_json.parent.mkdir(parents=True, exist_ok=True)
            args.quality_manifest_json.write_text(
                json.dumps(
                    {
                        "scores": str(quality_scores),
                        "summary": filter_result.summary(),
                        "rejected": filter_result.rejected,
                        "whitelisted": filter_result.whitelisted,
                        "missing": filter_result.missing,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Wrote {args.quality_manifest_json}")

    if args.limit is not None:
        clean_paths = clean_paths[: args.limit]

    sr = cfg["stft_cfg"]["sampling_rate"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    degraded_paths: list[str] = []
    manifest_entries: list[dict] = []

    for index, clean_path in enumerate(clean_paths):
        item_seed = args.seed + index
        rng = random.Random(item_seed)
        np.random.seed(item_seed)

        noise_path = noise_paths[rng.randrange(len(noise_paths))]
        rir_path = rir_paths[rng.randrange(len(rir_paths))]
        wind_noise_path = None

        clean_audio = load_audio(clean_path, sr)
        noise_audio = load_audio(noise_path, sr)
        rir_audio = load_audio(rir_path, sr)

        degrad_cfgs, selected_degrads = random_select_and_order(cfg, seed=item_seed)
        wind_noise_audio = None
        if "wind_noise" in selected_degrads:
            wind_noise_path = wind_noise_paths[rng.randrange(len(wind_noise_paths))]
            wind_noise_audio = load_audio(wind_noise_path, sr)

        _, degraded_audio = apply_degradation_with_wind(
            cfg,
            clean_audio,
            noise_audio,
            rir_audio,
            wind_noise_audio,
            degrad_cfgs,
            selected_degrads,
            seed=item_seed,
        )

        out_path = make_output_path(clean_path, args.output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")

        sf.write(out_path, degraded_audio.squeeze(), sr, subtype="FLOAT")
        degraded_paths.append(str(out_path.resolve()))

        manifest_entries.append(
            {
                "index": index,
                "seed": item_seed,
                "clean_path": clean_path,
                "degraded_path": str(out_path.resolve()),
                "noise_path": noise_path,
                "rir_path": rir_path,
                "wind_noise_path": wind_noise_path,
                "selected_degradations": selected_degrads,
                "degradation_config": degrad_cfgs,
            }
        )

        if (index + 1) % 25 == 0 or index == len(clean_paths) - 1:
            print(f"Generated {index + 1}/{len(clean_paths)}")

    args.output_json.write_text(json.dumps(degraded_paths, indent=2) + "\n")
    args.manifest_json.write_text(json.dumps(manifest_entries, indent=2) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.manifest_json}")


if __name__ == "__main__":
    main()
