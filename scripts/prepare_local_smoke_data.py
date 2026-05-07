import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".flac"}


def write_json_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n")


def collect_clean_paths(clean_root: Path) -> list[str]:
    paths = [
        path.resolve()
        for path in sorted(clean_root.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No clean audio files found in {clean_root}.")
    return [str(path) for path in paths]


def make_room_noise(sr: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = int(sr * duration)
    white = rng.normal(0.0, 1.0, samples)
    slow = np.convolve(white, np.ones(1600) / 1600, mode="same")
    room_tone = 0.02 * np.sin(2 * np.pi * 120 * np.arange(samples) / sr)
    noise = 0.03 * white + 0.18 * slow + room_tone
    peak = np.max(np.abs(noise))
    return (0.2 * noise / peak).astype(np.float32)


def make_mild_rir(sr: int) -> np.ndarray:
    rir = np.zeros(int(0.35 * sr), dtype=np.float32)
    rir[0] = 1.0
    for delay_ms, gain in [(18, 0.35), (43, 0.2), (91, 0.12), (147, 0.06)]:
        rir[int(delay_ms * sr / 1000)] = gain
    tail_start = int(0.02 * sr)
    tail = 0.025 * np.exp(-np.linspace(0.0, 5.0, len(rir) - tail_start))
    rir[tail_start:] += tail.astype(np.float32)
    return rir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare synthetic local-smoke filelists for pipeline checks."
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("data/SR/clean"),
        help="Directory containing the local clean mini sample.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/local_smoke"),
        help="Directory for generated local-smoke assets and JSON files.",
    )
    parser.add_argument("--sampling-rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    noise_dir = args.output_root / "noise"
    rir_dir = args.output_root / "rir"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rir_dir.mkdir(parents=True, exist_ok=True)

    clean_paths = collect_clean_paths(args.clean_root)
    noise_path = noise_dir / "synthetic_room_noise.wav"
    rir_path = rir_dir / "synthetic_mild_room_rir.wav"

    sf.write(noise_path, make_room_noise(args.sampling_rate, args.duration, args.seed), args.sampling_rate)
    sf.write(rir_path, make_mild_rir(args.sampling_rate), args.sampling_rate)

    write_json_list(args.output_root / "clean.json", clean_paths)
    write_json_list(args.output_root / "noise.json", [str(noise_path.resolve())])
    write_json_list(args.output_root / "rir.json", [str(rir_path.resolve())])

    print(f"Wrote {args.output_root / 'clean.json'}")
    print(f"Wrote {args.output_root / 'noise.json'}")
    print(f"Wrote {args.output_root / 'rir.json'}")


if __name__ == "__main__":
    main()
