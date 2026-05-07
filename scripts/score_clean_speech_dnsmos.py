from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from scripts.quality_filtering import load_json_list


TARGET_FS = 16000
PRIMARY_MODEL_URL = (
    "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/"
    "DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
P808_MODEL_URL = (
    "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/"
    "DNSMOS/DNSMOS/model_v8.onnx"
)


def download_if_not_exists(url: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"File already exists: {local_path}")
        return
    print(f"{local_path} not found, downloading from {url}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, local_path)
    print(f"Downloaded: {local_path}")


def load_scp(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        uid, audio_path = line.split(maxsplit=1)
        items.append((uid, audio_path))
    return items


def load_inputs(path: Path, input_format: str) -> list[tuple[str, str]]:
    if input_format == "scp":
        return load_scp(path)
    clean_paths = load_json_list(path)
    return [(Path(audio_path).stem, audio_path) for audio_path in clean_paths]


def load_audio(path: str) -> tuple[np.ndarray, int]:
    import numpy as np
    import soundfile as sf

    audio, fs = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1:
        raise ValueError(f"{path} must be mono or channel-last stereo audio, got shape {audio.shape}.")
    return audio, fs


def make_dnsmos_model(primary_model: Path, p808_model: Path, device: str, convert_to_torch: bool):
    try:
        from espnet2.enh.layers.dnsmos import DNSMOS_local
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DNSMOS scoring requires ESPnet and ONNX runtime support. "
            "Install compatible packages first, for example: "
            "pip install espnet onnxruntime"
        ) from exc

    use_gpu = "cuda" in device
    return DNSMOS_local(
        str(primary_model),
        str(p808_model),
        use_gpu=use_gpu,
        convert_to_torch=convert_to_torch,
    )


def score_audio(model, audio: np.ndarray, fs: int) -> dict[str, float]:
    if fs != TARGET_FS:
        import soxr

        audio = soxr.resample(audio, fs, TARGET_FS)
        fs = TARGET_FS
    scores = model(audio, fs)
    return normalize_dnsmos_scores(scores)


def normalize_dnsmos_scores(scores: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "ovrl": ("OVRL", "DNSMOS_OVRL", "ovrl", "overall"),
        "sig": ("SIG", "DNSMOS_SIG", "sig"),
        "bak": ("BAK", "DNSMOS_BAK", "bak"),
        "p808": ("P808", "P.808", "P808_MOS", "DNSMOS_P808", "p808"),
    }
    normalized: dict[str, float] = {}
    for name, keys in aliases.items():
        for key in keys:
            if key in scores:
                normalized[name] = float(scores[key])
                break
    if not normalized:
        raise ValueError(f"DNSMOS model returned no recognized score fields: {scores}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score clean speech with DNSMOS and write JSONL usable by filter_clean_speech.py."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON list or Kaldi-style scp.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL score file.")
    parser.add_argument("--input-format", choices=["json", "scp"], default="json")
    parser.add_argument("--device", default="cpu", help="DNSMOS device, e.g. cpu or cuda.")
    parser.add_argument("--primary-model", type=Path, default=Path("./eval/DNSMOS/sig_bak_ovr.onnx"))
    parser.add_argument("--p808-model", type=Path, default=Path("./eval/DNSMOS/model_v8.onnx"))
    parser.add_argument("--no-download", action="store_true", help="Do not auto-download DNSMOS ONNX files.")
    parser.add_argument("--convert-to-torch", action="store_true", help="Pass convert_to_torch=True to ESPnet DNSMOS.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick partial run.")
    args = parser.parse_args()

    if not args.no_download:
        download_if_not_exists(PRIMARY_MODEL_URL, args.primary_model)
        download_if_not_exists(P808_MODEL_URL, args.p808_model)
    if not args.primary_model.exists():
        raise FileNotFoundError(f"Primary DNSMOS model not found: {args.primary_model}")
    if not args.p808_model.exists():
        raise FileNotFoundError(f"P808 DNSMOS model not found: {args.p808_model}")

    items = load_inputs(args.input, args.input_format)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError(f"No input audio found in {args.input}")

    model = make_dnsmos_model(args.primary_model, args.p808_model, args.device, args.convert_to_torch)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    with args.output_jsonl.open("w") as writer:
        for uid, audio_path in tqdm(items):
            audio, fs = load_audio(audio_path)
            scores = score_audio(model, audio, fs)
            record = {
                "utt_id": uid,
                "path": audio_path,
                "source": infer_source(audio_path),
                **scores,
            }
            writer.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote {args.output_jsonl}")


def infer_source(audio_path: str) -> str:
    parts = Path(audio_path).parts
    for marker in ("EARS", "VCTK", "LibriTTS", "CommonVoice", "Common_Voice", "DNS", "MLS"):
        if marker in parts or any(marker.lower() in part.lower() for part in parts):
            return marker
    return ""


if __name__ == "__main__":
    main()
