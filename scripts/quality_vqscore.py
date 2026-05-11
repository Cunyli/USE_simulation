from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from scripts.quality_utils import load_json_list


TARGET_FS = 16000
VQSCORE_RAW_BASE = "https://raw.githubusercontent.com/JasonSWFu/VQscore/main"
DEFAULT_CONFIG = "config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml"
DEFAULT_CHECKPOINT = (
    "exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/"
    "checkpoint-dnsmos_ovr_CC=0.835.pkl"
)
DEFAULT_ASSETS = (
    "models/VQVAE_models.py",
    "models/vector_quantize_pytorch.py",
    DEFAULT_CONFIG,
    DEFAULT_CHECKPOINT,
)
DEFAULT_VQSCORE_ROOT = Path("./eval/quality/VQscore")
LEGACY_VQSCORE_ROOT = Path("./external/VQscore")


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


def resolve_under_root(path: Path | None, root: Path, default_relative: str) -> Path:
    resolved = Path(default_relative) if path is None else path
    if resolved.is_absolute():
        return resolved
    return root / resolved


def resolve_vqscore_root(root: Path) -> Path:
    if root.exists():
        return root
    if root == DEFAULT_VQSCORE_ROOT and LEGACY_VQSCORE_ROOT.exists():
        print(f"Using legacy VQScore asset path: {LEGACY_VQSCORE_ROOT}")
        return LEGACY_VQSCORE_ROOT
    return root


def download_if_not_exists(url: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"File already exists: {local_path}")
        return
    print(f"{local_path} not found, downloading from {url}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, local_path)
    print(f"Downloaded: {local_path}")


def download_default_assets(root: Path) -> None:
    for relative_path in DEFAULT_ASSETS:
        download_if_not_exists(f"{VQSCORE_RAW_BASE}/{relative_path}", root / relative_path)


def make_device(device_name: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "VQScore scoring requires PyTorch and torchaudio. Install compatible packages first, "
            "for example: pip install torch torchaudio pyyaml"
        ) from exc

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for VQScore, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def make_vqscore_model(vqscore_root: Path, config_path: Path, checkpoint_path: Path, device):
    try:
        import torch
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "VQScore scoring requires torch, torchaudio, PyYAML, and the official VQscore repo."
        ) from exc

    if not vqscore_root.exists():
        raise FileNotFoundError(
            f"VQScore assets not found: {vqscore_root}. "
            "Rerun without --no-download to download the required official files."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"VQScore config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQScore checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(vqscore_root.resolve()))
    try:
        from models.VQVAE_models import VQVAE_QE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import models.VQVAE_models from {vqscore_root}. "
            "Make sure --vqscore-root points to the official JasonSWFu/VQscore checkout."
        ) from exc

    with config_path.open() as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    if config.get("task") != "Quality_Estimation":
        raise ValueError(f"{config_path} is not a Quality_Estimation config.")

    model = VQVAE_QE(**config["VQVAE_params"]).to(device).eval()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"]["VQVAE"])
    return model, config


def stft_magnitude(x, hop_size: int, fft_size: int = 512, win_length: int = 512):
    import torch

    window = torch.hann_window(win_length, device=x.device)
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window=window,
        return_complex=True,
    )
    return torch.clamp(x_stft.abs(), min=1e-7).transpose(2, 1)


def cos_loss(original, quantized):
    import torch

    eps = 1e-5
    original_norm = torch.norm(original, p=2, dim=-1, keepdim=True) + eps
    quantized_norm = torch.norm(quantized, p=2, dim=-1, keepdim=True) + eps
    cos_frame = torch.sum(original / original_norm * quantized / quantized_norm, dim=-1)
    return -torch.mean(cos_frame)


def load_audio(path: str, device):
    import torch
    import torchaudio

    speech, fs = torchaudio.load(path)
    if speech.ndim != 2:
        raise ValueError(f"{path} must load as [channels, samples], got shape {tuple(speech.shape)}.")
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)
    if fs != TARGET_FS:
        speech = torchaudio.functional.resample(speech, fs, TARGET_FS)
        fs = TARGET_FS
    return speech.to(device), fs


def score_audio(model, config: dict[str, Any], audio_path: str, device) -> float:
    import torch

    hop_size = 256
    speech, _ = load_audio(audio_path, device)
    with torch.no_grad():
        spectrum = stft_magnitude(speech, hop_size=hop_size)
        if config.get("input_transform") == "log1p":
            spectrum = torch.log1p(spectrum)
        z = model.CNN_1D_encoder(spectrum)
        zq, _indices, _vqloss, _distance = model.quantizer(z, stochastic=False, update=False)
        return float((-cos_loss(z.transpose(2, 1).cpu(), zq.cpu())).item())


def infer_source(audio_path: str) -> str:
    parts = Path(audio_path).parts
    for marker in ("EARS", "VCTK", "LibriTTS", "CommonVoice", "Common_Voice", "DNS", "MLS"):
        if marker in parts or any(marker.lower() in part.lower() for part in parts):
            return marker
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score clean speech with the official JasonSWFu/VQscore quality-estimation model "
            "and write JSONL usable by filter_clean_speech.py."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON list or Kaldi-style scp.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL score file.")
    parser.add_argument("--input-format", choices=["json", "scp"], default="json")
    parser.add_argument("--vqscore-root", type=Path, default=DEFAULT_VQSCORE_ROOT)
    parser.add_argument("--config", type=Path, default=None, help=f"VQScore config. Defaults to {DEFAULT_CONFIG}.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"VQScore checkpoint. Defaults to {DEFAULT_CHECKPOINT}.",
    )
    parser.add_argument(
        "--download-repo",
        action="store_true",
        help="Deprecated alias kept for old commands. Required VQScore assets are downloaded by default.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Do not auto-download missing VQScore assets. By default only the required official "
            "models/config/QE checkpoint files are downloaded, not the full repo."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick partial run.")
    args = parser.parse_args()

    args.vqscore_root = resolve_vqscore_root(args.vqscore_root)
    config_path = resolve_under_root(args.config, args.vqscore_root, DEFAULT_CONFIG)
    checkpoint_path = resolve_under_root(args.checkpoint, args.vqscore_root, DEFAULT_CHECKPOINT)
    if not args.no_download:
        if args.config is not None or args.checkpoint is not None:
            print("Auto-download only covers the default VQScore config and checkpoint.")
        download_default_assets(args.vqscore_root)

    items = load_inputs(args.input, args.input_format)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError(f"No input audio found in {args.input}")

    device = make_device(args.device)
    print(f"device: {device}")
    model, config = make_vqscore_model(args.vqscore_root, config_path, checkpoint_path, device)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    with args.output_jsonl.open("w") as writer:
        for uid, audio_path in tqdm(items):
            vqscore = score_audio(model, config, audio_path, device)
            record = {
                "utt_id": uid,
                "path": audio_path,
                "source": infer_source(audio_path),
                "vqscore": vqscore,
            }
            writer.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
