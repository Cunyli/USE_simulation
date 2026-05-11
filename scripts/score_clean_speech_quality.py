from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.quality_utils import load_json_list
from scripts.quality_dnsmos import (
    P808_MODEL_URL,
    PRIMARY_MODEL_URL,
    DEFAULT_DNSMOS_DIR,
    download_if_not_exists as download_dnsmos_if_not_exists,
    infer_source,
    load_audio as load_dnsmos_audio,
    make_dnsmos_model,
    resolve_default_model_path,
    score_audio as score_dnsmos_audio,
)
from scripts.quality_vqscore import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_VQSCORE_ROOT,
    download_default_assets as download_vqscore_default_assets,
    make_device,
    make_vqscore_model,
    resolve_under_root,
    resolve_vqscore_root,
    score_audio as score_vqscore_audio,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a clean-speech JSON/scp with DNSMOS and VQScore, then write one merged quality JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input clean JSON list or Kaldi-style scp.")
    parser.add_argument("--output-json", type=Path, required=True, help="Merged quality JSON output.")
    parser.add_argument("--input-format", choices=["json", "scp"], default="json")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick partial run.")

    parser.add_argument("--dnsmos-device", default="cpu", help="DNSMOS device, e.g. cpu or cuda.")
    parser.add_argument("--dnsmos-primary-model", type=Path, default=DEFAULT_DNSMOS_DIR / "sig_bak_ovr.onnx")
    parser.add_argument("--dnsmos-p808-model", type=Path, default=DEFAULT_DNSMOS_DIR / "model_v8.onnx")
    parser.add_argument("--dnsmos-convert-to-torch", action="store_true")

    parser.add_argument("--vqscore-device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--vqscore-root", type=Path, default=DEFAULT_VQSCORE_ROOT)
    parser.add_argument("--vqscore-config", type=Path, default=None, help=f"Defaults to {DEFAULT_CONFIG}.")
    parser.add_argument("--vqscore-checkpoint", type=Path, default=None, help=f"Defaults to {DEFAULT_CHECKPOINT}.")

    parser.add_argument("--no-download", action="store_true", help="Do not auto-download missing model assets.")
    args = parser.parse_args()

    items = load_inputs(args.input, args.input_format)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError(f"No input audio found in {args.input}")

    args.dnsmos_primary_model = resolve_default_model_path(
        args.dnsmos_primary_model, "sig_bak_ovr.onnx"
    )
    args.dnsmos_p808_model = resolve_default_model_path(args.dnsmos_p808_model, "model_v8.onnx")
    args.vqscore_root = resolve_vqscore_root(args.vqscore_root)
    vqscore_config = resolve_under_root(args.vqscore_config, args.vqscore_root, DEFAULT_CONFIG)
    vqscore_checkpoint = resolve_under_root(
        args.vqscore_checkpoint, args.vqscore_root, DEFAULT_CHECKPOINT
    )

    if not args.no_download:
        download_dnsmos_if_not_exists(PRIMARY_MODEL_URL, args.dnsmos_primary_model)
        download_dnsmos_if_not_exists(P808_MODEL_URL, args.dnsmos_p808_model)
        download_vqscore_default_assets(args.vqscore_root)

    dnsmos_model = make_dnsmos_model(
        args.dnsmos_primary_model,
        args.dnsmos_p808_model,
        args.dnsmos_device,
        args.dnsmos_convert_to_torch,
    )
    vqscore_device = make_device(args.vqscore_device)
    vqscore_model, vqscore_cfg = make_vqscore_model(
        args.vqscore_root,
        vqscore_config,
        vqscore_checkpoint,
        vqscore_device,
    )

    from tqdm import tqdm

    started = time.perf_counter()
    records = {}
    for _uid, audio_path in tqdm(items):
        dnsmos_audio, fs = load_dnsmos_audio(audio_path)
        dnsmos_scores = score_dnsmos_audio(dnsmos_model, dnsmos_audio, fs)
        vqscore = score_vqscore_audio(vqscore_model, vqscore_cfg, audio_path, vqscore_device)
        records[audio_path] = {
            "source": infer_source(audio_path),
            "scores": {
                **dnsmos_scores,
                "vqscore": vqscore,
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "record_count": len(records),
                "elapsed_sec": elapsed,
                "sec_per_file": elapsed / len(records) if records else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
