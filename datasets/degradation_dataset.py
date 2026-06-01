import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from torch.utils import data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_json_list(path):
    with Path(path).expanduser().open() as f:
        values = json.load(f)
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [str(x) for x in values]


def _read_mono(path, target_sr=None, start=0, stop=None):
    audio, sr = sf.read(path, start=start, stop=stop, always_2d=True)
    audio = audio[:, :1].T.astype(np.float32, copy=False)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        sr = target_sr
    return audio, sr


def _read_noise_like(path, target_sr, target_len, rng):
    info = sf.info(path)
    duration = target_len / target_sr
    source_len = int(np.ceil(duration * info.samplerate)) + 1
    if info.frames > source_len:
        start = int(rng.integers(0, info.frames - source_len + 1))
        return _read_mono(path, target_sr=target_sr, start=start, stop=start + source_len)[0]
    return _read_mono(path, target_sr=target_sr)[0]


def _match_length(audio, length):
    if audio.shape[1] > length:
        return audio[:, :length]
    if audio.shape[1] < length:
        return np.pad(audio, ((0, 0), (0, length - audio.shape[1])), constant_values=0)
    return audio


def _first_active_start(clean, seg_len, threshold=0.01, min_active_ratio=0.05):
    max_start = clean.shape[1] - seg_len
    if max_start <= 0:
        return 0
    active = (np.abs(clean[0]) > threshold).astype(np.float32)
    prefix = np.concatenate(([0.0], np.cumsum(active, dtype=np.float64)))
    counts = prefix[seg_len:] - prefix[:-seg_len]
    valid = np.flatnonzero(counts >= seg_len * min_active_ratio)
    return int(valid[0]) if valid.size else 0


def _crop_or_pad(audio, wav_len, sample_rate, random_start, rng):
    orig_len = audio.shape[1]
    if wav_len is None or wav_len <= 0:
        return audio, orig_len

    seg_len = int(float(wav_len) * sample_rate)
    if seg_len < orig_len:
        start = int(rng.integers(0, orig_len - seg_len + 1)) if random_start else 0
        audio = audio[:, start : start + seg_len]
    elif seg_len > orig_len:
        audio = np.pad(audio, ((0, 0), (0, seg_len - orig_len)), constant_values=0)
    return audio, orig_len


def _crop_or_pad_pair(noisy, clean, wav_len, sample_rate, random_start, rng):
    orig_len = min(noisy.shape[1], clean.shape[1])
    noisy = noisy[:, :orig_len]
    clean = clean[:, :orig_len]
    if wav_len is None or wav_len <= 0:
        return noisy, clean, orig_len

    seg_len = int(float(wav_len) * sample_rate)
    if seg_len < orig_len:
        start = int(rng.integers(0, orig_len - seg_len + 1)) if random_start else _first_active_start(clean, seg_len)
        noisy = noisy[:, start : start + seg_len]
        clean = clean[:, start : start + seg_len]
    elif seg_len > orig_len:
        pad = seg_len - orig_len
        noisy = np.pad(noisy, ((0, 0), (0, pad)), constant_values=0)
        clean = np.pad(clean, ((0, 0), (0, pad)), constant_values=0)
    return noisy, clean, orig_len


def _normalize_pair(noisy, clean):
    scale = 0.9 / (max(np.max(np.abs(noisy)), np.max(np.abs(clean))) + 1e-12)
    return noisy * scale, clean * scale


def _stable_seed(seed, *parts):
    text = "|".join([str(seed), *[str(x) for x in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


class OnlineDegradationDataset(data.Dataset):
    """PyTorch Dataset that generates degraded speech in memory with USE_simulation."""

    def __init__(
        self,
        clean_json,
        noise_json,
        rir_json,
        simulation_config,
        wind_noise_json=None,
        quality_json=None,
        wav_len=None,
        num_per_epoch=0,
        random_start=False,
        target_sample_rate=None,
        mode="train",
        normalize=True,
        seed=0,
        vqscore_threshold=None,
        use_dnsmos_filter=False,
        missing_scores="keep",
    ):
        if mode not in ["train", "valid", "validation", "test"]:
            raise ValueError(f"Unsupported mode: {mode}")
        if missing_scores not in ["keep", "drop", "error"]:
            raise ValueError("missing_scores must be one of: keep, drop, error")

        self.clean_json = Path(clean_json).expanduser()
        self.noise_json = Path(noise_json).expanduser()
        self.rir_json = Path(rir_json).expanduser()
        self.simulation_config = Path(simulation_config).expanduser()
        self.wind_noise_json = Path(wind_noise_json).expanduser() if wind_noise_json else None
        self.quality_json = Path(quality_json).expanduser() if quality_json else None
        self.wav_len = wav_len
        self.num_per_epoch = int(num_per_epoch)
        self.random_start = bool(random_start)
        self.target_sample_rate = target_sample_rate
        self.mode = "valid" if mode == "validation" else mode
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.vqscore_threshold = vqscore_threshold
        self.use_dnsmos_filter = bool(use_dnsmos_filter)
        self.missing_scores = missing_scores
        self.epoch = -1
        self._epoch_rng = random.Random(self.seed)

        self.cfg = yaml.safe_load(self.simulation_config.read_text())
        self.clean_paths = _load_json_list(self.clean_json)
        self.noise_paths = _load_json_list(self.noise_json)
        self.rir_paths = _load_json_list(self.rir_json)
        self.wind_noise_paths = _load_json_list(self.wind_noise_json) if self.wind_noise_json else []
        self.quality = self._load_quality(self.quality_json)
        self.meta = self._build_meta()
        if not self.meta:
            raise ValueError("No clean examples remain after filtering.")
        if not self.noise_paths:
            raise ValueError(f"noise_json is empty: {self.noise_json}")
        if not self.rir_paths:
            raise ValueError(f"rir_json is empty: {self.rir_json}")
        if self.cfg.get("degradation_cfg", {}).get("wind_noise_prob", 0.0) > 0 and not self.wind_noise_paths:
            raise ValueError("wind_noise_json is required when wind_noise_prob > 0.")

        self.sample_data_per_epoch(self.mode)

    @staticmethod
    def _load_quality(path):
        if path is None:
            return {}
        with path.open() as f:
            values = json.load(f)
        if not isinstance(values, dict):
            raise ValueError("quality_json must contain an object keyed by clean audio path.")
        return values

    def _passes_quality(self, clean_path):
        if not self.quality:
            return True
        item = self.quality.get(clean_path)
        scores = item.get("scores", {}) if isinstance(item, dict) else None
        if not scores:
            if self.missing_scores == "keep":
                return True
            if self.missing_scores == "drop":
                return False
            raise KeyError(f"Missing quality scores for clean path: {clean_path}")

        if self.vqscore_threshold is not None:
            value = scores.get("vqscore")
            if value is None:
                return self.missing_scores == "keep"
            if float(value) < float(self.vqscore_threshold):
                return False

        if self.use_dnsmos_filter:
            keys = ["ovrl", "sig", "bak", "p808"]
            if any(k not in scores for k in keys):
                return self.missing_scores == "keep"
            if any(float(scores[k]) < 3.0 for k in keys):
                return False
        return True

    def _build_meta(self):
        meta = []
        for idx, clean_path in enumerate(self.clean_paths):
            if not Path(clean_path).is_file():
                raise FileNotFoundError(f"Clean path does not exist: {clean_path}")
            if self._passes_quality(clean_path):
                meta.append({"id": f"item_{idx:06d}", "clean_path": clean_path})
        return meta

    def sample_data_per_epoch(self, mode=None):
        mode = mode or self.mode
        if self.num_per_epoch <= 0:
            self.meta_selected = list(self.meta)
        elif mode == "train":
            if self.num_per_epoch <= len(self.meta):
                self.meta_selected = self._epoch_rng.sample(self.meta, self.num_per_epoch)
            else:
                self.meta_selected = self._epoch_rng.choices(self.meta, k=self.num_per_epoch)
        else:
            self.meta_selected = self.meta[: self.num_per_epoch]
        self.epoch += 1

    def __getitem__(self, idx):
        from simulate_degradation import apply_degradation_with_wind, random_select_and_order

        info = self.meta_selected[idx]
        clean_path = info["clean_path"]
        if self.mode == "train":
            item_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        else:
            item_seed = _stable_seed(self.seed, self.mode, clean_path)

        py_rng = random.Random(item_seed)
        rng = np.random.default_rng(item_seed)
        clean, clean_sr = _read_mono(clean_path, target_sr=self.target_sample_rate)
        sample_rate = clean_sr
        clean, orig_len = _crop_or_pad(clean, self.wav_len, sample_rate, self.random_start, rng)
        target_len = clean.shape[1]

        noise_path = self.noise_paths[py_rng.randrange(len(self.noise_paths))]
        rir_path = self.rir_paths[py_rng.randrange(len(self.rir_paths))]
        noise = _read_noise_like(noise_path, sample_rate, target_len, rng)
        rir = _read_mono(rir_path, target_sr=sample_rate)[0]

        item_cfg = copy.deepcopy(self.cfg)
        item_cfg.setdefault("stft_cfg", {})["sampling_rate"] = int(sample_rate)
        degrad_cfgs, selected_degrads = random_select_and_order(item_cfg, seed=item_seed)

        wind_noise_path = None
        wind_noise = None
        if "wind_noise" in selected_degrads:
            wind_noise_path = self.wind_noise_paths[py_rng.randrange(len(self.wind_noise_paths))]
            wind_noise = _read_noise_like(wind_noise_path, sample_rate, target_len, rng)

        clean_out, noisy = apply_degradation_with_wind(
            item_cfg,
            clean,
            noise,
            rir,
            wind_noise,
            degrad_cfgs,
            selected_degrads,
            seed=item_seed,
        )
        noisy = _match_length(noisy, target_len)
        clean_out = _match_length(clean_out, target_len)
        if self.normalize:
            noisy, clean_out = _normalize_pair(noisy, clean_out)

        item_info = {
            "id": info["id"],
            "sample_rate": int(sample_rate),
            "length": int(orig_len),
            "clean_path": clean_path,
            "noise_path": noise_path,
            "rir_path": rir_path,
            "wind_noise_path": wind_noise_path or "",
            "seed": int(item_seed),
            "selected_degradations": ",".join(selected_degrads),
        }
        return noisy.astype(np.float32).squeeze(), clean_out.astype(np.float32).squeeze(), item_info

    def __len__(self):
        return len(self.meta_selected)


class FixedPairDataset(data.Dataset):
    """PyTorch Dataset for fixed noisy/clean pairs exported by USE_simulation."""

    def __init__(
        self,
        pair_manifest,
        wav_len=None,
        num_per_epoch=0,
        random_start=False,
        target_sample_rate=None,
        mode="train",
        normalize=True,
        seed=0,
    ):
        self.pair_manifest = Path(pair_manifest).expanduser()
        self.wav_len = wav_len
        self.num_per_epoch = int(num_per_epoch)
        self.random_start = bool(random_start)
        self.target_sample_rate = target_sample_rate
        self.mode = "valid" if mode == "validation" else mode
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self._epoch_rng = random.Random(self.seed)
        self.meta = self._load_manifest(self.pair_manifest)
        if not self.meta:
            raise ValueError(f"No pairs found in {self.pair_manifest}")
        self.sample_data_per_epoch(self.mode)

    @staticmethod
    def _load_manifest(path):
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            return [
                {
                    "id": row.get("uid") or Path(row["noisy_filepath"]).stem,
                    "noisy_path": row["noisy_filepath"],
                    "clean_path": row["clean_filepath"],
                    "sample_rate": int(row["sample_rate"]) if row.get("sample_rate") else None,
                }
                for row in rows
            ]

        with path.open() as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return [
            {
                "id": row.get("id") or row.get("uid") or Path(row["noisy_path"]).stem,
                "noisy_path": row.get("noisy_path") or row.get("noisy_filepath"),
                "clean_path": row.get("clean_path") or row.get("clean_filepath"),
                "sample_rate": row.get("sample_rate"),
            }
            for row in rows
        ]

    def sample_data_per_epoch(self, mode=None):
        mode = mode or self.mode
        if self.num_per_epoch <= 0 or self.num_per_epoch >= len(self.meta):
            self.meta_selected = list(self.meta)
        elif mode == "train":
            self.meta_selected = self._epoch_rng.sample(self.meta, self.num_per_epoch)
        else:
            self.meta_selected = self.meta[: self.num_per_epoch]

    def __getitem__(self, idx):
        info = self.meta_selected[idx]
        noisy, sr = _read_mono(info["noisy_path"], target_sr=self.target_sample_rate)
        clean, clean_sr = _read_mono(info["clean_path"], target_sr=self.target_sample_rate)
        sample_rate = sr
        if clean_sr != sample_rate:
            raise ValueError(f"Sample-rate mismatch after loading pair: {info}")

        rng_seed = int(np.random.default_rng().integers(0, 2**32 - 1)) if self.mode == "train" else idx
        rng = np.random.default_rng(rng_seed)
        noisy, clean, orig_len = _crop_or_pad_pair(
            noisy, clean, self.wav_len, sample_rate, self.random_start, rng
        )

        if self.normalize:
            noisy, clean = _normalize_pair(noisy, clean)
        item_info = {
            "id": info["id"],
            "sample_rate": int(sample_rate),
            "length": int(orig_len),
            "noisy_path": info["noisy_path"],
            "clean_path": info["clean_path"],
        }
        return noisy.astype(np.float32).squeeze(), clean.astype(np.float32).squeeze(), item_info

    def __len__(self):
        return len(self.meta_selected)


class CleanSpeechDataset(data.Dataset):
    """PyTorch Dataset for clean-only speech training."""

    def __init__(
        self,
        speech_csvs=None,
        clean_json=None,
        wav_len=None,
        num_per_epoch=0,
        random_start=False,
        target_sample_rate=None,
        mode="train",
        normalize=True,
        seed=0,
        dnsmos_threshold=3.0,
    ):
        if mode not in ["train", "valid", "validation", "test"]:
            raise ValueError(f"Unsupported mode: {mode}")
        self.speech_csvs = speech_csvs or []
        self.clean_json = clean_json
        self.wav_len = wav_len
        self.num_per_epoch = int(num_per_epoch)
        self.random_start = bool(random_start)
        self.target_sample_rate = target_sample_rate
        self.mode = "valid" if mode == "validation" else mode
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.dnsmos_threshold = float(dnsmos_threshold)
        self._epoch_rng = random.Random(self.seed)

        self.meta = self._load_meta()
        if not self.meta:
            raise ValueError("No clean speech examples found.")
        self.sample_data_per_epoch(self.mode)

    def _load_meta(self):
        meta = []
        if self.clean_json:
            for idx, clean_path in enumerate(_load_json_list(self.clean_json)):
                meta.append({"id": f"item_{idx:06d}", "clean_path": clean_path, "sample_rate": None})

        for csv_path in self.speech_csvs:
            df = pd.read_csv(csv_path).dropna(subset=["uid", "sample_rate", "filepath"])
            for row in df.itertuples():
                if hasattr(row, "dnsmos_ovrl") and (
                    row.dnsmos_ovrl < self.dnsmos_threshold
                    or row.dnsmos_sig < self.dnsmos_threshold
                    or row.dnsmos_bak < self.dnsmos_threshold
                    or row.dnsmos_p808 < self.dnsmos_threshold
                ):
                    continue
                if self.target_sample_rate is None or int(row.sample_rate) >= int(self.target_sample_rate):
                    meta.append(
                        {
                            "id": str(row.uid),
                            "clean_path": row.filepath,
                            "sample_rate": int(row.sample_rate),
                        }
                    )
        return meta

    def sample_data_per_epoch(self, mode=None):
        mode = mode or self.mode
        if self.num_per_epoch <= 0 or self.num_per_epoch >= len(self.meta):
            self.meta_selected = list(self.meta)
        elif mode == "train":
            self.meta_selected = self._epoch_rng.sample(self.meta, self.num_per_epoch)
        else:
            self.meta_selected = self.meta[: self.num_per_epoch]

    def __getitem__(self, idx):
        info = self.meta_selected[idx]
        sample_rate = self.target_sample_rate or info.get("sample_rate")
        clean, sr = _read_mono(info["clean_path"], target_sr=sample_rate)
        if self.mode == "train":
            rng_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        else:
            rng_seed = _stable_seed(self.seed, self.mode, idx, info["id"])
        rng = np.random.default_rng(rng_seed)
        clean, orig_len = _crop_or_pad(clean, self.wav_len, sr, self.random_start, rng)
        if self.normalize:
            clean = clean * (0.9 / (np.max(np.abs(clean)) + 1e-12))

        item_info = {
            "id": info["id"],
            "sample_rate": int(sr),
            "length": int(orig_len),
            "clean_path": info["clean_path"],
        }
        return clean.astype(np.float32), item_info

    def __len__(self):
        return len(self.meta_selected)
