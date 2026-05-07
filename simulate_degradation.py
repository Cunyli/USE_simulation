import os
import json
import random
import torch
import torch.utils.data
import librosa
import numpy as np
import scipy
from simulate_degradation_utils import *
import yaml


def random_select_and_order(cfg, seed=None):
    '''
    Randomly select and order the degradation configurations.
    '''
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    degradation_cfg = cfg["degradation_cfg"]
    degrad_configs = {
        "snr": random.choice(degradation_cfg["snr"]),
        "bandwidth_sr": random.choice(degradation_cfg["bandwidth_sr"]),
        "bandwidth_type": random.choice(degradation_cfg["bandwidth_type"]),
        "lowpass_type": random.choice(degradation_cfg["lowpass_type"]),
        "lowpass_order": random.choice(degradation_cfg["lowpass_order"]),
        "clipping_min": np.random.uniform(*degradation_cfg["clipping_min"]),
        "clipping_max": np.random.uniform(*degradation_cfg["clipping_max"]),
        "packet_duration": random.choice(degradation_cfg["packet_duration"]),
        "packet_loss_rate": np.random.uniform(*degradation_cfg["packet_loss_rate"]),
        "max_continuous_packet_loss": degradation_cfg.get("max_continuous_packet_loss", 5),
        "codec": random.choice(
            degradation_cfg.get(
                "codec",
                [{"format": "mp3", "encoder": "None", "qscale": 4}],
            )
        ),
        "wind_snr": random.choice(degradation_cfg.get("wind_snr", [0, 5, 10])),
        "mic_eq_low_shelf_gain_db": np.random.uniform(*degradation_cfg.get("mic_eq_low_shelf_gain_db", [-6.0, 6.0])),
        "mic_eq_high_shelf_gain_db": np.random.uniform(*degradation_cfg.get("mic_eq_high_shelf_gain_db", [-8.0, 4.0])),
        "mic_eq_peak_gain_db": np.random.uniform(*degradation_cfg.get("mic_eq_peak_gain_db", [-6.0, 6.0])),
        "mic_eq_peak_freq": random.choice(degradation_cfg.get("mic_eq_peak_freq", [500, 1000, 2000, 3500])),
        "mic_eq_peak_width_oct": random.choice(degradation_cfg.get("mic_eq_peak_width_oct", [0.5, 1.0, 1.5])),
        "agc_threshold_db": np.random.uniform(*degradation_cfg.get("agc_threshold_db", [-30.0, -12.0])),
        "agc_ratio": np.random.uniform(*degradation_cfg.get("agc_ratio", [1.5, 4.0])),
        "agc_attack_ms": np.random.uniform(*degradation_cfg.get("agc_attack_ms", [5.0, 50.0])),
        "agc_release_ms": np.random.uniform(*degradation_cfg.get("agc_release_ms", [50.0, 300.0])),
        "agc_makeup_gain_db": np.random.uniform(*degradation_cfg.get("agc_makeup_gain_db", [-3.0, 6.0])),
        "adc_bit_depth": random.choice(degradation_cfg.get("adc_bit_depth", [8, 12, 16])),
        "adc_lowpass_freq": random.choice(degradation_cfg.get("adc_lowpass_freq", [3400, 6000, 7600])),
        "adc_highpass_freq": random.choice(degradation_cfg.get("adc_highpass_freq", [20, 80, 120])),
    }

    degrad_order_map = {
        "reverb": 1,
        "noise": 2,
        "wind_noise": 3,
        "mic_eq": 4,
        "agc": 5,
        "adc": 6,
        "clipping": 7,
        "bandwidth": 8,
        "codec": 9,
        "packet_loss": 10,
    }
    degrad_types = list(degrad_order_map)
    degrad_probs = {
        "noise": degradation_cfg.get("noise_prob", 1.0),
        "reverb": degradation_cfg.get("reverb_prob", 0.7),
        "wind_noise": degradation_cfg.get("wind_noise_prob", 0.0),
        "mic_eq": degradation_cfg.get("mic_eq_prob", 0.0),
        "agc": degradation_cfg.get("agc_prob", 0.0),
        "adc": degradation_cfg.get("adc_prob", 0.0),
        "clipping": degradation_cfg.get("clipping_prob", 0.5),
        "bandwidth": degradation_cfg.get("bandwidth_prob", 0.7),
        "codec": degradation_cfg.get("codec_prob", 0.0),
        "packet_loss": degradation_cfg.get("packet_loss_prob", 0.0),
    }

    selected_degradations = [x for x in degrad_types if random.random() < degrad_probs[x]]

    if len(selected_degradations) == 0:
        probs = np.array([degrad_probs[x] for x in degrad_types], dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(degrad_types)) / len(degrad_types)
        selected_degradations = np.random.choice(degrad_types, p=probs, size=1).tolist()

    max_degradations = degradation_cfg.get("max_degradations", None)
    if max_degradations is not None and len(selected_degradations) > max_degradations:
        selected_degradations = random.sample(selected_degradations, max_degradations)

    selected_degradations = sorted(selected_degradations, key=lambda x: degrad_order_map[x])

    return degrad_configs, selected_degradations

# degrad_configs = {"snr": 20, "bandwidth_freq": 1000, ...}, selected_degradations = ["noise", "reverb", ...]


def apply_degradation(cfg, speech_sample, noise_sample, rir_sample, degradation_configs,
                      selected_degradations: list, seed: int = None):
    '''Apply degradations to speech sample
    Args:
        speech_sample: The original speech signal. (np.ndarray): a single speech sample (1, T)
        noise_sample: The noise signal to be added. a single noise sample (1, T)
        rir_sample: The room impulse response for reverberation. a single room impulse response (RIR) (1, T)
        degradation_configs: The configuration parameters for each degradation type.
        selected_degradations: The list of selected degradation types to apply.
    '''
    return apply_degradation_with_wind(
        cfg,
        speech_sample,
        noise_sample,
        rir_sample,
        None,
        degradation_configs,
        selected_degradations,
        seed=seed,
    )


def apply_degradation_with_wind(
    cfg,
    speech_sample,
    noise_sample,
    rir_sample,
    wind_noise_sample,
    degradation_configs,
    selected_degradations: list,
    seed: int = None,
):
    assert type(selected_degradations) == list, "selected_degradations must be a list."
    assert len(selected_degradations) >= 1, "At least one degradation type must be selected."

    sr = cfg["stft_cfg"]["sampling_rate"]
    rng = np.random.default_rng(seed=seed)
    degraded_sample = speech_sample.copy()

    for degrad in selected_degradations:
        if degrad == "reverb":
            degraded_sample = add_reverberation(degraded_sample, rir_sample)
        elif degrad == "noise":
            degraded_sample, _ = mix_noise(degraded_sample, noise_sample, degradation_configs["snr"], rng)
        elif degrad == "wind_noise":
            if wind_noise_sample is None:
                raise ValueError("wind_noise_sample is required when wind_noise is selected.")
            degraded_sample, _ = mix_noise(degraded_sample, wind_noise_sample, degradation_configs["wind_snr"], rng)
        elif degrad == "mic_eq":
            degraded_sample = microphone_frequency_response(
                degraded_sample,
                sr,
                degradation_configs["mic_eq_low_shelf_gain_db"],
                degradation_configs["mic_eq_high_shelf_gain_db"],
                degradation_configs["mic_eq_peak_gain_db"],
                degradation_configs["mic_eq_peak_freq"],
                degradation_configs["mic_eq_peak_width_oct"],
            )
        elif degrad == "agc":
            degraded_sample = automatic_gain_control(
                degraded_sample,
                sr,
                degradation_configs["agc_threshold_db"],
                degradation_configs["agc_ratio"],
                degradation_configs["agc_attack_ms"],
                degradation_configs["agc_release_ms"],
                degradation_configs["agc_makeup_gain_db"],
            )
        elif degrad == "adc":
            degraded_sample = adc_effect(
                degraded_sample,
                sr,
                degradation_configs["adc_bit_depth"],
                degradation_configs["adc_lowpass_freq"],
                degradation_configs["adc_highpass_freq"],
            )
        elif degrad == "clipping":
            degraded_sample = clipping(degraded_sample, degradation_configs["clipping_min"], degradation_configs["clipping_max"])
        elif degrad == "bandwidth":
            degraded_sample = bandwidth_limitation(
                degraded_sample,
                sr,
                degradation_configs["bandwidth_sr"],
                degradation_configs["bandwidth_type"],
                degradation_configs["lowpass_type"],
                degradation_configs["lowpass_order"],
            )
        elif degrad == "codec":
            degraded_sample = codec_compression(degraded_sample, sr, **degradation_configs["codec"])
        elif degrad == "packet_loss":
            degraded_sample = packet_loss(
                degraded_sample,
                sr,
                degradation_configs["packet_duration"],
                degradation_configs["packet_loss_rate"],
                degradation_configs["max_continuous_packet_loss"],
                rng,
            )
        else:
            raise ValueError(f"Unsupported degradation type: {degrad}")
    
    return speech_sample, degraded_sample
