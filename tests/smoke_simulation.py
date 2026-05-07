import numpy as np

from simulate_degradation import apply_degradation_with_wind


def main() -> None:
    sr = 16000
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)

    speech = (0.2 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1).astype(np.float32)
    noise = np.random.default_rng(1234).normal(0.0, 0.05, speech.shape).astype(np.float32)
    wind_noise = np.random.default_rng(5678).normal(0.0, 0.05, speech.shape).astype(np.float32)
    rir = np.zeros((1, 256), dtype=np.float32)
    rir[0, 0] = 1.0
    rir[0, 80] = 0.3

    cfg = {
        "stft_cfg": {"sampling_rate": sr},
    }
    degradation_configs = {
        "snr": 5,
        "bandwidth_sr": 8000,
        "bandwidth_type": "kaiser_fast",
        "lowpass_type": "butter",
        "lowpass_order": 4,
        "clipping_min": 0.01,
        "clipping_max": 0.99,
        "packet_duration": 20,
        "packet_loss_rate": 0.05,
        "max_continuous_packet_loss": 5,
        "codec": {"format": "ogg", "encoder": "vorbis", "qscale": 5},
        "wind_snr": 5,
        "mic_eq_low_shelf_gain_db": -3.0,
        "mic_eq_high_shelf_gain_db": -4.0,
        "mic_eq_peak_gain_db": 2.0,
        "mic_eq_peak_freq": 2000,
        "mic_eq_peak_width_oct": 1.0,
        "agc_threshold_db": -24.0,
        "agc_ratio": 2.5,
        "agc_attack_ms": 10.0,
        "agc_release_ms": 120.0,
        "agc_makeup_gain_db": 2.0,
        "adc_bit_depth": 12,
        "adc_lowpass_freq": 6000,
        "adc_highpass_freq": 80,
    }

    clean, degraded = apply_degradation_with_wind(
        cfg,
        speech,
        noise,
        rir,
        wind_noise,
        degradation_configs,
        [
            "reverb",
            "noise",
            "wind_noise",
            "mic_eq",
            "agc",
            "adc",
            "clipping",
            "bandwidth",
            "codec",
            "packet_loss",
        ],
        seed=1234,
    )

    assert clean.shape == degraded.shape == speech.shape
    assert np.isfinite(degraded).all()
    assert not np.allclose(clean, degraded)
    print(f"smoke ok: shape={degraded.shape}, rms={np.sqrt(np.mean(degraded ** 2)):.6f}")


if __name__ == "__main__":
    main()
