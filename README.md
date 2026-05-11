# USE Simulation

Standalone simulation utilities copied from `SEMambapp-Interspeech`.

## Files

- `simulate_degradation.py`: selects and applies degradation chains.
- `simulate_degradation_utils.py`: noise, reverberation, bandwidth, clipping, packet-loss, and codec utilities.
- `scripts/build_val_degraded.py`: generates offline degraded validation audio from JSON filelists.
- `scripts/score_clean_speech_quality.py`: scores clean-speech filelists with DNSMOS and VQScore, then writes one quality JSON.
- `scripts/filter_clean_speech.py`: filters clean-speech JSON filelists using precomputed DNSMOS/VQScore metadata.
- `scripts/prepare_local_smoke_data.py`: creates synthetic local-only fixtures for testing the pipeline without server data.
- `scripts/quality_*.py`: internal helpers used by the scoring and filtering scripts.
- `config.yaml`: broad degradation and STFT defaults copied from the original project.
- `configs/phone_room_22050.yaml`: Priority 1 AVQI profile, noise + RIR only at 22050 Hz.
- `data/*.json`: copied filelists from the original project.

## Smoke Test

```bash
conda run -n use_simulation python -m tests.smoke_simulation
```

The codec distortion uses `torchaudio.io.AudioEffector`, so the conda environment also needs an FFmpeg version supported by the installed torchaudio build. For `torch==2.2.2` / `torchaudio==2.2.2`, this environment is currently using FFmpeg 4.3.2.

## Priority 1: Phone-In-Room

`configs/phone_room_22050.yaml` is the first AVQI-oriented profile. It keeps
only recording-environment effects:

- additive room noise
- room impulse response

It disables codec, packet loss, wind noise, bandwidth limitation, clipping,
microphone EQ, AGC, and ADC effects. Inputs are resampled to 22050 Hz and
outputs are written at 22050 Hz.

For local pipeline checks without server data, generate clearly marked synthetic
fixtures:

```bash
conda run -n use_simulation python -m scripts.prepare_local_smoke_data
```

Then run the mini sample:

```bash
conda run -n use_simulation python -m scripts.build_val_degraded \
  --config configs/phone_room_22050.yaml \
  --clean-json data/local_smoke/clean.json \
  --noise-json data/local_smoke/noise.json \
  --rir-json data/local_smoke/rir.json \
  --output-root data/local_smoke/phone_room_22050 \
  --output-json data/local_smoke/phone_room_22050.json \
  --manifest-json data/local_smoke/phone_room_22050_manifest.json \
  --force
```

`data/local_smoke/` is ignored by git and should not be treated as formal
experiment data.

## Generate Validation Audio

The copied filelists currently point to `/scratch/elec/t412-speechcom/...`, so real generation needs that filesystem to be mounted or the JSON paths updated.

## Clean-Target Quality Filtering

This repository can score clean targets with DNSMOS and VQScore, then write one
quality JSON that downstream filtering and simulation can reuse. DNSMOS follows
the URGENT-style ESPnet `DNSMOS_local` workflow and uses Microsoft's ONNX
models. VQScore uses the official `JasonSWFu/VQscore` quality-estimation model.
Downloaded assets live under `eval/quality/DNSMOS` and `eval/quality/VQscore`.

DNSMOS scoring requires optional dependencies that are not needed for normal
simulation:

```bash
pip install espnet==202412 onnxruntime
```

Run both DNSMOS and VQScore on a clean JSON filelist and write one merged
quality file:

```bash
conda run -n use_simulation python -m scripts.score_clean_speech_quality \
  --input data/train_speech.json \
  --input-format json \
  --output-json data/train_clean_quality.json \
  --dnsmos-device cpu \
  --vqscore-device auto
```

By default, the VQScore wrapper uses the official quality-estimation config and
checkpoint:

```text
config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml
exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/checkpoint-dnsmos_ovr_CC=0.835.pkl
```

Override them with `--config` and `--checkpoint` if you want to use a different
VQScore model. Add `--no-download` to require all VQScore files to already exist
locally. The official VQScore quality-estimation checkpoint is about 11 MB. The
two DNSMOS ONNX files are much smaller, about 1.1 MB and 0.2 MB.

Downstream filtering and simulation can consume `data/train_clean_quality.json`
directly. The filtering code also accepts CSV, JSONL, a JSON list of objects, or
a JSON object keyed by audio path when you need to use older score files.

DNSMOS filtering is optional and defaults to off:

```bash
conda run -n use_simulation python -m scripts.filter_clean_speech \
  --input-json data/train_speech.json \
  --output-json data/train_speech.filtered.json \
  --scores data/train_clean_quality.json \
  --use-dnsmos-filter \
  --dnsmos-threshold 3.0 \
  --vqscore-threshold 0.65 \
  --whitelist data/quality_whitelist.txt \
  --manifest-json data/train_speech.quality_manifest.json
```

The same logic can be applied while generating validation audio:

```bash
conda run -n use_simulation python -m scripts.build_val_degraded \
  --config configs/phone_room_22050.yaml \
  --clean-json data/val_clean.json \
  --noise-json data/train_noise.json \
  --rir-json data/train_rir.json \
  --quality-scores data/train_clean_quality.json \
  --use-dnsmos-filter \
  --vqscore-threshold 0.65 \
  --quality-whitelist data/quality_whitelist.txt \
  --quality-manifest-json data/val_clean.quality_manifest.json
```

The intended policy is layered: use DNSMOS as an optional coarse floor
(`OVRL/SIG/BAK/P.808 >= 3.0`), then use VQScore as the stricter clean-target
quality check. Keep the DNSMOS switch under explicit user control because
DNSMOS can down-rank atypical but useful speech.

Do not blindly remove EARS, whisper, highly emotional speech, extreme-pitch
speech, dysphonic/pathological voice, or other atypical-but-clean recordings.
Put those sources or path patterns in a whitelist and audit samples manually
before discarding them. A whitelist file can be JSON or plain text, for example:

```text
EARS
whisper
pathological
```

## Distortions

The Level 2 URGENT-style simulation covers:

- additive noise
- reverberation
- clipping
- bandwidth limitation
- codec distortion
- packet loss
- wind noise

`noise` and `wind noise` are sampled from separate file lists. `wind_noise` expects real or pre-generated wind-noise wav files, for example the `wind_noise.scp` produced by URGENT's `simulation/simulate_wind_noise.py` can be converted to a JSON list and passed as `--wind-noise-json`.
