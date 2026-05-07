import json
import tempfile
from pathlib import Path

from scripts.quality_filtering import (
    QualityFilterConfig,
    filter_clean_paths,
    load_quality_scores,
)


def main() -> None:
    clean_paths = [
        "/data/commonvoice/low.wav",
        "/data/vctk/good.wav",
        "/data/EARS/whisper.wav",
    ]
    scores = {
        clean_paths[0]: {"ovrl": 2.9, "sig": 3.2, "bak": 3.2, "p808": 3.1, "vqscore": 0.70},
        clean_paths[1]: {"ovrl": 3.4, "sig": 3.5, "bak": 3.3, "p808": 3.2, "vqscore": 0.66},
        clean_paths[2]: {"ovrl": 2.4, "sig": 2.8, "bak": 3.1, "p808": 2.9, "vqscore": 0.40},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / "scores.json"
        score_path.write_text(json.dumps(scores) + "\n")
        score_index = load_quality_scores(score_path)

    dnsmos_off = filter_clean_paths(
        clean_paths,
        score_index,
        QualityFilterConfig(use_dnsmos=False, vqscore_threshold=0.65),
    )
    assert dnsmos_off.kept_paths == clean_paths[:2]

    dnsmos_on = filter_clean_paths(
        clean_paths,
        score_index,
        QualityFilterConfig(
            use_dnsmos=True,
            dnsmos_threshold=3.0,
            vqscore_threshold=0.65,
            whitelist_patterns=["EARS"],
        ),
    )
    assert dnsmos_on.kept_paths == clean_paths[1:]
    assert dnsmos_on.whitelisted == [clean_paths[2]]
    assert dnsmos_on.rejected[0]["reason"] == "low_dnsmos_ovrl"

    print("quality filter ok")


if __name__ == "__main__":
    main()
