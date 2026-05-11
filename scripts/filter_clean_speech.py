import argparse
import json
from pathlib import Path

from scripts.quality_utils import (
    QualityFilterConfig,
    filter_clean_paths,
    load_json_list,
    load_quality_scores,
    load_whitelist_patterns,
    quality_filter_enabled,
    write_json_list,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a clean-speech JSON filelist using precomputed quality scores."
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Input clean speech JSON list.")
    parser.add_argument("--output-json", type=Path, required=True, help="Filtered output JSON list.")
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Quality-score CSV, JSONL, JSON list, or JSON object.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional manifest containing filter summary and rejected paths.",
    )
    parser.add_argument(
        "--use-dnsmos-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DNSMOS filtering. Defaults to disabled.",
    )
    parser.add_argument("--dnsmos-threshold", type=float, default=3.0)
    parser.add_argument(
        "--dnsmos-fields",
        nargs="+",
        default=["ovrl", "sig", "bak", "p808"],
        choices=["ovrl", "sig", "bak", "p808"],
        help="DNSMOS dimensions that must meet the threshold.",
    )
    parser.add_argument(
        "--vqscore-threshold",
        type=float,
        default=None,
        help="Optional VQScore threshold. The Rethinking USE paper used 0.65 as a balanced default.",
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=None,
        help="Optional JSON/text list of path or source substrings to keep regardless of scores.",
    )
    parser.add_argument(
        "--missing-scores",
        choices=["fail", "keep", "drop"],
        default="fail",
        help="How to handle clean paths missing from the score file.",
    )
    args = parser.parse_args()

    config = QualityFilterConfig(
        use_dnsmos=args.use_dnsmos_filter,
        dnsmos_threshold=args.dnsmos_threshold,
        dnsmos_fields=tuple(args.dnsmos_fields),
        vqscore_threshold=args.vqscore_threshold,
        whitelist_patterns=load_whitelist_patterns(args.whitelist),
        missing_scores=args.missing_scores,
    )
    if not quality_filter_enabled(config):
        raise ValueError("Enable at least one filter: --use-dnsmos-filter or --vqscore-threshold.")

    clean_paths = load_json_list(args.input_json)
    scores = load_quality_scores(args.scores)
    result = filter_clean_paths(clean_paths, scores, config)

    write_json_list(args.output_json, result.kept_paths)
    if args.manifest_json:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(
            json.dumps(
                {
                    "config": {
                        "use_dnsmos_filter": config.use_dnsmos,
                        "dnsmos_threshold": config.dnsmos_threshold,
                        "dnsmos_fields": list(config.dnsmos_fields),
                        "vqscore_threshold": config.vqscore_threshold,
                        "whitelist_patterns": config.whitelist_patterns,
                        "missing_scores": config.missing_scores,
                    },
                    "summary": result.summary(),
                    "rejected": result.rejected,
                    "whitelisted": result.whitelisted,
                    "missing": result.missing,
                },
                indent=2,
            )
            + "\n"
        )

    print(json.dumps(result.summary(), indent=2))
    print(f"Wrote {args.output_json}")
    if args.manifest_json:
        print(f"Wrote {args.manifest_json}")


if __name__ == "__main__":
    main()
