from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scripts.quality_filtering import load_json_list, load_quality_score_records


METRICS = ("ovrl", "sig", "bak", "p808", "vqscore")
DNSSCORE_METRICS = ("ovrl", "sig", "bak", "p808")


def merge_score_files(score_paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for score_path in score_paths:
        for record in load_quality_score_records(score_path):
            item_path = str(record["path"])
            target = merged.setdefault(item_path, {"path": item_path, "source": None, "scores": {}})
            if record.get("source") and not target.get("source"):
                target["source"] = record["source"]
            target["scores"].update(record.get("scores", {}))
    return merged


def summarize_metric(values: list[float], threshold: float | None) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "missing_count": 0,
        }

    summary: dict[str, Any] = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p05": percentile(values, 5),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "max": max(values),
    }
    if threshold is not None:
        below_count = sum(value < threshold for value in values)
        summary[f"below_{threshold:g}_count"] = below_count
        summary[f"below_{threshold:g}_rate"] = below_count / len(values)
    return summary


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def path_key(path: str) -> str:
    return str(Path(path))


def build_summary(
    records_by_path: dict[str, dict[str, Any]],
    expected_paths: list[str] | None,
    dnsmos_threshold: float,
    vqscore_threshold: float,
) -> dict[str, Any]:
    if expected_paths is None:
        expected_set = set(records_by_path)
    else:
        expected_set = {path_key(path) for path in expected_paths}

    scored_set = {path_key(path) for path in records_by_path}
    missing_paths = sorted(expected_set - scored_set)

    metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    missing_metric_counts: dict[str, int] = {metric: 0 for metric in METRICS}
    dnsmos_any_below = 0
    complete_dnsmos_count = 0

    for path, record in records_by_path.items():
        if path_key(path) not in expected_set:
            continue
        scores = record.get("scores", {})
        for metric in METRICS:
            value = scores.get(metric)
            if value is None:
                missing_metric_counts[metric] += 1
            else:
                metric_values[metric].append(float(value))

        dnsmos_values = [scores.get(metric) for metric in DNSSCORE_METRICS]
        if all(value is not None for value in dnsmos_values):
            complete_dnsmos_count += 1
            if any(float(value) < dnsmos_threshold for value in dnsmos_values):
                dnsmos_any_below += 1

    metric_thresholds = {
        "ovrl": dnsmos_threshold,
        "sig": dnsmos_threshold,
        "bak": dnsmos_threshold,
        "p808": dnsmos_threshold,
        "vqscore": vqscore_threshold,
    }
    metrics = {}
    for metric in METRICS:
        metric_summary = summarize_metric(metric_values[metric], metric_thresholds[metric])
        metric_summary["missing_count"] = missing_metric_counts[metric]
        metrics[metric] = metric_summary

    return {
        "expected_count": len(expected_set),
        "scored_count": len(scored_set & expected_set),
        "missing_score_count": len(missing_paths),
        "missing_score_paths": missing_paths,
        "dnsmos_any_below_threshold_count": dnsmos_any_below,
        "dnsmos_any_below_threshold_rate": dnsmos_any_below / complete_dnsmos_count
        if complete_dnsmos_count
        else None,
        "metrics": metrics,
    }


def worst_records(
    records_by_path: dict[str, dict[str, Any]],
    metric: str,
    expected_paths: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    expected_set = {path_key(path) for path in expected_paths} if expected_paths is not None else None
    rows = []
    for path, record in records_by_path.items():
        if expected_set is not None and path_key(path) not in expected_set:
            continue
        scores = record.get("scores", {})
        if metric not in scores:
            continue
        rows.append(
            {
                "path": path,
                "source": record.get("source"),
                "metric": metric,
                "value": scores[metric],
                "scores": scores,
            }
        )
    rows.sort(key=lambda item: item["value"])
    return rows[:limit]


def print_metric_table(summary: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "expected_count": summary["expected_count"],
                "scored_count": summary["scored_count"],
                "missing_score_count": summary["missing_score_count"],
                "dnsmos_any_below_threshold_count": summary["dnsmos_any_below_threshold_count"],
                "dnsmos_any_below_threshold_rate": summary["dnsmos_any_below_threshold_rate"],
            },
            indent=2,
        )
    )
    for metric, metric_summary in summary["metrics"].items():
        if metric_summary["count"] == 0:
            print(f"{metric}: count=0 missing={metric_summary['missing_count']}")
            continue
        print(
            f"{metric}: count={metric_summary['count']} "
            f"mean={metric_summary['mean']:.4f} "
            f"median={metric_summary['median']:.4f} "
            f"p05={metric_summary['p05']:.4f} "
            f"p95={metric_summary['p95']:.4f} "
            f"min={metric_summary['min']:.4f} "
            f"max={metric_summary['max']:.4f} "
            f"missing={metric_summary['missing_count']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize DNSMOS/VQScore distributions without filtering. "
            "Use this for protected pathological fine-tuning data."
        )
    )
    parser.add_argument("--scores", type=Path, nargs="+", required=True, help="One or more score files.")
    parser.add_argument("--clean-json", type=Path, default=None, help="Optional expected clean speech JSON list.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional full summary JSON.")
    parser.add_argument("--worst-json", type=Path, default=None, help="Optional lowest-score sample list JSON.")
    parser.add_argument("--worst-n", type=int, default=20)
    parser.add_argument("--worst-metric", choices=METRICS, default="vqscore")
    parser.add_argument("--dnsmos-threshold", type=float, default=3.0)
    parser.add_argument("--vqscore-threshold", type=float, default=0.65)
    args = parser.parse_args()

    records_by_path = merge_score_files(args.scores)
    expected_paths = load_json_list(args.clean_json) if args.clean_json is not None else None
    summary = build_summary(records_by_path, expected_paths, args.dnsmos_threshold, args.vqscore_threshold)
    worst = worst_records(records_by_path, args.worst_metric, expected_paths, args.worst_n)

    print_metric_table(summary)
    if worst:
        print(f"Lowest {len(worst)} by {args.worst_metric}:")
        for item in worst:
            print(f"{item['value']:.4f}\t{item['path']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output_json}")
    if args.worst_json:
        args.worst_json.parent.mkdir(parents=True, exist_ok=True)
        args.worst_json.write_text(json.dumps(worst, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.worst_json}")


if __name__ == "__main__":
    main()
