import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PATH_FIELDS = ("path", "audio_path", "filepath", "file", "wav", "clean_path", "utt")
SOURCE_FIELDS = ("source", "dataset", "corpus", "collection")
DNSSCORE_ALIASES = {
    "ovrl": ("ovrl", "overall", "dnsmos_ovrl", "dns_ovrl"),
    "sig": ("sig", "signal", "dnsmos_sig", "dns_sig"),
    "bak": ("bak", "background", "dnsmos_bak", "dns_bak"),
    "p808": ("p808", "p_808", "p.808", "dnsmos_p808", "dnsmos_p_808"),
}
VQSCORE_ALIASES = ("vqscore", "vq_score", "vq", "voice_quality_score")


@dataclass
class QualityFilterConfig:
    use_dnsmos: bool = False
    dnsmos_threshold: float = 3.0
    dnsmos_fields: tuple[str, ...] = ("ovrl", "sig", "bak", "p808")
    vqscore_threshold: float | None = None
    whitelist_patterns: list[str] = field(default_factory=list)
    missing_scores: str = "fail"


@dataclass
class QualityFilterResult:
    kept_paths: list[str]
    rejected: list[dict[str, Any]]
    whitelisted: list[str]
    missing: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "input_count": len(self.kept_paths) + len(self.rejected),
            "kept_count": len(self.kept_paths),
            "rejected_count": len(self.rejected),
            "whitelisted_count": len(self.whitelisted),
            "missing_score_count": len(self.missing),
            "reject_reasons": _count_reasons(self.rejected),
        }


def load_json_list(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return [str(item) for item in data]


def write_json_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n")


def load_whitelist_patterns(path: Path | None) -> list[str]:
    if path is None:
        return []
    if path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"{path} must contain a JSON list of path/source patterns.")
        return [str(item) for item in data]
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_quality_scores(path: Path) -> dict[str, dict[str, Any]]:
    records = _read_score_records(path)
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        item_path = _record_path(record)
        if item_path is None:
            continue
        normalized = _normalize_record(record)
        for key in _path_keys(item_path):
            index.setdefault(key, normalized)
    return index


def load_quality_score_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for record in _read_score_records(path):
        item_path = _record_path(record)
        if item_path is not None:
            records.append(_normalize_record(record))
    return records


def filter_clean_paths(
    clean_paths: list[str],
    score_index: dict[str, dict[str, Any]],
    config: QualityFilterConfig,
) -> QualityFilterResult:
    if config.missing_scores not in {"fail", "keep", "drop"}:
        raise ValueError("missing_scores must be one of: fail, keep, drop.")

    kept: list[str] = []
    rejected: list[dict[str, Any]] = []
    whitelisted: list[str] = []
    missing: list[str] = []

    for clean_path in clean_paths:
        record = _lookup_score(score_index, clean_path)
        if _is_whitelisted(clean_path, record, config.whitelist_patterns):
            kept.append(clean_path)
            whitelisted.append(clean_path)
            continue

        if record is None:
            missing.append(clean_path)
            if config.missing_scores == "fail":
                raise KeyError(f"No quality score record found for {clean_path}")
            if config.missing_scores == "keep":
                kept.append(clean_path)
            else:
                rejected.append({"path": clean_path, "reason": "missing_scores"})
            continue

        reject_reason = _reject_reason(record, config)
        if reject_reason is None:
            kept.append(clean_path)
        else:
            rejected.append({"path": clean_path, "reason": reject_reason, "scores": record.get("scores", {})})

    return QualityFilterResult(kept, rejected, whitelisted, missing)


def quality_filter_enabled(config: QualityFilterConfig) -> bool:
    return config.use_dnsmos or config.vqscore_threshold is not None


def _read_score_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise TypeError(f"{path} JSON list must contain score objects.")
        return data
    if isinstance(data, dict):
        records = []
        for item_path, value in data.items():
            if isinstance(value, dict):
                records.append({"path": item_path, **value})
            else:
                records.append({"path": item_path, "vqscore": value})
        return records
    raise TypeError(f"{path} must be CSV, JSONL, a JSON list, or a JSON object.")


def _record_path(record: dict[str, Any]) -> str | None:
    normalized = {_canonical_key(key): value for key, value in record.items()}
    for field in PATH_FIELDS:
        value = normalized.get(_canonical_key(field))
        if value:
            return str(value)
    return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = _flatten_record(record)
    scores: dict[str, float] = {}

    for score_name, aliases in DNSSCORE_ALIASES.items():
        value = _first_float(normalized, aliases)
        if value is not None:
            scores[score_name] = value

    vqscore = _first_float(normalized, VQSCORE_ALIASES)
    if vqscore is not None:
        scores["vqscore"] = vqscore

    source = None
    for field_name in SOURCE_FIELDS:
        value = normalized.get(_canonical_key(field_name))
        if value:
            source = str(value)
            break

    return {
        "path": _record_path(record),
        "source": source,
        "scores": scores,
    }


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        normalized[_canonical_key(key)] = value
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                normalized[_canonical_key(nested_key)] = nested_value
                normalized[_canonical_key(f"{key}_{nested_key}")] = nested_value
    return normalized


def _first_float(record: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = record.get(_canonical_key(alias))
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _reject_reason(record: dict[str, Any], config: QualityFilterConfig) -> str | None:
    scores = record.get("scores", {})
    if config.use_dnsmos:
        for field_name in config.dnsmos_fields:
            value = scores.get(field_name)
            if value is None:
                return f"missing_dnsmos_{field_name}"
            if value < config.dnsmos_threshold:
                return f"low_dnsmos_{field_name}"

    if config.vqscore_threshold is not None:
        value = scores.get("vqscore")
        if value is None:
            return "missing_vqscore"
        if value < config.vqscore_threshold:
            return "low_vqscore"

    return None


def _lookup_score(score_index: dict[str, dict[str, Any]], path: str) -> dict[str, Any] | None:
    for key in _path_keys(path):
        if key in score_index:
            return score_index[key]
    return None


def _path_keys(path: str) -> list[str]:
    raw = str(path)
    keys = [raw]
    try:
        p = Path(raw)
        keys.extend([str(p), p.name])
        if p.is_absolute():
            keys.append(str(p.resolve()))
    except OSError:
        pass
    return list(dict.fromkeys(keys))


def _is_whitelisted(path: str, record: dict[str, Any] | None, patterns: list[str]) -> bool:
    if not patterns:
        return False
    haystacks = [path, Path(path).name]
    if record:
        if record.get("path"):
            haystacks.append(str(record["path"]))
        if record.get("source"):
            haystacks.append(str(record["source"]))
    haystacks = [item.lower() for item in haystacks]
    return any(pattern.lower() in item for pattern in patterns for item in haystacks)


def _canonical_key(value: str) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _count_reasons(rejected: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts
