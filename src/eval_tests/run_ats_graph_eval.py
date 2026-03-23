import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.agents.ats_processor_agent import build_ats_resume_evaluation_graph


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
LABEL_ORDER = ["weak", "moderate", "strong"]
LABEL_TO_TARGET_SCORE = {"weak": 0.2, "moderate": 0.5, "strong": 0.8}


@dataclass
class Sample:
    domain: str
    resume_path: Path
    jd_path: Path
    expected_label: str


@dataclass
class SampleResult:
    sample: Sample
    final_score: float
    predicted_label: str
    is_correct: bool
    skills_match_score: float
    experience_match_score: float
    role_match_score: float
    education_match_score: float
    requirement_constraint_score: float
    judge_verdict: str
    judge_confidence: float


def _normalize_label(raw_label: str) -> str:
    value = raw_label.strip().lower()
    if value == "mod":
        return "moderate"
    return value


def _infer_label_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    prefix = stem.split("_", 1)[0]
    label = _normalize_label(prefix)
    if label not in LABEL_ORDER:
        raise ValueError(f"Unsupported label in file name: {path.name}")
    return label


def _score_to_label(score: float) -> str:
    if score < 0.4:
        return "weak"
    if score < 0.7:
        return "moderate"
    return "strong"


def discover_samples() -> tuple[list[Sample], list[str]]:
    if not EXAMPLES_DIR.exists():
        raise FileNotFoundError(f"Examples directory not found: {EXAMPLES_DIR}")

    samples: list[Sample] = []
    warnings: list[str] = []

    for domain_dir in sorted([p for p in EXAMPLES_DIR.iterdir() if p.is_dir()]):
        jd_files = sorted(domain_dir.glob("test_*-jd.json"))
        resume_files = sorted(domain_dir.glob("*.txt"))

        if not jd_files:
            if resume_files:
                warnings.append(
                    f"[WARN] {domain_dir.name}: resume files exist but no JD json found."
                )
            continue

        jd_path = jd_files[0]
        if len(jd_files) > 1:
            warnings.append(
                f"[WARN] {domain_dir.name}: multiple JD files found; using {jd_path.name}."
            )

        if not resume_files:
            warnings.append(f"[WARN] {domain_dir.name}: JD exists but no resume .txt files.")
            continue

        for resume_path in resume_files:
            try:
                expected_label = _infer_label_from_filename(resume_path)
            except ValueError as err:
                warnings.append(f"[WARN] {domain_dir.name}: {err}")
                continue
            samples.append(
                Sample(
                    domain=domain_dir.name.lower(),
                    resume_path=resume_path,
                    jd_path=jd_path,
                    expected_label=expected_label,
                )
            )

    return samples, warnings


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _calc_confusion(results: list[SampleResult]) -> dict[str, dict[str, int]]:
    confusion = {label: {p: 0 for p in LABEL_ORDER} for label in LABEL_ORDER}
    for result in results:
        confusion[result.sample.expected_label][result.predicted_label] += 1
    return confusion


def _calc_classification_metrics(
    confusion: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in LABEL_ORDER:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABEL_ORDER if other != label)
        fn = sum(confusion[label][other] for other in LABEL_ORDER if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def _group_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _ranking_sanity_by_domain(results: list[SampleResult]) -> dict[str, str]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        domain_bucket = grouped.setdefault(
            result.sample.domain, {k: [] for k in LABEL_ORDER}
        )
        domain_bucket[result.sample.expected_label].append(result.final_score)

    verdicts: dict[str, str] = {}
    for domain, buckets in grouped.items():
        if not all(buckets[label] for label in LABEL_ORDER):
            verdicts[domain] = "insufficient_labels"
            continue
        weak_m = _group_mean(buckets["weak"])
        mod_m = _group_mean(buckets["moderate"])
        strong_m = _group_mean(buckets["strong"])
        verdicts[domain] = "pass" if strong_m > mod_m > weak_m else "fail"
    return verdicts


async def run_eval() -> int:
    samples, warnings = discover_samples()

    print(f"Discovered {len(samples)} samples under: {EXAMPLES_DIR}")
    for warning in warnings:
        print(warning)

    if not samples:
        print("[ERROR] No valid samples found; nothing to evaluate.")
        return 1

    graph = build_ats_resume_evaluation_graph()
    results: list[SampleResult] = []

    print("\nPer-sample results:")
    for idx, sample in enumerate(samples, start=1):
        resume_text = sample.resume_path.read_text(encoding="utf-8")
        jd_payload = json.loads(sample.jd_path.read_text(encoding="utf-8"))
        jd_text = str(jd_payload.get("description", ""))

        output = await graph.ainvoke({"resume_text": resume_text, "jd_text": jd_text})
        final_score = _safe_float(output.get("final_score"))
        predicted_label = _score_to_label(final_score)

        result = SampleResult(
            sample=sample,
            final_score=final_score,
            predicted_label=predicted_label,
            is_correct=predicted_label == sample.expected_label,
            skills_match_score=_safe_float(output.get("skills_match_score")),
            experience_match_score=_safe_float(output.get("experience_match_score")),
            role_match_score=_safe_float(output.get("role_match_score")),
            education_match_score=_safe_float(output.get("education_match_score")),
            requirement_constraint_score=_safe_float(
                output.get("requirement_constraint_score")
            ),
            judge_verdict=str(output.get("judge_verdict", "")),
            judge_confidence=_safe_float(output.get("judge_confidence")),
        )
        results.append(result)

        print(
            f"{idx:02d}. [{result.sample.domain}] {result.sample.resume_path.name} | "
            f"expected={result.sample.expected_label:<8} "
            f"predicted={result.predicted_label:<8} score={result.final_score:.3f} "
            f"correct={result.is_correct}"
        )

    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / len(results)

    confusion = _calc_confusion(results)
    metrics = _calc_classification_metrics(confusion)
    macro_f1 = _group_mean([metrics[label]["f1"] for label in LABEL_ORDER])

    mae_values = [
        abs(r.final_score - LABEL_TO_TARGET_SCORE[r.sample.expected_label]) for r in results
    ]
    mae_to_target = _group_mean(mae_values)

    by_domain: dict[str, list[float]] = {}
    by_class: dict[str, list[float]] = {label: [] for label in LABEL_ORDER}
    for r in results:
        by_domain.setdefault(r.sample.domain, []).append(r.final_score)
        by_class[r.sample.expected_label].append(r.final_score)

    ranking_sanity = _ranking_sanity_by_domain(results)

    print("\nAggregate metrics:")
    print(f"- total_samples: {len(results)}")
    print(f"- accuracy: {accuracy:.3f}")
    print(f"- macro_f1: {macro_f1:.3f}")
    print(f"- mae_to_target: {mae_to_target:.3f}")

    print("\nPer-class metrics:")
    for label in LABEL_ORDER:
        class_metrics = metrics[label]
        print(
            f"- {label:<8} precision={class_metrics['precision']:.3f} "
            f"recall={class_metrics['recall']:.3f} f1={class_metrics['f1']:.3f} "
            f"mean_score={_group_mean(by_class[label]):.3f}"
        )

    print("\nConfusion matrix (expected -> predicted counts):")
    for expected in LABEL_ORDER:
        row = " ".join(f"{pred}:{confusion[expected][pred]}" for pred in LABEL_ORDER)
        print(f"- {expected:<8} {row}")

    print("\nDomain mean scores:")
    for domain in sorted(by_domain):
        print(f"- {domain:<10} mean_score={_group_mean(by_domain[domain]):.3f}")

    print("\nDomain ranking sanity (strong > moderate > weak):")
    for domain in sorted(ranking_sanity):
        print(f"- {domain:<10} {ranking_sanity[domain]}")

    print("\nComponent score means:")
    print(f"- skills_match_score: {_group_mean([r.skills_match_score for r in results]):.3f}")
    print(
        "- experience_match_score: "
        f"{_group_mean([r.experience_match_score for r in results]):.3f}"
    )
    print(f"- role_match_score: {_group_mean([r.role_match_score for r in results]):.3f}")
    print(
        f"- education_match_score: {_group_mean([r.education_match_score for r in results]):.3f}"
    )
    print(
        "- requirement_constraint_score: "
        f"{_group_mean([r.requirement_constraint_score for r in results]):.3f}"
    )

    print("\nJudge outputs:")
    verdict_counts: dict[str, int] = {}
    for r in results:
        verdict_counts[r.judge_verdict] = verdict_counts.get(r.judge_verdict, 0) + 1
    for verdict, count in sorted(verdict_counts.items()):
        print(f"- {verdict or 'unknown'}: {count}")
    print(
        f"- mean_judge_confidence: {_group_mean([r.judge_confidence for r in results]):.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_eval()))
