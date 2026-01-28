import csv
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = os.path.dirname(os.path.abspath(__file__))


def _safe_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def load_csv(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            ex_id = row.get("exercise_id")
            if not ex_id:
                continue
            out[ex_id] = row
        return out


def pick(row: Dict[str, str], key: str) -> Optional[float]:
    return _safe_float(row.get(key, ""))


def compare() -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Optional[float]]]]:
    ma1_path = os.path.join(ROOT, "outputs", "multi_agent", "rq2_results.csv")
    ma2_path = os.path.join(ROOT, "outputs", "multi_agent_2nd", "rq2_results.csv")
    sa_path = os.path.join(ROOT, "outputs", "single_agent", "rq1_results.csv")

    ma1 = load_csv(ma1_path)
    ma2 = load_csv(ma2_path)
    sa = load_csv(sa_path)

    exercise_ids = sorted(set(ma1) | set(ma2) | set(sa))

    metrics = [
        "judge_score_0_100",
        "class_f1",
        "relation_f1",
        "multiplicity_accuracy",
        "render_success_generated",
        "render_success_ground_truth",
    ]

    rows: List[Dict[str, str]] = []
    for ex_id in exercise_ids:
        r1 = ma1.get(ex_id, {})
        r2 = ma2.get(ex_id, {})
        rs = sa.get(ex_id, {})

        out: Dict[str, str] = {"exercise_id": ex_id}
        for m in metrics:
            v1 = pick(r1, m)
            v2 = pick(r2, m)
            vs = pick(rs, m)
            out[f"{m}_ma1"] = "" if v1 is None else str(v1)
            out[f"{m}_ma2"] = "" if v2 is None else str(v2)
            out[f"{m}_sa"] = "" if vs is None else str(vs)
            out[f"{m}_delta_ma2_ma1"] = "" if (v1 is None or v2 is None) else str(v2 - v1)
            out[f"{m}_delta_ma2_sa"] = "" if (vs is None or v2 is None) else str(v2 - vs)

        # Extra multi-agent-only fields that are often useful to compare
        out["critic_score_0_100_ma1"] = "" if pick(r1, "critic_score_0_100") is None else str(pick(r1, "critic_score_0_100"))
        out["critic_score_0_100_ma2"] = "" if pick(r2, "critic_score_0_100") is None else str(pick(r2, "critic_score_0_100"))
        out["selected_source_ma2"] = r2.get("selected_source", "")
        out["render_success_selected_ma2"] = "" if pick(r2, "render_success_selected") is None else str(pick(r2, "render_success_selected"))
        rows.append(out)

    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for label, dataset in [("ma1", ma1), ("ma2", ma2), ("sa", sa)]:
        summary[label] = {}
        for m in metrics:
            summary[label][m] = _mean(pick(r, m) for r in dataset.values())
        if label in ("ma1", "ma2"):
            summary[label]["critic_score_0_100"] = _mean(pick(r, "critic_score_0_100") for r in dataset.values())
        if label == "ma2":
            summary[label]["render_success_selected"] = _mean(pick(r, "render_success_selected") for r in dataset.values())

    return rows, summary


def main() -> None:
    rows, summary = compare()

    out_csv = os.path.join(ROOT, "outputs", "comparison_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else ["exercise_id"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote:", out_csv)
    print("\nAverages (mean over available exercises):")
    for label in ("ma1", "ma2", "sa"):
        items = ", ".join(
            f"{k}={summary[label][k]:.3f}" for k in sorted(summary[label].keys()) if summary[label][k] is not None
        )
        print(f"- {label}: {items}")


if __name__ == "__main__":
    main()

