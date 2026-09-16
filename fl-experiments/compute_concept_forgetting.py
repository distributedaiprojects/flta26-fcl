#!/usr/bin/env python3
"""Compute concept-level forgetting from FL result CSV files.

This script reads result CSV files produced by the federated learning experiments
and computes forgetting and retention metrics at the concept level.

For each client and concept label set, we compute the accuracy trajectory for each fixed round segment. Then we compute forgetting and retention per client-concept segment and average those values first within each client and then across clients.

Each segment corresponds to a contiguous block of rounds, so concept changes can
be handled as repeated segments within one experiment run.

For each client concept segment, we compute:
  P_max,c,j,s = maximum accuracy for client j on concept c in segment s
  P_N,c,j,s = final accuracy for client j on concept c in segment s

The forgetting score for each client concept segment is:
  forgetting_accuracy_c,j,s = P_max,c,j,s - P_N,c,j,s

Each client's overall score is the mean of its concept segment values, and the
federation summary is the mean over all clients.

Usage:
  python fl-experiments/compute_concept_forgetting.py fl-experiments/results --segment-size 40
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

SCENARIO1_EXPERIMENT_IDS = {311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 331, 332}
SCENARIO2_EXPERIMENT_IDS = {321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 333, 334}


def parse_metrics(raw: Any) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        value = ast.literal_eval(str(raw))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def load_csv_rows(path: Path, run_id: str) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_run_id"] = run_id
            yield row


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        return int(text) if text != "" else None
    except ValueError:
        return None


def compute_accuracy_curve_auc(rounds: List[int], accuracies: List[float]) -> float:
    if len(rounds) < 2:
        return float(sum(accuracies) / len(accuracies)) if accuracies else 0.0
    auc = 0.0
    for i in range(len(rounds) - 1):
        x0, x1 = rounds[i], rounds[i + 1]
        y0, y1 = accuracies[i], accuracies[i + 1]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    span = rounds[-1] - rounds[0]
    return auc / span if span > 0 else float(sum(accuracies) / len(accuracies))


def load_experiment_config(experiment_id: int) -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent / "conf" / f"exp{experiment_id}.py"
    if not config_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"exp{experiment_id}", config_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("__")}


def infer_algorithm_name(experiment_id: int) -> str:
    config = load_experiment_config(experiment_id)
    if not config:
        return "unknown"

    if config.get("use_cfed", False):
        return "CFeD"
    if config.get("fedprox_mu", None) is not None:
        return "FedProx"

    use_smo = config.get("use_smo", False)
    use_sample_selection = config.get("use_sample_selection", False)

    if use_sample_selection and use_smo:
        return "SS-KD-FCL"
    if use_sample_selection:
        return "SS-FCL"
    if use_smo:
        return "KD-FCL"
    return "FedAvg"


def extract_concept_histories(rows: Iterable[Dict[str, str]], segment_size: int) -> Dict[Tuple[int, str, int, Tuple[int, ...], int, int], List[float]]:
    client_concept_data: Dict[Tuple[int, str, int, Tuple[int, ...], int, int], List[float]] = defaultdict(list)
    for row in rows:
        if "experiment_id" not in row or "round" not in row or "client_label_metrics" not in row:
            continue

        experiment_id = safe_int(row["experiment_id"])
        round_number = safe_int(row["round"])
        run_id = str(row.get("_run_id", "unknown"))
        if experiment_id is None or round_number is None:
            continue

        segment_id = (round_number - 1) // segment_size if segment_size > 0 else 0
        metrics = parse_metrics(row["client_label_metrics"])
        if not metrics:
            continue

        for client_id_raw, entries in metrics.items():
            if not isinstance(entries, list) or not entries:
                continue
            first_entry = entries[0]
            if not isinstance(first_entry, dict):
                continue
            client_id = safe_int(client_id_raw)
            if client_id is None:
                continue
            labels = tuple(first_entry.get("labels", []))
            accuracy = float(first_entry.get("accuracy", 0.0))
            client_concept_data[(experiment_id, run_id, client_id, labels, segment_id, round_number)].append(accuracy)

    return client_concept_data


def compute_forgetting_for_concept(segment_accuracies: List[float], num_concepts_for_client: int) -> Tuple[float, float, float, float, float]:
    if not segment_accuracies:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    final_accuracy = segment_accuracies[-1]
    early_accuracy = mean(segment_accuracies) if num_concepts_for_client == 1 else max(segment_accuracies[:2])
    forgetting_accuracy = early_accuracy - final_accuracy
    retention_accuracy = final_accuracy / early_accuracy if early_accuracy > 0 else 0.0
    accuracy_curve_auc = compute_accuracy_curve_auc(list(range(len(segment_accuracies))), segment_accuracies)
    return early_accuracy, final_accuracy, forgetting_accuracy, retention_accuracy, accuracy_curve_auc


def compute_generic_concept_forgetting(segment_accuracies: List[float]) -> Tuple[float, float, float, float, float]:
    if not segment_accuracies:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    final_accuracy = segment_accuracies[-1]
    if len(segment_accuracies) > 1:
        early_accuracy = max(segment_accuracies[:-1])
    else:
        early_accuracy = final_accuracy

    forgetting_accuracy = early_accuracy - final_accuracy
    retention_accuracy = final_accuracy / early_accuracy if early_accuracy > 0 else 0.0
    accuracy_curve_auc = compute_accuracy_curve_auc(list(range(len(segment_accuracies))), segment_accuracies)
    return early_accuracy, final_accuracy, forgetting_accuracy, retention_accuracy, accuracy_curve_auc


def compute_concept_forgetting(
    client_concept_data: Dict[Tuple[int, str, int, Tuple[int, ...], int, int], List[float]],
    segment_size: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    per_client_metrics: Dict[Tuple[int, str, int], List[Dict[str, float]]] = defaultdict(list)

    grouped: Dict[Tuple[int, str, int, Tuple[int, ...], int], List[Tuple[int, float]]] = defaultdict(list)
    concept_segment_max: Dict[Tuple[int, str, int, Tuple[int, ...]], Dict[int, float]] = defaultdict(dict)

    for (experiment_id, run_id, client_id, labels, segment_id, round_number), accuracies in client_concept_data.items():
        grouped[(experiment_id, run_id, client_id, labels, segment_id)].append((round_number, mean(accuracies)))

    for (experiment_id, run_id, client_id, labels, segment_id), entries in grouped.items():
        if not entries:
            continue
        segment_max = max(entry[1] for entry in entries)
        concept_key = (experiment_id, run_id, client_id, labels)
        concept_segment_max[concept_key][segment_id] = segment_max

    client_concept_map: Dict[Tuple[int, str, int], Dict[Tuple[int, ...], List[float]]] = defaultdict(lambda: defaultdict(list))
    for (experiment_id, run_id, client_id, labels), segment_map in concept_segment_max.items():
        segment_ids = sorted(segment_map)
        accuracies = [segment_map[s] for s in segment_ids]
        client_concept_map[(experiment_id, run_id, client_id)][labels] = accuracies

    for (experiment_id, run_id, client_id), concept_map in sorted(client_concept_map.items()):
        if not concept_map:
            continue

        num_concepts_for_client = len(concept_map)
        client_forgetting_scores: List[float] = []
        client_retention_scores: List[float] = []
        client_auc_scores: List[float] = []
        is_special_scenario = experiment_id in SCENARIO1_EXPERIMENT_IDS or experiment_id in SCENARIO2_EXPERIMENT_IDS

        for labels, accuracies in sorted(concept_map.items()):
            if is_special_scenario:
                early_accuracy, final_accuracy, forgetting_accuracy, retention_accuracy, accuracy_curve_auc = compute_forgetting_for_concept(
                    accuracies, num_concepts_for_client
                )
            else:
                early_accuracy, final_accuracy, forgetting_accuracy, retention_accuracy, accuracy_curve_auc = compute_generic_concept_forgetting(accuracies)

            client_forgetting_scores.append(forgetting_accuracy)
            client_retention_scores.append(retention_accuracy)
            client_auc_scores.append(accuracy_curve_auc)

            results.append(
                {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "client_id": client_id,
                    "concept_labels": labels,
                    "segment_ids": tuple(sorted(concept_segment_max[(experiment_id, run_id, client_id, labels)])),
                    "last_segment_id": max(concept_segment_max[(experiment_id, run_id, client_id, labels)]),
                    "early_accuracy": round(early_accuracy, 6),
                    "final_accuracy": round(final_accuracy, 6),
                    "forgetting_accuracy": round(forgetting_accuracy, 6),
                    "retention_accuracy": round(retention_accuracy, 6),
                    "accuracy_curve_auc": round(accuracy_curve_auc, 6),
                }
            )

        client_average_forgetting = mean(client_forgetting_scores)
        client_average_retention = mean(client_retention_scores)
        client_average_auc = mean(client_auc_scores)
        per_client_metrics[(experiment_id, run_id, client_id)].append(
            {
                "average_forgetting_accuracy": client_average_forgetting,
                "average_retention_accuracy": client_average_retention,
                "average_accuracy_curve_auc": client_average_auc,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    run_client_averages: Dict[Tuple[int, str], List[Dict[str, float]]] = defaultdict(list)

    for (experiment_id, run_id, client_id), client_metrics in per_client_metrics.items():
        if not client_metrics:
            continue
        run_client_averages[(experiment_id, run_id)].append(client_metrics[0])

    experiment_run_stats: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (experiment_id, run_id), client_metrics in run_client_averages.items():
        if not client_metrics:
            continue
        num_clients = len(client_metrics)
        experiment_run_stats[experiment_id].append(
            {
                "run_id": run_id,
                "num_clients": num_clients,
                "average_forgetting_accuracy": sum(m["average_forgetting_accuracy"] for m in client_metrics) / num_clients,
                "average_retention_accuracy": sum(m["average_retention_accuracy"] for m in client_metrics) / num_clients,
                "average_accuracy_curve_auc": sum(m["average_accuracy_curve_auc"] for m in client_metrics) / num_clients,
            }
        )

    for experiment_id, run_stats in experiment_run_stats.items():
        num_runs = len(run_stats)
        if num_runs == 0:
            continue
        total_clients = sum(stat["num_clients"] for stat in run_stats)
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "num_runs": num_runs,
                "num_clients": round(total_clients / num_runs, 2),
                "average_forgetting_accuracy": sum(stat["average_forgetting_accuracy"] for stat in run_stats) / num_runs,
                "average_retention_accuracy": round(sum(stat["average_retention_accuracy"] for stat in run_stats) / num_runs, 6),
                "average_accuracy_curve_auc": round(sum(stat["average_accuracy_curve_auc"] for stat in run_stats) / num_runs, 6),
            }
        )

    summary_rows.sort(key=lambda item: item["experiment_id"])
    return results, summary_rows


def save_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_csv_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    raise FileNotFoundError(f"Path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute concept-level forgetting from FL result CSVs")
    parser.add_argument("input_path", help="CSV file or directory containing result CSV files")
    parser.add_argument("--output-dir", default=None, help="Directory to write output CSVs; defaults to the input directory")
    parser.add_argument("--algorithm", default="unknown", help="Name of the algorithm or method used for the experiment")
    parser.add_argument("--segment-size", default=40, type=int, help="Number of rounds per concept segment")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    csv_paths = discover_csv_paths(input_path)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_path}")

    all_rows: List[Dict[str, Any]] = []
    for csv_path in csv_paths:
        if csv_path.name in {"concept_forgetting_details.csv", "concept_forgetting_summary.csv"}:
            continue
        all_rows.extend(load_csv_rows(csv_path, csv_path.stem))

    segment_size = max(1, args.segment_size)
    client_concept_data = extract_concept_histories(all_rows, segment_size)
    concept_results, summary_results = compute_concept_forgetting(client_concept_data, segment_size)

    algorithm_name = args.algorithm
    algorithm_map: Dict[int, str] = {}
    if algorithm_name == "unknown":
        experiment_ids = {row["experiment_id"] for row in all_rows if "experiment_id" in row}
        for exp_id in experiment_ids:
            if isinstance(exp_id, str):
                exp_id_int = safe_int(exp_id)
                if exp_id_int is not None:
                    algorithm_map[exp_id_int] = infer_algorithm_name(exp_id_int)
            elif isinstance(exp_id, int):
                algorithm_map[exp_id] = infer_algorithm_name(exp_id)
    else:
        algorithm_map = defaultdict(lambda: algorithm_name)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        input_path if input_path.is_dir() else input_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = out_dir / "concept_forgetting_details.csv"
    summary_path = out_dir / "concept_forgetting_summary.csv"

    save_csv(
        details_path,
        [
            "experiment_id",
            "algorithm",
            "run_id",
            "client_id",
            "concept_labels",
            "segment_ids",
            "last_segment_id",
            "early_accuracy",
            "final_accuracy",
            "forgetting_accuracy",
            "retention_accuracy",
            "accuracy_curve_auc",
        ],
        [
            {
                **row,
                "algorithm": algorithm_map.get(row["experiment_id"], algorithm_name),
            }
            for row in concept_results
        ],
    )

    save_csv(
        summary_path,
        [
            "experiment_id",
            "algorithm",
            "num_runs",
            "num_clients",
            "average_forgetting_accuracy",
            "average_retention_accuracy",
            "average_accuracy_curve_auc",
        ],
        [
            {
                **row,
                "algorithm": algorithm_map.get(row["experiment_id"], algorithm_name),
            }
            for row in summary_results
        ],
    )

    print(f"Wrote: {details_path}")
    print(f"Wrote: {summary_path}")
    print("Summary:")
    for row in summary_results:
        print(
            f"algorithm={algorithm_name} exp={row['experiment_id']} clients={row['num_clients']} avg_forgetting_acc={row['average_forgetting_accuracy']:.6f} avg_retention_acc={row['average_retention_accuracy']:.6f}"
        )


if __name__ == "__main__":
    main()
