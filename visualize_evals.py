import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

list_experiments_path = {"base": "evals/eval_results_base_1176764.json", "dsa": "evals/eval_results_dsa_1210338.json"}
tasks_to_plot = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "ruler", "mmlu", "babilong"]


def plot_bar_comparison_between_models(experiments: Dict[str, Dict[str, List[Tuple[str, float]]]]):
    """
    Create bar chart comparisons for all metrics across different experiments.

    Args:
        experiments: Dict mapping experiment names to their metrics
                    Format: {exp_name: {metric_name: [(task, value), ...]}}
    """
    # Get all unique metrics across all experiments
    all_metrics = set()
    for exp_metrics in experiments.values():
        all_metrics.update(exp_metrics.keys())

    all_metrics = sorted(all_metrics)

    # Collect all unique task-metric combinations and filter out those with no data
    all_task_metrics = set()
    task_metric_has_data = {}

    for exp_metrics in experiments.values():
        for metric, task_values in exp_metrics.items():
            for task, value in task_values:
                key = f"{task}_{metric}"
                all_task_metrics.add(key)
                # Track if this combination has non-zero data in any experiment
                if key not in task_metric_has_data:
                    task_metric_has_data[key] = False
                if value != 0:
                    task_metric_has_data[key] = True

    # Filter to only include combinations that have data
    all_task_metrics = sorted([tm for tm in all_task_metrics if task_metric_has_data.get(tm, False)])

    # Create a single subplot
    fig, ax = plt.subplots(1, 1, figsize=(max(16, len(all_task_metrics) * 0.5), 8))

    x = np.arange(len(all_task_metrics))
    width = 0.8 / len(experiments)

    for exp_idx, (exp_name, exp_metrics) in enumerate(experiments.items()):
        values = []
        for task_metric in all_task_metrics:
            # Parse task and metric from combined string
            # Try to find the metric by checking all known metrics
            found = False
            task = None
            metric = None

            for m in all_metrics:
                if task_metric.endswith(f"_{m}"):
                    task = task_metric[:-len(m)-1]
                    metric = m
                    found = True
                    break

            if not found:
                values.append(0)
                continue

            # Find value for this task-metric combination
            value = 0.0
            if metric in exp_metrics:
                for t, v in exp_metrics[metric]:
                    if t == task:
                        value = float(v)
                        break
            values.append(value)

        offset = width * exp_idx - 0.8 / 2 + width / 2
        ax.bar(x + offset, values, width, label=exp_name, alpha=0.8)

    ax.set_xlabel('Task - Metric')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of all metrics across tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(all_task_metrics, rotation=90, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to metrics_comparison.png")
    plt.show()


def get_metrics(path: str):
    # Load data from JSON file
    with open(path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    tasks = list(results.keys())
    metrics = {}

    for task in tasks:
        if task not in tasks_to_plot:
            continue
        for metric, value in results[task].items():
            # Only grab acc,none metric
            if metric != "acc,none":
                continue
            # Skip non-numeric values like 'alias'
            if not isinstance(value, (int, float)):
                continue
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append((task, value))

    # metric[metric_name] = [(task, value), ...] # metric_name example: acc,none
    return metrics

if __name__ == "__main__":
    experiments = {}
    for exp_name, path in list_experiments_path.items():
        metrics = get_metrics(path)
        experiments[exp_name] = metrics

    plot_bar_comparison_between_models(experiments)