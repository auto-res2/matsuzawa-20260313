# [VALIDATOR FIX - Attempt 2]
# [PROBLEM]: Hydra with config_path=None cannot override keys - workflow passes results_dir=... but Hydra says "Key 'results_dir' is not in struct"
# [CAUSE]: Previous attempt (Attempt 1) converted from argparse to Hydra but used config_path=None without defining config structure. When config_path=None, Hydra requires either +key=value syntax or a ConfigStore registration to define valid keys.
# [FIX]: Add a ConfigStore registration with a dataclass to define the expected config structure (results_dir, run_ids) so Hydra accepts key=value overrides without +.
#
# [OLD CODE from Attempt 1]:
# @hydra.main(version_base=None, config_path=None)
# def main(cfg: DictConfig) -> None:
#     results_dir = Path(cfg.results_dir)
#     if isinstance(cfg.run_ids, str):
#         run_ids = json.loads(cfg.run_ids)
#
# [NEW CODE]:
"""Evaluation script for analyzing and comparing experiment results."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


@dataclass
class EvaluateConfig:
    """Configuration for evaluation script."""

    results_dir: str = ".research/results"
    run_ids: str = "[]"


# Register config schema
cs = ConfigStore.instance()
cs.store(name="config", node=EvaluateConfig)


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Config will be passed entirely via command line
    results_dir = Path(cfg.results_dir)
    # Handle run_ids as either a string (JSON) or a list
    if isinstance(cfg.run_ids, str):
        run_ids = json.loads(cfg.run_ids)
    else:
        run_ids = list(cfg.run_ids)

    print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    print(f"Results directory: {results_dir}")

    # Get WandB config from environment or default
    wandb_entity = os.getenv("WANDB_ENTITY", "airas")
    wandb_project = os.getenv("WANDB_PROJECT", "2026-0313-matsuzawa")

    # Initialize WandB API
    api = wandb.Api()

    # Fetch metrics for each run
    all_metrics = {}
    all_configs = {}

    for run_id in run_ids:
        print(f"\nFetching metrics for {run_id}...")

        # Find most recent run with this display name
        runs = api.runs(
            f"{wandb_entity}/{wandb_project}",
            filters={"display_name": run_id},
            order="-created_at",
        )

        if not runs:
            print(f"Warning: No WandB run found for {run_id}")
            continue

        run = runs[0]

        # Get summary metrics
        summary = run.summary._json_dict
        config = run.config

        all_metrics[run_id] = summary
        all_configs[run_id] = config

        # Export per-run metrics
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Accuracy: {summary.get('accuracy', 0):.4f}")
        print(f"  Total samples: {summary.get('total', 0)}")
        print(f"  Saved to {run_dir / 'metrics.json'}")

        # Create per-run figure
        create_per_run_figures(run_dir, run_id, summary)

    if not all_metrics:
        print("Error: No metrics found for any run")
        return

    # Create comparison directory
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Compute aggregated metrics
    aggregated = compute_aggregated_metrics(all_metrics, run_ids)

    with open(comparison_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated metrics saved to {comparison_dir / 'aggregated_metrics.json'}")
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"Best proposed: {aggregated['best_proposed']}")
    print(f"Best baseline: {aggregated['best_baseline']}")
    print(f"Gap: {aggregated['gap']:.4f}")

    # Generate comparison figures
    create_comparison_figures(comparison_dir, all_metrics, run_ids)

    print(f"\nEvaluation complete!")


def create_per_run_figures(run_dir: Path, run_id: str, metrics: Dict[str, Any]) -> None:
    """
    Create per-run visualization figures.

    Args:
        run_dir: Directory to save figures
        run_id: Run identifier
        metrics: Run metrics
    """
    # Create a simple metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    metric_names = []
    metric_values = []

    for key in ["accuracy", "avg_output_tokens", "tokens_per_correct"]:
        if key in metrics:
            metric_names.append(key.replace("_", " ").title())
            metric_values.append(metrics[key])

    if metric_names:
        ax.bar(metric_names, metric_values)
        ax.set_title(f"Metrics for {run_id}")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = run_dir / "metrics_summary.pdf"
        plt.savefig(output_path)
        plt.close()

        print(f"  Figure saved: {output_path}")


def compute_aggregated_metrics(
    all_metrics: Dict[str, Dict], run_ids: List[str]
) -> Dict[str, Any]:
    """
    Compute aggregated metrics across all runs.

    Args:
        all_metrics: Metrics for each run
        run_ids: List of run IDs

    Returns:
        Aggregated metrics dictionary
    """
    # Primary metric is accuracy
    primary_metric = "accuracy"

    # Separate proposed and baseline runs
    proposed_runs = [rid for rid in run_ids if rid.startswith("proposed")]
    baseline_runs = [rid for rid in run_ids if rid.startswith("comparative")]

    # Get best scores
    best_proposed_score = 0.0
    best_proposed_run = None
    for run_id in proposed_runs:
        if run_id in all_metrics and primary_metric in all_metrics[run_id]:
            score = all_metrics[run_id][primary_metric]
            if score > best_proposed_score:
                best_proposed_score = score
                best_proposed_run = run_id

    best_baseline_score = 0.0
    best_baseline_run = None
    for run_id in baseline_runs:
        if run_id in all_metrics and primary_metric in all_metrics[run_id]:
            score = all_metrics[run_id][primary_metric]
            if score > best_baseline_score:
                best_baseline_score = score
                best_baseline_run = run_id

    # Compute gap
    gap = best_proposed_score - best_baseline_score

    # Organize metrics by run
    metrics_by_run = {}
    for run_id in run_ids:
        if run_id in all_metrics:
            metrics_by_run[run_id] = all_metrics[run_id]

    return {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed_run,
        "best_proposed_score": best_proposed_score,
        "best_baseline": best_baseline_run,
        "best_baseline_score": best_baseline_score,
        "gap": gap,
    }


def create_comparison_figures(
    comparison_dir: Path, all_metrics: Dict[str, Dict], run_ids: List[str]
) -> None:
    """
    Create comparison figures across all runs.

    Args:
        comparison_dir: Directory to save figures
        all_metrics: Metrics for each run
        run_ids: List of run IDs
    """
    # Set style
    sns.set_style("whitegrid")

    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = []
    labels = []
    colors = []

    for run_id in run_ids:
        if run_id in all_metrics and "accuracy" in all_metrics[run_id]:
            accuracies.append(all_metrics[run_id]["accuracy"])
            labels.append(run_id)
            # Color code: proposed vs baseline
            if run_id.startswith("proposed"):
                colors.append("steelblue")
            else:
                colors.append("coral")

    if accuracies:
        bars = ax.bar(range(len(labels)), accuracies, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison Across Methods")
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = comparison_dir / "comparison_accuracy.pdf"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    # 2. Efficiency comparison (tokens per correct answer)
    fig, ax = plt.subplots(figsize=(10, 6))

    efficiencies = []
    labels = []
    colors = []

    for run_id in run_ids:
        if run_id in all_metrics and "tokens_per_correct" in all_metrics[run_id]:
            tpc = all_metrics[run_id]["tokens_per_correct"]
            if tpc > 0:  # Skip invalid values
                efficiencies.append(tpc)
                labels.append(run_id)
                if run_id.startswith("proposed"):
                    colors.append("steelblue")
                else:
                    colors.append("coral")

    if efficiencies:
        bars = ax.bar(range(len(labels)), efficiencies, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Tokens per Correct Answer")
        ax.set_title("Efficiency Comparison (Lower is Better)")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = comparison_dir / "comparison_efficiency.pdf"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    # 3. Average output tokens comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_tokens = []
    labels = []
    colors = []

    for run_id in run_ids:
        if run_id in all_metrics and "avg_output_tokens" in all_metrics[run_id]:
            avg_tokens.append(all_metrics[run_id]["avg_output_tokens"])
            labels.append(run_id)
            if run_id.startswith("proposed"):
                colors.append("steelblue")
            else:
                colors.append("coral")

    if avg_tokens:
        bars = ax.bar(range(len(labels)), avg_tokens, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Average Output Tokens")
        ax.set_title("Token Usage Comparison")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = comparison_dir / "comparison_tokens.pdf"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
