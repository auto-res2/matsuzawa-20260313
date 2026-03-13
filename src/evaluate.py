"""Evaluation script for aggregating results from multiple runs."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import wandb
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# Use non-interactive backend for server environments
matplotlib.use("Agg")


def fetch_run_from_wandb(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API by display name.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()

    # Fetch runs with matching display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        # Try with -sanity and -pilot suffixes
        for suffix in ["-sanity", "-pilot"]:
            runs = api.runs(
                f"{entity}/{project}{suffix}",
                filters={"display_name": run_id},
                order="-created_at",
            )
            if runs:
                break

    if not runs:
        raise ValueError(f"No run found with display name: {run_id}")

    # Get most recent run with that name
    run = runs[0]

    # Fetch history (logged metrics over time)
    history = run.history()

    return {
        "id": run.id,
        "name": run.name,
        "config": run.config,
        "summary": dict(run.summary),
        "history": history.to_dict("records") if not history.empty else [],
    }


def export_per_run_metrics(run_data: Dict, output_dir: Path) -> None:
    """
    Export per-run metrics to JSON file.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract key metrics
    metrics = {
        "run_id": run_data["name"],
        "accuracy": run_data["summary"].get("accuracy", 0.0),
        "total_samples": run_data["summary"].get("total_samples", 0),
        "correct_samples": run_data["summary"].get("correct_samples", 0),
        "invalid_rate": run_data["summary"].get("invalid_rate", 0.0),
        "avg_word_count": run_data["summary"].get("avg_word_count", 0.0),
    }

    # Save to JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Exported metrics for {run_data['name']} to {output_dir / 'metrics.json'}")


def create_per_run_figures(run_data: Dict, output_dir: Path) -> None:
    """
    Create per-run visualization figures.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    history = run_data["history"]
    if not history:
        print(f"No history data for {run_data['name']}, skipping figures")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(history)

    # Plot accuracy over samples
    if "accuracy" in df.columns and "samples_processed" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(df["samples_processed"], df["accuracy"], marker="o", markersize=3)
        plt.xlabel("Samples Processed")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Progress: {run_data['name']}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_progress.pdf")
        plt.close()
        print(f"Saved {output_dir / 'accuracy_progress.pdf'}")

    # Plot invalid rate over samples
    if "invalid_rate" in df.columns and "samples_processed" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(
            df["samples_processed"],
            df["invalid_rate"],
            marker="o",
            markersize=3,
            color="red",
        )
        plt.xlabel("Samples Processed")
        plt.ylabel("Invalid Rate")
        plt.title(f"Invalid Rate Progress: {run_data['name']}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "invalid_rate_progress.pdf")
        plt.close()
        print(f"Saved {output_dir / 'invalid_rate_progress.pdf'}")


def create_comparison_figures(all_runs: List[Dict], output_dir: Path) -> None:
    """
    Create comparison figures across all runs.

    Args:
        all_runs: List of run data from WandB
        output_dir: Output directory for comparison figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: Accuracy comparison
    plt.figure(figsize=(10, 6))
    run_names = [run["name"] for run in all_runs]
    accuracies = [run["summary"].get("accuracy", 0.0) for run in all_runs]

    colors = ["#2E86AB" if "proposed" in name else "#A23B72" for name in run_names]
    plt.bar(range(len(run_names)), accuracies, color=colors)
    plt.xticks(range(len(run_names)), run_names, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Methods")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_accuracy.pdf")
    plt.close()
    print(f"Saved {output_dir / 'comparison_accuracy.pdf'}")

    # Bar chart: Average word count comparison
    plt.figure(figsize=(10, 6))
    word_counts = [run["summary"].get("avg_word_count", 0.0) for run in all_runs]

    plt.bar(range(len(run_names)), word_counts, color=colors)
    plt.xticks(range(len(run_names)), run_names, rotation=45, ha="right")
    plt.ylabel("Average Word Count")
    plt.title("Response Length Comparison Across Methods")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_word_count.pdf")
    plt.close()
    print(f"Saved {output_dir / 'comparison_word_count.pdf'}")

    # Combined plot: Accuracy vs Word Count
    plt.figure(figsize=(10, 6))
    for run, color in zip(all_runs, colors):
        acc = run["summary"].get("accuracy", 0.0)
        wc = run["summary"].get("avg_word_count", 0.0)
        plt.scatter(wc, acc, s=200, color=color, alpha=0.6, label=run["name"])

    plt.xlabel("Average Word Count")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Response Length")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_accuracy_vs_length.pdf")
    plt.close()
    print(f"Saved {output_dir / 'comparison_accuracy_vs_length.pdf'}")

    # Line plot: Accuracy progress over samples (if history available)
    plt.figure(figsize=(12, 6))
    has_history = False
    for run in all_runs:
        history = run["history"]
        if history:
            df = pd.DataFrame(history)
            if "accuracy" in df.columns and "samples_processed" in df.columns:
                plt.plot(
                    df["samples_processed"],
                    df["accuracy"],
                    marker="o",
                    markersize=2,
                    label=run["name"],
                    alpha=0.7,
                )
                has_history = True

    if has_history:
        plt.xlabel("Samples Processed")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Progress Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_accuracy_progress.pdf")
        plt.close()
        print(f"Saved {output_dir / 'comparison_accuracy_progress.pdf'}")


def aggregate_metrics(all_runs: List[Dict], output_dir: Path) -> None:
    """
    Aggregate metrics across all runs.

    Args:
        all_runs: List of run data from WandB
        output_dir: Output directory for aggregated metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate proposed and baselines
    proposed_runs = [r for r in all_runs if "proposed" in r["name"]]
    baseline_runs = [r for r in all_runs if "proposed" not in r["name"]]

    # Calculate best proposed and best baseline
    best_proposed = (
        max(proposed_runs, key=lambda r: r["summary"].get("accuracy", 0.0))
        if proposed_runs
        else None
    )
    best_baseline = (
        max(baseline_runs, key=lambda r: r["summary"].get("accuracy", 0.0))
        if baseline_runs
        else None
    )

    # Calculate gap
    gap = 0.0
    if best_proposed and best_baseline:
        gap = best_proposed["summary"].get("accuracy", 0.0) - best_baseline[
            "summary"
        ].get("accuracy", 0.0)

    # Build metrics by run_id
    metrics_by_run = {}
    for run in all_runs:
        metrics_by_run[run["name"]] = {
            "accuracy": run["summary"].get("accuracy", 0.0),
            "total_samples": run["summary"].get("total_samples", 0),
            "avg_word_count": run["summary"].get("avg_word_count", 0.0),
            "invalid_rate": run["summary"].get("invalid_rate", 0.0),
        }

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run_id": metrics_by_run,
        "best_proposed": best_proposed["name"] if best_proposed else None,
        "best_proposed_accuracy": best_proposed["summary"].get("accuracy", 0.0)
        if best_proposed
        else 0.0,
        "best_baseline": best_baseline["name"] if best_baseline else None,
        "best_baseline_accuracy": best_baseline["summary"].get("accuracy", 0.0)
        if best_baseline
        else 0.0,
        "gap": gap,
    }

    # Save to JSON
    with open(output_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated Metrics:")
    print(
        f"  Best proposed: {aggregated['best_proposed']} (accuracy: {aggregated['best_proposed_accuracy']:.4f})"
    )
    print(
        f"  Best baseline: {aggregated['best_baseline']} (accuracy: {aggregated['best_baseline_accuracy']:.4f})"
    )
    print(f"  Gap: {aggregated['gap']:.4f}")
    print(f"\nSaved aggregated metrics to {output_dir / 'aggregated_metrics.json'}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate and aggregate results from multiple runs"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (defaults to WANDB_ENTITY env)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project (defaults to WANDB_PROJECT env)",
    )

    args = parser.parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")

    # Get WandB config
    wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY", "airas")
    wandb_project = args.wandb_project or os.getenv(
        "WANDB_PROJECT", "2026-0313-matsuzawa"
    )

    print(f"WandB: {wandb_entity}/{wandb_project}")

    # Fetch all runs from WandB
    all_runs = []
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        try:
            run_data = fetch_run_from_wandb(wandb_entity, wandb_project, run_id)
            all_runs.append(run_data)

            # Export per-run metrics
            run_output_dir = Path(args.results_dir) / run_id
            export_per_run_metrics(run_data, run_output_dir)
            create_per_run_figures(run_data, run_output_dir)

        except Exception as e:
            print(f"Error fetching {run_id}: {e}")
            continue

    if not all_runs:
        print("No runs fetched successfully. Exiting.")
        return

    # Create comparison directory
    comparison_dir = Path(args.results_dir) / "comparison"

    # Generate comparison figures
    print("\nGenerating comparison figures...")
    create_comparison_figures(all_runs, comparison_dir)

    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregate_metrics(all_runs, comparison_dir)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
