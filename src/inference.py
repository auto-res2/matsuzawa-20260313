"""Inference script for prompt-based evaluation."""

import json
import os
from pathlib import Path
from typing import Dict, List
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model import GeminiModel
from src.preprocess import load_gsm8k, extract_answer_from_response


def run_inference(cfg: DictConfig) -> Dict:
    """
    Run inference for a single run configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with metrics and results
    """
    # Initialize WandB if enabled
    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        # Adjust project name for sanity/pilot modes
        project = cfg.wandb.project
        if cfg.mode == "sanity":
            project = f"{project}-sanity"
        elif cfg.mode == "pilot":
            project = f"{project}-pilot"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run: {wandb.run.get_url()}")

    # Determine sample size based on mode
    max_samples = cfg.run.inference.max_samples
    if cfg.mode == "sanity":
        max_samples = 10  # Small sample for sanity check
    elif cfg.mode == "pilot":
        max_samples = 200  # Pilot dataset size
    # else: full mode uses config value (None = all)

    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name} ({cfg.run.dataset.split})")
    dataset = load_gsm8k(
        split=cfg.run.dataset.split,
        subset=cfg.run.dataset.subset,
        max_samples=max_samples,
        cache_dir=".cache",
    )
    print(f"Loaded {len(dataset)} samples")

    # Initialize model
    print(f"Initializing model: {cfg.run.model.name}")
    model = GeminiModel(
        model_name=cfg.run.model.name,
        temperature=cfg.run.model.temperature,
        max_tokens=cfg.run.model.max_tokens,
        top_p=cfg.run.model.top_p,
    )

    # Run inference
    results = []
    correct = 0
    invalid_count = 0
    total_word_count = 0

    print(f"Running inference with template: {cfg.run.inference.prompt_template}")
    for example in tqdm(dataset, desc="Processing"):
        # Format prompt
        prompt = model.format_prompt(
            cfg.run.inference.prompt_template, example["question"]
        )

        # Generate response
        response = model.generate(prompt)

        # Extract predicted answer
        predicted_answer = extract_answer_from_response(response["text"])

        # Check correctness
        is_correct = False
        is_valid = predicted_answer is not None

        if is_valid and example["answer"] is not None:
            # Allow small tolerance for floating point comparison
            is_correct = abs(predicted_answer - example["answer"]) < 0.01

        # Count words in response
        word_count = len(response["text"].split())
        total_word_count += word_count

        if is_correct:
            correct += 1
        if not is_valid:
            invalid_count += 1

        # Store result
        result = {
            "id": example["id"],
            "question": example["question"],
            "gold_answer": example["answer"],
            "predicted_answer": predicted_answer,
            "raw_response": response["text"],
            "prompt_template": cfg.run.inference.prompt_template,
            "is_correct": is_correct,
            "is_valid": is_valid,
            "word_count": word_count,
            "error": response.get("error"),
        }
        results.append(result)

        # Log to WandB periodically
        if wandb_enabled and len(results) % 10 == 0:
            wandb.log(
                {
                    "samples_processed": len(results),
                    "accuracy": correct / len(results),
                    "invalid_rate": invalid_count / len(results),
                }
            )

    # Calculate final metrics
    total_samples = len(results)
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    invalid_rate = invalid_count / total_samples if total_samples > 0 else 0.0
    avg_word_count = total_word_count / total_samples if total_samples > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_samples": correct,
        "invalid_samples": invalid_count,
        "invalid_rate": invalid_rate,
        "avg_word_count": avg_word_count,
    }

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total_samples})")
    print(f"  Invalid rate: {invalid_rate:.4f} ({invalid_count}/{total_samples})")
    print(f"  Avg word count: {avg_word_count:.1f}")

    # Sanity validation for inference tasks
    if cfg.mode == "sanity":
        sanity_pass = True
        reason = ""

        if total_samples < 5:
            sanity_pass = False
            reason = "insufficient_samples"
        elif invalid_count == total_samples:
            sanity_pass = False
            reason = "all_invalid"
        elif accuracy == 0 and invalid_count < total_samples:
            # If we have valid predictions but 0% accuracy, it's suspicious but not necessarily a failure
            # (might be genuinely hard problems)
            pass

        summary = {
            "samples": total_samples,
            "outputs_valid": total_samples - invalid_count,
            "outputs_unique": len(
                set(r["predicted_answer"] for r in results if r["is_valid"])
            ),
        }

        if sanity_pass:
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason={reason}")

        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Pilot validation for inference tasks
    if cfg.mode == "pilot":
        pilot_pass = True
        reason = ""

        if total_samples < 50:
            pilot_pass = False
            reason = "insufficient_samples"
        elif invalid_count == total_samples:
            pilot_pass = False
            reason = "all_invalid"
        elif accuracy == 0 and invalid_count < total_samples:
            # Similar to sanity, 0% accuracy is suspicious but might be legitimate
            pass

        summary = {
            "samples": total_samples,
            "primary_metric": "accuracy",
            "primary_metric_value": accuracy,
            "outputs_unique": len(
                set(r["predicted_answer"] for r in results if r["is_valid"])
            ),
        }

        if pilot_pass:
            print("PILOT_VALIDATION: PASS")
        else:
            print(f"PILOT_VALIDATION: FAIL reason={reason}")

        print(f"PILOT_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Save results to disk
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save per-example results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")

    # Log final metrics to WandB
    if wandb_enabled:
        wandb.summary.update(metrics)

        # Log some example results
        wandb_table = wandb.Table(
            columns=[
                "id",
                "question",
                "gold_answer",
                "predicted_answer",
                "is_correct",
                "word_count",
            ]
        )
        for result in results[:100]:  # Log first 100 examples
            wandb_table.add_data(
                result["id"],
                result["question"][:100],  # Truncate for display
                result["gold_answer"],
                result["predicted_answer"],
                result["is_correct"],
                result["word_count"],
            )
        wandb.log({"results_sample": wandb_table})

        wandb.finish()

    return metrics


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="../config", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        run_inference(cfg)

    main()
