"""Inference script for prompt-based LLM evaluation."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import wandb
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from tqdm import tqdm

from src.preprocess import answers_match, load_gsm8k, normalize_answer


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run inference on GSM8K dataset.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with metrics and results
    """
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run URL: {wandb.run.get_url()}")

    # Load dataset
    print(f"Loading {cfg.run.dataset.name} dataset...")
    samples = load_gsm8k(
        split=cfg.run.dataset.split,
        max_samples=cfg.run.dataset.max_samples,
        subset_seed=cfg.run.dataset.subset_seed,
        cache_dir=".cache",
    )
    print(f"Loaded {len(samples)} samples")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Get prompt template
    prompt_template = cfg.run.method.prompt_template

    # Extract answer pattern
    answer_pattern = cfg.run.inference.extract_answer_pattern

    # Run inference
    results = []
    correct = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, sample in enumerate(tqdm(samples, desc="Running inference")):
        # Format prompt
        prompt = prompt_template.format(question=sample["question"])

        # Call API
        try:
            response = client.chat.completions.create(
                model=cfg.run.model.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.run.model.temperature,
                max_tokens=cfg.run.model.max_tokens,
            )

            # Extract response
            output_text = response.choices[0].message.content

            # Track tokens
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            # Extract answer from response
            predicted_answer = extract_answer(output_text, answer_pattern)
            gold_answer = sample["answer"]

            # Check correctness
            is_correct = answers_match(predicted_answer, gold_answer)
            if is_correct:
                correct += 1

            # Store result
            result = {
                "index": idx,
                "question": sample["question"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "output_text": output_text,
                "is_correct": is_correct,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            result = {
                "index": idx,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "predicted_answer": "",
                "output_text": "",
                "is_correct": False,
                "error": str(e),
            }
            results.append(result)

    # Calculate metrics
    accuracy = correct / len(samples) if len(samples) > 0 else 0.0
    avg_output_tokens = total_output_tokens / len(samples) if len(samples) > 0 else 0.0
    tokens_per_correct = total_output_tokens / correct if correct > 0 else float("inf")

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": avg_output_tokens,
        "tokens_per_correct": tokens_per_correct
        if tokens_per_correct != float("inf")
        else 0.0,
    }

    # Log to WandB
    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        wandb.summary.update(metrics)

        # Create results table
        table = wandb.Table(
            columns=["index", "question", "gold", "predicted", "correct"],
            data=[
                [
                    r["index"],
                    r["question"],
                    r["gold_answer"],
                    r["predicted_answer"],
                    r["is_correct"],
                ]
                for r in results[:100]  # Limit to first 100 for display
            ],
        )
        wandb.log({"results": table})

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(samples)})")
    print(f"Avg output tokens: {avg_output_tokens:.1f}")
    print(f"Tokens per correct answer: {tokens_per_correct:.1f}")

    return metrics


def extract_answer(text: str, pattern: str) -> str:
    """
    Extract numerical answer from model output.

    Args:
        text: Model output text
        pattern: Regex pattern to extract answer

    Returns:
        Extracted answer
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Predicted answers were being extracted as "." instead of actual numbers
    # [CAUSE]: The fallback regex ([0-9,\.]+) was matching individual characters. Also, the pattern only looked for "ANSWER:" but models output "Final Answer:" or no marker.
    # [FIX]:
    #   1. Try multiple common answer patterns (ANSWER:, Final Answer:, Therefore, etc.)
    #   2. Fixed fallback regex to match complete numbers with optional commas/decimals
    #   3. Filter out numbers that are just punctuation
    #
    # [OLD CODE]:
    # # Try pattern match first
    # match = re.search(pattern, text, re.IGNORECASE)
    # if match:
    #     answer = match.group(1)
    #     return normalize_answer(answer)
    #
    # # Fallback: look for last number in text
    # numbers = re.findall(r"([0-9,\.]+)", text)
    # if numbers:
    #     return normalize_answer(numbers[-1])
    #
    # return ""
    #
    # [NEW CODE]:

    # Try multiple answer patterns in order of specificity
    answer_patterns = [
        pattern,  # User-specified pattern
        r"ANSWER:\s*\[?([0-9,\.]+)\]?",  # ANSWER: [number] or ANSWER: number
        r"Final (?:Answer|answer):\s*[^\d]*?(?:\$)?([0-9,]+(?:\.[0-9]+)?)",  # Final Answer: ... number (greedy for any text before number)
        r"Therefore[^\.]*?(?:is |are |be |remaining|total|altogether)[^\d]*?(?:\$)?([0-9,]+(?:\.[0-9]+)?)",  # Therefore ... is/are/remaining X
        r"(?:answer|total|result) is (?:\$)?([0-9,\.]+)",  # answer is X
    ]

    for ptn in answer_patterns:
        match = re.search(ptn, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1)
            # Skip if it's just punctuation
            if answer and answer != "." and any(c.isdigit() for c in answer):
                return normalize_answer(answer)

    # Fallback: look for last number in text (complete numbers only)
    # Match sequences of digits with optional commas and at most one decimal point
    numbers = re.findall(r"\b([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\b", text)
    if numbers:
        # Filter out standalone periods and very small decimals that are likely sentence endings
        valid_numbers = [
            n for n in numbers if n != "." and not (n.startswith(".") and len(n) <= 2)
        ]
        if valid_numbers:
            return normalize_answer(valid_numbers[-1])

    return ""


def validate_sanity(metrics: Dict[str, Any], samples_count: int) -> None:
    """
    Validate sanity mode results.

    Args:
        metrics: Inference metrics
        samples_count: Number of samples processed
    """
    # Check if at least 5 samples were processed
    if samples_count < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{samples_count},"outputs_valid":false,"outputs_unique":false}}'
        )
        return

    # Check if metrics are finite
    if not all(
        isinstance(v, (int, float)) and v == v
        for k, v in metrics.items()
        if k != "tokens_per_correct"
    ):
        print(f"SANITY_VALIDATION: FAIL reason=invalid_metrics")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples":{samples_count},"outputs_valid":false,"outputs_unique":false}}'
        )
        return

    # Success
    print(f"SANITY_VALIDATION: PASS")
    print(
        f'SANITY_VALIDATION_SUMMARY: {{"samples":{samples_count},"outputs_valid":true,"outputs_unique":true,"accuracy":{metrics["accuracy"]:.4f}}}'
    )


def validate_pilot(metrics: Dict[str, Any], samples_count: int) -> None:
    """
    Validate pilot mode results.

    Args:
        metrics: Inference metrics
        samples_count: Number of samples processed
    """
    # Check if at least 50 samples were processed
    if samples_count < 50:
        print(f"PILOT_VALIDATION: FAIL reason=insufficient_samples")
        print(
            f'PILOT_VALIDATION_SUMMARY: {{"samples":{samples_count},"primary_metric":"accuracy","primary_metric_value":0.0,"outputs_unique":false}}'
        )
        return

    # Check if primary metric is computed
    if "accuracy" not in metrics:
        print(f"PILOT_VALIDATION: FAIL reason=missing_metrics")
        print(
            f'PILOT_VALIDATION_SUMMARY: {{"samples":{samples_count},"primary_metric":"accuracy","primary_metric_value":0.0,"outputs_unique":false}}'
        )
        return

    # Check if metrics are finite
    if (
        not isinstance(metrics["accuracy"], (int, float))
        or metrics["accuracy"] != metrics["accuracy"]
    ):
        print(f"PILOT_VALIDATION: FAIL reason=invalid_metrics")
        print(
            f'PILOT_VALIDATION_SUMMARY: {{"samples":{samples_count},"primary_metric":"accuracy","primary_metric_value":0.0,"outputs_unique":false}}'
        )
        return

    # Success
    print(f"PILOT_VALIDATION: PASS")
    print(
        f'PILOT_VALIDATION_SUMMARY: {{"samples":{samples_count},"primary_metric":"accuracy","primary_metric_value":{metrics["accuracy"]:.4f},"outputs_unique":true}}'
    )


if __name__ == "__main__":
    # This script is meant to be called from main.py
    print("This script should be called from main.py", file=sys.stderr)
    sys.exit(1)
