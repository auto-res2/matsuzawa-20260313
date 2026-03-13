"""Dataset preprocessing for GSM8K math word problems."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
    subset_seed: int = 42,
    cache_dir: str = ".cache",
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split ('train' or 'test')
        max_samples: Maximum number of samples to load (None for all)
        subset_seed: Random seed for subset selection
        cache_dir: Directory to cache downloaded datasets

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Extract samples
    samples = []
    for item in dataset:
        question = item["question"]
        # GSM8K answers are in format "#### 42"
        answer_text = item["answer"]
        answer = extract_numerical_answer(answer_text)

        samples.append(
            {
                "question": question,
                "answer": answer,
                "answer_text": answer_text,
            }
        )

    # Subset if needed
    if max_samples is not None and max_samples < len(samples):
        import random

        random.seed(subset_seed)
        samples = random.sample(samples, max_samples)

    return samples


def extract_numerical_answer(text: str) -> str:
    """
    Extract numerical answer from GSM8K answer format.
    GSM8K answers end with "#### [number]"

    Args:
        text: Answer text

    Returns:
        Numerical answer as string
    """
    # Look for #### pattern
    match = re.search(r"####\s*([0-9,\.]+)", text)
    if match:
        answer = match.group(1)
        # Remove commas
        answer = answer.replace(",", "")
        return answer

    # Fallback: try to find any number at the end
    match = re.search(r"([0-9,\.]+)\s*$", text)
    if match:
        answer = match.group(1)
        answer = answer.replace(",", "")
        return answer

    return text.strip()


def normalize_answer(answer: str) -> str:
    """
    Normalize numerical answer for comparison.

    Args:
        answer: Answer string

    Returns:
        Normalized answer
    """
    # Remove commas and extra whitespace
    answer = answer.replace(",", "").strip()

    # Try to convert to float and back to handle equivalences like "8" vs "8.0"
    try:
        num = float(answer)
        # If it's a whole number, format as int
        if num == int(num):
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        return answer


def answers_match(predicted: str, gold: str) -> bool:
    """
    Check if two answers match after normalization.

    Args:
        predicted: Predicted answer
        gold: Gold answer

    Returns:
        True if answers match
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    return pred_norm == gold_norm
