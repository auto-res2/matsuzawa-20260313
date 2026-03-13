"""Dataset preprocessing for GSM8K."""

import re
from datasets import load_dataset
from typing import Dict, List, Optional


def load_gsm8k(
    split: str = "test",
    subset: str = "main",
    max_samples: Optional[int] = None,
    cache_dir: str = ".cache",
) -> List[Dict]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split (train or test)
        subset: Dataset subset (main)
        max_samples: Maximum number of samples to load (None for all)
        cache_dir: Cache directory for datasets

    Returns:
        List of dictionaries with keys: id, question, answer (gold numeric answer)
    """
    dataset = load_dataset("gsm8k", subset, split=split, cache_dir=cache_dir)

    samples = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        # Extract numeric answer from the answer string
        # GSM8K answers are in format: "#### {numeric_answer}"
        answer_text = example["answer"]
        gold_answer = extract_numeric_answer(answer_text)

        samples.append(
            {
                "id": f"gsm8k_{split}_{idx}",
                "question": example["question"],
                "answer": gold_answer,
                "full_answer_text": answer_text,
            }
        )

    return samples


def extract_numeric_answer(answer_text: str) -> Optional[float]:
    """
    Extract numeric answer from GSM8K answer text.
    GSM8K format: "Step-by-step solution\n#### numeric_answer"

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Numeric answer as float, or None if extraction fails
    """
    # Look for the #### marker
    if "####" in answer_text:
        answer_part = answer_text.split("####")[-1].strip()
    else:
        answer_part = answer_text.strip()

    # Remove commas and common currency symbols
    answer_part = answer_part.replace(",", "").replace("$", "").strip()

    # Try to extract a number (int or float)
    try:
        # First try as integer
        if "." not in answer_part:
            return float(int(answer_part))
        else:
            return float(answer_part)
    except (ValueError, AttributeError):
        # Try to find any number in the string
        match = re.search(r"-?\d+\.?\d*", answer_part)
        if match:
            return float(match.group())
        return None


def extract_answer_from_response(
    response_text: str, strict: bool = False
) -> Optional[float]:
    """
    Extract numeric answer from model response.

    Supports multiple formats:
    - "The answer is 42"
    - "Answer: 42"
    - "Final answer: 42"
    - "#### 42" (GSM8K format)
    - Just a number at the end

    Args:
        response_text: Model response text
        strict: If True, require explicit answer markers

    Returns:
        Extracted numeric answer as float, or None if extraction fails
    """
    if not response_text:
        return None

    response_text = response_text.strip()

    # Try to find answer with explicit markers
    patterns = [
        r"(?:final answer|answer)(?:\s*is)?(?:\s*:)?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"####\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"therefore(?:\s*,)?\s*(?:the answer is)?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            answer_str = match.group(1).replace(",", "")
            try:
                return float(answer_str)
            except ValueError:
                continue

    if not strict:
        # Look for the last number in the response
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response_text)
        if numbers:
            answer_str = numbers[-1].replace(",", "")
            try:
                return float(answer_str)
            except ValueError:
                pass

    return None
