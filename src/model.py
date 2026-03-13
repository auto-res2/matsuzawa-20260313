"""Model definitions - not used for inference-only tasks."""

# This file is not used for the current inference-only experiment.
# The experiment uses OpenAI API models directly without custom model definitions.
# This file is included to satisfy repository structure requirements.


def get_model():
    """Placeholder model getter."""
    raise NotImplementedError("This experiment uses OpenAI API models via inference.py")
