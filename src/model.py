"""Model wrapper for Gemini API."""

import os
import time
from typing import Dict, Optional
import google.generativeai as genai


class GeminiModel:
    """Wrapper for Google Gemini API."""

    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini model.

        Args:
            model_name: Name of the Gemini model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            api_key: Google API key (if None, reads from GOOGLE_API_KEY env var)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Configure API key
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=api_key)

        # Initialize model
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
        )

        self.model = genai.GenerativeModel(
            model_name=model_name, generation_config=generation_config
        )

    def generate(
        self, prompt: str, retry_on_error: bool = True, max_retries: int = 3
    ) -> Dict:
        """
        Generate response from Gemini API.

        Args:
            prompt: Input prompt
            retry_on_error: Whether to retry on API errors
            max_retries: Maximum number of retries

        Returns:
            Dictionary with:
                - text: Generated text
                - prompt: Input prompt
                - model: Model name
                - finish_reason: Completion reason
                - error: Error message if any
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)

                # Extract text from response
                if hasattr(response, "text"):
                    text = response.text
                elif hasattr(response, "parts"):
                    text = "".join(part.text for part in response.parts)
                else:
                    text = str(response)

                # Get finish reason if available
                finish_reason = None
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        finish_reason = str(candidate.finish_reason)

                return {
                    "text": text,
                    "prompt": prompt,
                    "model": self.model_name,
                    "finish_reason": finish_reason,
                    "error": None,
                }

            except Exception as e:
                error_msg = str(e)

                # Check if we should retry
                if not retry_on_error or attempt == max_retries - 1:
                    return {
                        "text": "",
                        "prompt": prompt,
                        "model": self.model_name,
                        "finish_reason": "error",
                        "error": error_msg,
                    }

                # Wait before retrying (exponential backoff)
                wait_time = 2**attempt
                time.sleep(wait_time)

        # Should not reach here, but just in case
        return {
            "text": "",
            "prompt": prompt,
            "model": self.model_name,
            "finish_reason": "error",
            "error": "Max retries exceeded",
        }

    def get_prompt_templates(self) -> Dict[str, str]:
        """
        Get prompt templates for different methods.

        Returns:
            Dictionary mapping template names to template strings
        """
        templates = {
            "ab_cot": """Solve this math problem using Adaptive Budgeted Chain-of-Thought:

1. First, classify the difficulty: Is this problem EASY (direct computation/retrieval) or HARD (multi-hop reasoning)?
2. Choose a reasoning budget:
   - Budget-1 (EASY): 1-2 short reasoning steps + answer check
   - Budget-2 (HARD): 3-5 short reasoning steps + answer check
3. Solve within the chosen budget using only essential facts
4. Verify your answer with a targeted check (recompute, check consistency, or cross-check with options)
5. Provide final answer in format: "Final answer: [number]"

Problem: {question}

Solution:""",
            "standard_cot": """Let's solve this math problem step by step.

Problem: {question}

Solution:""",
            "direct_answer": """Solve this math problem and provide only the final numerical answer.

Problem: {question}

Answer:""",
            "static_concise_cot": """Solve this math problem step by step. Be concise and use only essential reasoning steps.

Problem: {question}

Solution:""",
        }

        return templates

    def format_prompt(self, template_name: str, question: str) -> str:
        """
        Format a prompt using a template.

        Args:
            template_name: Name of the template
            question: Question to insert

        Returns:
            Formatted prompt string
        """
        templates = self.get_prompt_templates()
        if template_name not in templates:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(templates.keys())}"
            )

        return templates[template_name].format(question=question)
