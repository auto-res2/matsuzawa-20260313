"""Main orchestrator for running inference experiments."""

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments.

    Args:
        cfg: Hydra configuration
    """
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Results directory: {cfg.results_dir}")

    # Apply mode-specific overrides
    cfg = apply_mode_overrides(cfg)

    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # This is an inference-only task
    print(f"\nTask type: Inference")
    print(f"Method: {cfg.run.method.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print(f"Model: {cfg.run.model.name}")

    # Run inference directly (not as subprocess for simplicity)
    from src.inference import run_inference, validate_pilot, validate_sanity

    metrics = run_inference(cfg)

    # Run validation based on mode
    if cfg.mode == "sanity":
        validate_sanity(metrics, cfg.run.dataset.max_samples)
    elif cfg.mode == "pilot":
        validate_pilot(metrics, cfg.run.dataset.max_samples)

    print(f"\nExperiment completed: {cfg.run.run_id}")


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply mode-specific parameter overrides.

    Args:
        cfg: Original configuration

    Returns:
        Modified configuration
    """
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    mode = cfg.mode

    if mode == "sanity":
        # Sanity mode: minimal execution
        cfg.run.dataset.max_samples = 10
        cfg.wandb.mode = "online"
        # Use separate WandB namespace
        if "sanity" not in cfg.wandb.project:
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"

    elif mode == "pilot":
        # Pilot mode: 20% of full dataset (at least 50 samples)
        full_samples = cfg.run.dataset.max_samples
        pilot_samples = max(50, int(full_samples * 0.2))
        cfg.run.dataset.max_samples = pilot_samples
        cfg.wandb.mode = "online"
        # Use separate WandB namespace
        if "pilot" not in cfg.wandb.project:
            cfg.wandb.project = f"{cfg.wandb.project}-pilot"

    elif mode == "full":
        # Full mode: no changes needed
        cfg.wandb.mode = "online"

    return cfg


if __name__ == "__main__":
    main()
