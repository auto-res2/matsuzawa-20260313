"""Main orchestration script for inference experiments."""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Orchestrate a single inference run.

    This script:
    1. Loads configuration using Hydra
    2. Invokes inference.py as a subprocess
    3. Handles mode-specific overrides
    """
    print("=" * 80)
    print(f"Starting run: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print("=" * 80)

    # Ensure results directory exists
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # This is an inference-only experiment, so we call inference.py
    # Build command to run inference.py with the same config
    cmd = ["uv", "run", "python", "-u", "-m", "src.inference"]

    # Pass through Hydra overrides
    # We need to reconstruct the command line args that Hydra received
    for override in sys.argv[1:]:
        if "=" in override:
            cmd.append(override)

    print(f"\nRunning inference subprocess...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run inference as subprocess
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False,  # Stream output to console
        )

        print("\n" + "=" * 80)
        print(f"Run completed successfully: {cfg.run.run_id}")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"Run failed: {cfg.run.run_id}")
        print(f"Exit code: {e.returncode}")
        print("=" * 80)
        sys.exit(e.returncode)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Run interrupted by user")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
