"""
Checkpointing for long-running experiments.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime


class Checkpointer:
    """Save and load experiment checkpoints."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: dict, phase: int = 0):
        """Save checkpoint."""
        checkpoint = {
            'name': name,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'data': data,
        }

        filepath = self.checkpoint_dir / f"{name}_phase{phase}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except TypeError:
            # Fall back to pickle for non-JSON-serializable
            filepath = self.checkpoint_dir / f"{name}_phase{phase}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)

        print(f"✅ Checkpoint saved: {filepath}")
        return filepath

    def load(self, name: str, phase: int = 0) -> dict:
        """Load checkpoint if exists."""
        # Try JSON first
        filepath = self.checkpoint_dir / f"{name}_phase{phase}.json"
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)

        # Try pickle
        filepath = self.checkpoint_dir / f"{name}_phase{phase}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        return None

    def exists(self, name: str, phase: int = 0) -> bool:
        """Check if checkpoint exists."""
        return (
            (self.checkpoint_dir / f"{name}_phase{phase}.json").exists() or
            (self.checkpoint_dir / f"{name}_phase{phase}.pkl").exists()
        )
