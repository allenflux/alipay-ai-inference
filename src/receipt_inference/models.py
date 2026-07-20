"""Model registry and deployment-path resolution.

Keep model-specific filesystem conventions here so adding another model does
not make the command-line entry point conditional and difficult to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint_relative_path: Path
    description: str

    def default_checkpoint(self) -> Path:
        return PROJECT_ROOT / self.checkpoint_relative_path


MODEL_SPECS: dict[str, ModelSpec] = {
    "receipt_lrcnn_v1": ModelSpec(
        name="receipt_lrcnn_v1",
        checkpoint_relative_path=Path("checkpoints/receipt_lrcnn_v1/best.pt"),
        description="Five-field transfer receipt LRCNN detector",
    ),
}


def resolve_checkpoint(model_name: str, override: Path | None = None) -> Path:
    """Resolve and validate the checkpoint selected for one inference run."""
    try:
        spec = MODEL_SPECS[model_name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {model_name!r}; available models: {choices}") from error

    checkpoint = override.expanduser() if override is not None else spec.default_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Copy best.pt to: {spec.default_checkpoint()}"
        )
    return checkpoint.resolve()

