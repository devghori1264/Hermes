from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    description: str
    datasets: list[str]
    model_version: str
    parameters: dict[str, str | float | int]
    metrics: dict[str, float]
    git_commit: str = ""
    seed: int = 42
    hardware_spec: str = "standard"

class ManifestRegistry:
    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest(self, manifest: ExperimentManifest) -> str:
        payload = {
            "experiment_id": manifest.experiment_id,
            "description": manifest.description,
            "datasets": manifest.datasets,
            "model_version": manifest.model_version,
            "parameters": manifest.parameters,
            "metrics": manifest.metrics,
            "git_commit": manifest.git_commit,
            "seed": manifest.seed,
            "hardware_spec": manifest.hardware_spec
        }
        filepath = self.registry_dir / f"{manifest.experiment_id}.json"
        filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf8")
        return str(filepath)

    def load_manifest(self, experiment_id: str) -> ExperimentManifest:
        filepath = self.registry_dir / f"{experiment_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"manifest not found: {experiment_id}")
        payload = json.loads(filepath.read_text(encoding="utf8"))
        return ExperimentManifest(
            experiment_id=str(payload.get("experiment_id")),
            description=str(payload.get("description")),
            datasets=list(payload.get("datasets", [])),
            model_version=str(payload.get("model_version")),
            parameters=dict(payload.get("parameters", {})),
            metrics={str(k): float(v) for k, v in payload.get("metrics", {}).items()},
            git_commit=str(payload.get("git_commit", "")),
            seed=int(payload.get("seed", 42)),
            hardware_spec=str(payload.get("hardware_spec", "standard"))
        )
