import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import hashlib

@dataclass
class PublicationManifest:
    project_name: str
    version: str
    git_commit: str
    hyperparameters: dict[str, Any]
    benchmark_results: list[dict[str, Any]]
    causal_estimates: dict[str, float]
    hardware_specs: dict[str, str]

class ManifestBuilder:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_data: dict[str, Any] = {}

    def add_section(self, section_name: str, data: Any) -> None:
        self.manifest_data[section_name] = data

    def _compute_checksum(self, filepath: Path) -> str:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_reproducibility_bundle(self, files_to_include: list[str]) -> str:
        bundle_path = self.output_dir / "publication_bundle.json"
        
        file_checksums = {}
        for filepath in files_to_include:
            p = Path(filepath)
            if p.exists():
                file_checksums[p.name] = self._compute_checksum(p)
                
        self.manifest_data["artifact_checksums"] = file_checksums
        
        with open(bundle_path, "w") as f:
            json.dump(self.manifest_data, f, indent=2)
            
        return str(bundle_path)
