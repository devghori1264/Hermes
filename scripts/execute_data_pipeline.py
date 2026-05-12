from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.data.ingestion.entity_resolution import EntityResolver
from src.data.quality.drift import LeakageDetector

def main() -> None:
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    records = [
        {"id": "1", "title": "Example", "year": "2024", "source": "domain_a", "label": 1.0, "popularity": 0.8},
        {"id": "2", "title": "example", "year": "2024", "source": "domain_b", "label": 0.0, "popularity": 0.3}
    ]
    
    resolver = EntityResolver()
    resolved = resolver.resolve(records, domain="news")
    
    df = pd.DataFrame(records)
    detector = LeakageDetector(target_column="label", correlation_threshold=0.95)
    leakage_report = detector.detect(df)
    
    if leakage_report.has_leakage:
        raise ValueError(f"target leakage detected in columns: {leakage_report.leaked_columns}")
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with (output_dir / "resolved_entities.json").open("w") as f:
        import json
        payload = [{"canonical_id": r.canonical_id, "attributes": r.merged_attributes} for r in resolved]
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()
