from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any


class VersionStage(str, Enum):
    REGISTERED = "registered"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


_VALID_PROMOTIONS: dict[VersionStage, VersionStage] = {
    VersionStage.REGISTERED: VersionStage.STAGING,
    VersionStage.STAGING: VersionStage.PRODUCTION,
    VersionStage.PRODUCTION: VersionStage.ARCHIVED,
}


class ArtifactIntegrityError(Exception):
    pass
@dataclass(frozen=True)
class ModelArtifactRecord:
    version_id: str
    model_type: str
    artifact_path: str
    checksum_sha256: str
    stage: str
    registered_at: float
    promoted_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    promotion_reason: str = ""


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    version_id: str
    from_stage: str
    to_stage: str
    reason: str


@dataclass(frozen=True)
class RollbackRecord:
    rolled_back_version_id: str
    replacement_version_id: str
    reason: str
    timestamp: float


@dataclass(frozen=True)
class RegistrySnapshot:
    total_versions: int
    versions_by_stage: dict[str, int]
    model_types: list[str]
    production_versions: dict[str, str]
    rollback_count: int


@dataclass(frozen=True)
class ModelRegistryConfig:
    registry_root: Path = field(default_factory=lambda: Path("artifacts/model_registry"))
    index_filename: str = "version_index.json"
    copy_artifacts: bool = True


def _compute_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_version_id(model_type: str, sequence: int, checksum: str) -> str:
    return f"{model_type}_v{sequence}_{checksum[:8]}"

class ModelVersionRegistry:
    def __init__(
        self,
        config: ModelRegistryConfig | None = None,
        *,
        time_fn: Any | None = None,
    ) -> None:
        self._config = config or ModelRegistryConfig()
        self._time_fn = time_fn or time.time
        self._records: dict[str, ModelArtifactRecord] = {}
        self._rollbacks: list[RollbackRecord] = []
        self._next_sequence: dict[str, int] = {}
        self._config.registry_root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._config.registry_root / self._config.index_filename
        self._load_index()

    @property
    def config(self) -> ModelRegistryConfig:
        return self._config

    def register(
        self,
        artifact_path: Path,
        model_type: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifactRecord:

        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_path}")
        model_type = model_type.strip().lower()
        if not model_type:
            raise ValueError("model_type must not be empty")

        checksum = _compute_checksum(artifact_path)
        sequence = self._next_sequence.get(model_type, 1)
        version_id = _build_version_id(model_type, sequence, checksum)
        self._next_sequence[model_type] = sequence + 1

        if self._config.copy_artifacts:
            dest_dir = self._config.registry_root / model_type / version_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / artifact_path.name
            shutil.copy2(str(artifact_path), str(dest_path))
            stored_path = str(dest_path)
        else:
            stored_path = str(artifact_path)

        record = ModelArtifactRecord(
            version_id=version_id,
            model_type=model_type,
            artifact_path=stored_path,
            checksum_sha256=checksum,
            stage=VersionStage.REGISTERED.value,
            registered_at=float(self._time_fn()),
            metadata=dict(metadata) if metadata else {},
        )
        self._records[version_id] = record
        self._persist_index()
        return record

    def promote(
        self,
        version_id: str,
        to_stage: VersionStage,
        reason: str,
    ) -> PromotionDecision:

        record = self._records.get(version_id)
        if record is None:
            return PromotionDecision(
                allowed=False,
                version_id=version_id,
                from_stage="unknown",
                to_stage=to_stage.value,
                reason=f"version {version_id} not found in registry",
            )

        current_stage = VersionStage(record.stage)
        expected_next = _VALID_PROMOTIONS.get(current_stage)
        if expected_next is None or expected_next != to_stage:
            return PromotionDecision(
                allowed=False,
                version_id=version_id,
                from_stage=current_stage.value,
                to_stage=to_stage.value,
                reason=f"cannot promote from {current_stage.value} to {to_stage.value}",
            )

        if to_stage == VersionStage.PRODUCTION:
            self._archive_current_production(record.model_type, reason=f"replaced by {version_id}")

        promoted = ModelArtifactRecord(
            version_id=record.version_id,
            model_type=record.model_type,
            artifact_path=record.artifact_path,
            checksum_sha256=record.checksum_sha256,
            stage=to_stage.value,
            registered_at=record.registered_at,
            promoted_at=float(self._time_fn()),
            metadata=record.metadata,
            promotion_reason=reason,
        )
        self._records[version_id] = promoted
        self._persist_index()
        return PromotionDecision(
            allowed=True,
            version_id=version_id,
            from_stage=current_stage.value,
            to_stage=to_stage.value,
            reason=reason,
        )

    def rollback(
        self,
        model_type: str,
        replacement_version_id: str,
        reason: str,
    ) -> RollbackRecord | None:

        model_type = model_type.strip().lower()
        replacement = self._records.get(replacement_version_id)
        if replacement is None:
            return None
        if replacement.model_type != model_type:
            return None

        current_prod = self.active_production(model_type)
        if current_prod is not None:
            self._force_stage(current_prod.version_id, VersionStage.ARCHIVED, reason=f"rolled back: {reason}")

        self._force_stage(replacement_version_id, VersionStage.PRODUCTION, reason=f"rollback target: {reason}")

        rollback_record = RollbackRecord(
            rolled_back_version_id=current_prod.version_id if current_prod else "",
            replacement_version_id=replacement_version_id,
            reason=reason,
            timestamp=float(self._time_fn()),
        )
        self._rollbacks.append(rollback_record)
        self._persist_index()
        return rollback_record

    def get(self, version_id: str) -> ModelArtifactRecord | None:
        return self._records.get(version_id)

    def active_production(self, model_type: str) -> ModelArtifactRecord | None:
        model_type = model_type.strip().lower()
        for record in self._records.values():
            if record.model_type == model_type and record.stage == VersionStage.PRODUCTION.value:
                return record
        return None

    def list_versions(
        self,
        model_type: str | None = None,
        stage: VersionStage | None = None,
    ) -> list[ModelArtifactRecord]:
        results: list[ModelArtifactRecord] = []
        for record in self._records.values():
            if model_type is not None and record.model_type != model_type.strip().lower():
                continue
            if stage is not None and record.stage != stage.value:
                continue
            results.append(record)
        results.sort(key=lambda r: r.registered_at, reverse=True)
        return results

    def rollback_history(self, model_type: str | None = None) -> list[RollbackRecord]:
        if model_type is None:
            return list(self._rollbacks)
        model_type = model_type.strip().lower()
        return [
            r for r in self._rollbacks
            if any(
                rec.model_type == model_type
                for vid in (r.rolled_back_version_id, r.replacement_version_id)
                if (rec := self._records.get(vid)) is not None
            )
        ]

    def snapshot(self) -> RegistrySnapshot:
        stage_counts: dict[str, int] = {}
        model_types_set: set[str] = set()
        production_map: dict[str, str] = {}
        for record in self._records.values():
            stage_counts[record.stage] = stage_counts.get(record.stage, 0) + 1
            model_types_set.add(record.model_type)
            if record.stage == VersionStage.PRODUCTION.value:
                production_map[record.model_type] = record.version_id
        return RegistrySnapshot(
            total_versions=len(self._records),
            versions_by_stage=stage_counts,
            model_types=sorted(model_types_set),
            production_versions=production_map,
            rollback_count=len(self._rollbacks),
        )

    def verify(self, version_id: str) -> bool:
        record = self._records.get(version_id)
        if record is None:
            raise FileNotFoundError(f"version {version_id} not in registry")
        artifact_path = Path(record.artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact file missing: {artifact_path}")
        current_checksum = _compute_checksum(artifact_path)
        if current_checksum != record.checksum_sha256:
            raise ArtifactIntegrityError(
                f"checksum mismatch for {version_id}: "
                f"expected {record.checksum_sha256}, got {current_checksum}"
            )
        return True

    def _archive_current_production(self, model_type: str, *, reason: str) -> None:
        current = self.active_production(model_type)
        if current is None:
            return
        self._force_stage(current.version_id, VersionStage.ARCHIVED, reason=reason)

    def _force_stage(self, version_id: str, stage: VersionStage, *, reason: str) -> None:
        record = self._records.get(version_id)
        if record is None:
            return
        updated = ModelArtifactRecord(
            version_id=record.version_id,
            model_type=record.model_type,
            artifact_path=record.artifact_path,
            checksum_sha256=record.checksum_sha256,
            stage=stage.value,
            registered_at=record.registered_at,
            promoted_at=float(self._time_fn()),
            metadata=record.metadata,
            promotion_reason=reason,
        )
        self._records[version_id] = updated

    def _persist_index(self) -> None:
        payload = {
            "records": {vid: asdict(rec) for vid, rec in self._records.items()},
            "rollbacks": [asdict(rb) for rb in self._rollbacks],
            "sequences": dict(self._next_sequence),
        }
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf8",
        )

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf8"))
        except (json.JSONDecodeError, OSError):
            return

        for vid, rec_dict in raw.get("records", {}).items():
            self._records[vid] = ModelArtifactRecord(
                version_id=str(rec_dict["version_id"]),
                model_type=str(rec_dict["model_type"]),
                artifact_path=str(rec_dict["artifact_path"]),
                checksum_sha256=str(rec_dict["checksum_sha256"]),
                stage=str(rec_dict["stage"]),
                registered_at=float(rec_dict["registered_at"]),
                promoted_at=float(rec_dict["promoted_at"]) if rec_dict.get("promoted_at") is not None else None,
                metadata=dict(rec_dict.get("metadata", {})),
                promotion_reason=str(rec_dict.get("promotion_reason", "")),
            )
        for rb_dict in raw.get("rollbacks", []):
            self._rollbacks.append(RollbackRecord(
                rolled_back_version_id=str(rb_dict["rolled_back_version_id"]),
                replacement_version_id=str(rb_dict["replacement_version_id"]),
                reason=str(rb_dict["reason"]),
                timestamp=float(rb_dict["timestamp"]),
            ))
        self._next_sequence = {str(k): int(v) for k, v in raw.get("sequences", {}).items()}
