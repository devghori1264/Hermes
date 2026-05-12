from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model_registry.versioning import (
    ArtifactIntegrityError,
    ModelArtifactRecord,
    ModelRegistryConfig,
    ModelVersionRegistry,
    PromotionDecision,
    RegistrySnapshot,
    RollbackRecord,
    VersionStage,
    _build_version_id,
    _compute_checksum,
)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _write_artifact(directory: Path, name: str = "model.json", content: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    payload = content or json.dumps({"weights": [0.1, 0.2], "bias": 0.01})
    path.write_text(payload, encoding="utf8")
    return path

def _build_registry(tmp_path: Path, clock: FakeClock | None = None) -> ModelVersionRegistry:
    config = ModelRegistryConfig(registry_root=tmp_path / "registry", copy_artifacts=True)
    return ModelVersionRegistry(config=config, time_fn=clock or FakeClock())

class TestComputeChecksum:
    def test_produces_consistent_digest(self, tmp_path: Path) -> None:
        artifact = _write_artifact(tmp_path, content="hello world")
        a = _compute_checksum(artifact)
        b = _compute_checksum(artifact)
        assert a == b
        assert len(a) == 64

    def test_different_content_different_digest(self, tmp_path: Path) -> None:
        artifact_a = _write_artifact(tmp_path, name="a.json", content="alpha")
        artifact_b = _write_artifact(tmp_path, name="b.json", content="beta")
        assert _compute_checksum(artifact_a) != _compute_checksum(artifact_b)

class TestBuildVersionId:
    def test_format(self) -> None:
        vid = _build_version_id("ranking", 3, "abcdef1234567890")
        assert vid == "ranking_v3_abcdef12"

    def test_truncates_checksum_to_eight(self) -> None:
        vid = _build_version_id("embedding", 1, "0" * 64)
        assert vid.endswith("_00000000")

class TestRegistration:
    def test_register_creates_record(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "training_output")
        record = registry.register(artifact, "ranking", metadata={"auc": 0.87})
        assert record.model_type == "ranking"
        assert record.stage == "registered"
        assert record.metadata["auc"] == 0.87
        assert record.checksum_sha256 != ""

    def test_register_copies_artifact(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "training_output")
        record = registry.register(artifact, "ranking")
        stored = Path(record.artifact_path)
        assert stored.exists()
        assert str(stored).startswith(str(tmp_path / "registry"))

    def test_register_without_copy(self, tmp_path: Path) -> None:
        config = ModelRegistryConfig(registry_root=tmp_path / "registry", copy_artifacts=False)
        registry = ModelVersionRegistry(config=config, time_fn=FakeClock())
        artifact = _write_artifact(tmp_path / "training_output")
        record = registry.register(artifact, "ranking")
        assert record.artifact_path == str(artifact)

    def test_register_assigns_sequential_version_ids(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "out_a", content="a"), "ranking")
        b = registry.register(_write_artifact(tmp_path / "out_b", content="b"), "ranking")
        assert "_v1_" in a.version_id
        assert "_v2_" in b.version_id

    def test_register_rejects_missing_file(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.register(tmp_path / "nonexistent.json", "ranking")

    def test_register_rejects_empty_model_type(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        with pytest.raises(ValueError, match="must not be empty"):
            registry.register(artifact, "  ")

class TestPromotion:
    def test_promote_registered_to_staging(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        decision = registry.promote(record.version_id, VersionStage.STAGING, "offline gate passed")
        assert decision.allowed is True
        assert decision.from_stage == "registered"
        assert decision.to_stage == "staging"

    def test_promote_staging_to_production(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        registry.promote(record.version_id, VersionStage.STAGING, "gate 1")
        decision = registry.promote(record.version_id, VersionStage.PRODUCTION, "ab test")
        assert decision.allowed is True

    def test_promote_production_to_archived(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        registry.promote(record.version_id, VersionStage.STAGING, "gate 1")
        registry.promote(record.version_id, VersionStage.PRODUCTION, "gate 2")
        decision = registry.promote(record.version_id, VersionStage.ARCHIVED, "superseded")
        assert decision.allowed is True

    def test_skip_promotion_stage_rejected(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        decision = registry.promote(record.version_id, VersionStage.PRODUCTION, "skip staging")
        assert decision.allowed is False
        assert "cannot promote" in decision.reason

    def test_promote_unknown_version_rejected(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        decision = registry.promote("nonexistent_v99_00000000", VersionStage.STAGING, "test")
        assert decision.allowed is False
        assert "not found" in decision.reason

    def test_production_promotion_archives_current(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a", content="model_a"), "ranking")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        registry.promote(a.version_id, VersionStage.PRODUCTION, "deploy")
        b = registry.register(_write_artifact(tmp_path / "b", content="model_b"), "ranking")
        registry.promote(b.version_id, VersionStage.STAGING, "gate")
        registry.promote(b.version_id, VersionStage.PRODUCTION, "deploy v2")
        archived_a = registry.get(a.version_id)
        assert archived_a is not None
        assert archived_a.stage == "archived"

    def test_promotion_records_reason(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        registry.promote(record.version_id, VersionStage.STAGING, "ndcg improved by 5 percent")
        updated = registry.get(record.version_id)
        assert updated is not None
        assert updated.promotion_reason == "ndcg improved by 5 percent"

class TestRollback:
    def test_rollback_swaps_production(self, tmp_path: Path) -> None:
        clock = FakeClock()
        registry = _build_registry(tmp_path, clock)
        a = registry.register(_write_artifact(tmp_path / "a", content="v1"), "ranking")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        registry.promote(a.version_id, VersionStage.PRODUCTION, "deploy")
        b = registry.register(_write_artifact(tmp_path / "b", content="v2"), "ranking")
        registry.promote(b.version_id, VersionStage.STAGING, "gate")
        registry.promote(b.version_id, VersionStage.PRODUCTION, "deploy v2")
        clock.advance(60)
        rollback = registry.rollback("ranking", a.version_id, "regression detected")
        assert rollback is not None
        assert rollback.replacement_version_id == a.version_id
        assert rollback.rolled_back_version_id == b.version_id
        current = registry.active_production("ranking")
        assert current is not None
        assert current.version_id == a.version_id

    def test_rollback_returns_none_for_unknown_version(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        result = registry.rollback("ranking", "nonexistent_v1_00000000", "test")
        assert result is None

    def test_rollback_returns_none_for_wrong_model_type(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a"), "ranking")
        result = registry.rollback("embedding", a.version_id, "wrong type")
        assert result is None

    def test_rollback_history_is_recorded(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a", content="v1"), "ranking")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        registry.promote(a.version_id, VersionStage.PRODUCTION, "deploy")
        b = registry.register(_write_artifact(tmp_path / "b", content="v2"), "ranking")
        registry.promote(b.version_id, VersionStage.STAGING, "gate")
        registry.promote(b.version_id, VersionStage.PRODUCTION, "deploy v2")
        registry.rollback("ranking", a.version_id, "regression")
        history = registry.rollback_history()
        assert len(history) == 1
        assert history[0].reason == "regression"

class TestQueries:
    def test_active_production_returns_correct_version(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a"), "ranking")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        registry.promote(a.version_id, VersionStage.PRODUCTION, "deploy")
        current = registry.active_production("ranking")
        assert current is not None
        assert current.version_id == a.version_id

    def test_active_production_returns_none_when_empty(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        assert registry.active_production("ranking") is None

    def test_list_versions_filter_by_type(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        registry.register(_write_artifact(tmp_path / "a", content="r"), "ranking")
        registry.register(_write_artifact(tmp_path / "b", content="e"), "embedding")
        ranking_versions = registry.list_versions(model_type="ranking")
        assert len(ranking_versions) == 1
        assert ranking_versions[0].model_type == "ranking"

    def test_list_versions_filter_by_stage(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a"), "ranking")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        staging = registry.list_versions(stage=VersionStage.STAGING)
        assert len(staging) == 1
        registered = registry.list_versions(stage=VersionStage.REGISTERED)
        assert len(registered) == 0

    def test_list_versions_ordered_by_registration_time(self, tmp_path: Path) -> None:
        clock = FakeClock()
        registry = _build_registry(tmp_path, clock)
        registry.register(_write_artifact(tmp_path / "a", content="first"), "ranking")
        clock.advance(10)
        registry.register(_write_artifact(tmp_path / "b", content="second"), "ranking")
        versions = registry.list_versions()
        assert versions[0].registered_at > versions[1].registered_at

class TestIntegrity:
    def test_verify_passes_for_unmodified_artifact(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        assert registry.verify(record.version_id) is True

    def test_verify_raises_on_tampered_artifact(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        stored = Path(record.artifact_path)
        stored.write_text("tampered content", encoding="utf8")
        with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
            registry.verify(record.version_id)

    def test_verify_raises_for_missing_artifact(self, tmp_path: Path) -> None:
        config = ModelRegistryConfig(registry_root=tmp_path / "registry", copy_artifacts=False)
        registry = ModelVersionRegistry(config=config, time_fn=FakeClock())
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking")
        artifact.unlink()
        with pytest.raises(FileNotFoundError):
            registry.verify(record.version_id)

    def test_verify_raises_for_unknown_version(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        with pytest.raises(FileNotFoundError, match="not in registry"):
            registry.verify("nonexistent_v1_00000000")

class TestSnapshot:
    def test_snapshot_reflects_state(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a", content="r1"), "ranking")
        registry.register(_write_artifact(tmp_path / "b", content="e1"), "embedding")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        registry.promote(a.version_id, VersionStage.PRODUCTION, "deploy")
        snap = registry.snapshot()
        assert snap.total_versions == 2
        assert "ranking" in snap.model_types
        assert "embedding" in snap.model_types
        assert snap.production_versions["ranking"] == a.version_id
        assert snap.rollback_count == 0

    def test_snapshot_tracks_rollbacks(self, tmp_path: Path) -> None:
        registry = _build_registry(tmp_path)
        a = registry.register(_write_artifact(tmp_path / "a", content="v1"), "ranking")
        registry.promote(a.version_id, VersionStage.STAGING, "gate")
        registry.promote(a.version_id, VersionStage.PRODUCTION, "deploy")
        b = registry.register(_write_artifact(tmp_path / "b", content="v2"), "ranking")
        registry.promote(b.version_id, VersionStage.STAGING, "gate")
        registry.promote(b.version_id, VersionStage.PRODUCTION, "deploy v2")
        registry.rollback("ranking", a.version_id, "regression")
        snap = registry.snapshot()
        assert snap.rollback_count == 1

class TestPersistence:
    def test_index_survives_reload(self, tmp_path: Path) -> None:
        clock = FakeClock()
        config = ModelRegistryConfig(registry_root=tmp_path / "registry")
        registry = ModelVersionRegistry(config=config, time_fn=clock)
        artifact = _write_artifact(tmp_path / "out")
        record = registry.register(artifact, "ranking", metadata={"seed": 42})
        registry.promote(record.version_id, VersionStage.STAGING, "gate")

        reloaded = ModelVersionRegistry(config=config, time_fn=clock)
        restored = reloaded.get(record.version_id)
        assert restored is not None
        assert restored.stage == "staging"
        assert restored.metadata["seed"] == 42

    def test_corrupted_index_starts_empty(self, tmp_path: Path) -> None:
        config = ModelRegistryConfig(registry_root=tmp_path / "registry")
        config.registry_root.mkdir(parents=True, exist_ok=True)
        (config.registry_root / config.index_filename).write_text("not json", encoding="utf8")
        registry = ModelVersionRegistry(config=config, time_fn=FakeClock())
        assert registry.snapshot().total_versions == 0

    def test_sequence_survives_reload(self, tmp_path: Path) -> None:
        config = ModelRegistryConfig(registry_root=tmp_path / "registry")
        registry = ModelVersionRegistry(config=config, time_fn=FakeClock())
        registry.register(_write_artifact(tmp_path / "a", content="r1"), "ranking")
        registry.register(_write_artifact(tmp_path / "b", content="r2"), "ranking")

        reloaded = ModelVersionRegistry(config=config, time_fn=FakeClock())
        third = reloaded.register(_write_artifact(tmp_path / "c", content="r3"), "ranking")
        assert "_v3_" in third.version_id
