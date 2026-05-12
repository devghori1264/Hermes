from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import load_settings
from src.features.feature_store import FeatureStore
from src.features.multimodal_pipeline import (
    HubertAudioEncoder,
    DeterministicHashEncoder,
    EncoderRequest,
    Florence2ImageEncoder,
    MultimodalPipeline,
    TextSentenceEncoder,
    VideoMaeFrameEncoder,
)

@dataclass(frozen=True)
class MultimodalFeatureVector:
    text_summary: str
    image_attributes: list[str]
    embeddings: dict[str, list[float]]
    fused_embedding: list[float]
    provenance: dict[str, Any]


class MultimodalFeatureService:
    def __init__(self, model_name: str = "multimodal_baseline_v1", store: FeatureStore | None = None) -> None:
        settings = load_settings()
        self.model_name = model_name
        encoders = self._build_encoders(
            use_model_encoders=settings.enable_model_encoders,
            text_model_id=settings.text_encoder_model_id,
            florence_model_id=settings.florence_model_id,
        )
        self._pipeline = MultimodalPipeline(
            encoders=encoders,
            store=store,
        )
        self._active_mode = "model_backed" if settings.enable_model_encoders else "deterministic_hash"

    def _build_encoders(self, use_model_encoders: bool, text_model_id: str, florence_model_id: str) -> list:
        if not use_model_encoders:
            return [
                DeterministicHashEncoder(name="baseline_text", modality="text"),
                DeterministicHashEncoder(name="baseline_image", modality="image"),
                DeterministicHashEncoder(name="baseline_audio", modality="audio"),
                DeterministicHashEncoder(name="baseline_video", modality="video"),
            ]
        try:
            return [
                TextSentenceEncoder(name="sentence_text", model_id=text_model_id),
                Florence2ImageEncoder(name="florence2_image", model_id=florence_model_id),
                HubertAudioEncoder(name="signal_audio"),
                VideoMaeFrameEncoder(name="frame_video"),
            ]
        except Exception:
            return [
                DeterministicHashEncoder(name="baseline_text", modality="text"),
                DeterministicHashEncoder(name="baseline_image", modality="image"),
                DeterministicHashEncoder(name="baseline_audio", modality="audio"),
                DeterministicHashEncoder(name="baseline_video", modality="video"),
            ]

    def extract(
        self,
        title: str,
        overview: str | None = None,
        item_id: str | None = None,
        domain: str = "movies",
        assets: dict[str, list[Any]] | None = None,
    ) -> MultimodalFeatureVector:
        summary = overview.strip() if overview else title
        request = EncoderRequest(
            item_id=item_id,
            domain=domain,
            title=title,
            overview=overview,
            assets=assets or {},
        )
        bundle = self._pipeline.encode(request)
        image_attributes = bundle.attributes.get("image", [])
        return MultimodalFeatureVector(
            text_summary=summary,
            image_attributes=image_attributes,
            embeddings=bundle.embeddings,
            fused_embedding=bundle.fused_embedding,
            provenance={
                "model": self.model_name,
                "encoders": bundle.provenance.get("encoders", []),
                "feature_version": bundle.provenance.get("feature_version"),
                "mode": self._active_mode,
            },
        )
