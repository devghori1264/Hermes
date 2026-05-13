from src.features.feature_store import InMemoryFeatureStore
from src.features.multimodal_pipeline import DeterministicHashEncoder, EncoderRequest, MultimodalPipeline
from src.features.multimodal_features import MultimodalFeatureService


def test_pipeline_caches_embeddings() -> None:
    store = InMemoryFeatureStore()
    pipeline = MultimodalPipeline(
        encoders=[DeterministicHashEncoder(name="text", modality="text", dim=8)],
        store=store,
    )
    request = EncoderRequest(item_id="item_1", domain="movies", title="Example")
    first = pipeline.encode(request)
    second = pipeline.encode(request)

    assert first.fused_embedding == second.fused_embedding
    assert store.count() == 1


def test_model_backed_multimodal_service_includes_florence_encoder(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MODEL_ENCODERS", "1")
    service = MultimodalFeatureService()

    encoder_types = [type(encoder).__name__ for encoder in service._pipeline._encoders]

    assert "TextSentenceEncoder" in encoder_types
    assert "Florence2ImageEncoder" in encoder_types
    assert "HubertAudioEncoder" in encoder_types
    assert "VideoMaeFrameEncoder" in encoder_types
