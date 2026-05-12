from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from typing import Any
import urllib.request
import wave

from src.data.ingestion.base import stable_item_id
from src.domain.schema import AssetRef
from src.features.feature_store import FeatureKey, FeatureStore, build_feature_record

import numpy as np

@dataclass(frozen=True)
class EncoderRequest:
    item_id: str | None
    domain: str
    title: str
    overview: str | None = None
    assets: dict[str, list[AssetRef]] = field(default_factory=dict)


@dataclass(frozen=True)
class EncoderOutput:
    embedding: list[float]
    attributes: list[str]
    metadata: dict[str, Any]


class BaseEncoder:
    name: str
    modality: str

    def encode(self, request: EncoderRequest) -> EncoderOutput:
        raise NotImplementedError


def _load_binary(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        with urllib.request.urlopen(uri, timeout=20) as response:
            return response.read()
    with open(uri, "rb") as handle:
        return handle.read()


def _load_image(uri: str):
    from PIL import Image

    payload = _load_binary(uri)
    return Image.open(io.BytesIO(payload)).convert("RGB")


def _load_wav(uri: str) -> tuple[np.ndarray, int]:
    payload = _load_binary(uri)
    with wave.open(io.BytesIO(payload), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError("only 16bit pcm wav is supported")
    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    return signal, sample_rate


def _safe_std(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.std(signal))


def _coerce_dim(values: list[float], dim: int) -> list[float]:
    if not values:
        return [0.0] * dim
    if len(values) >= dim:
        return values[:dim]
    out = list(values)
    while len(out) < dim:
        out.extend(values[: min(len(values), dim - len(out))])
    return out[:dim]


def hash_to_vector(payload: str, dim: int) -> list[float]:
    digest = hashlib.sha256(payload.encode("utf8")).digest()
    values: list[float] = []
    for i in range(dim):
        values.append(digest[i % len(digest)] / 255.0)
    return values


def tokenize(payload: str, limit: int) -> list[str]:
    tokens = [token for token in payload.lower().split() if token]
    return tokens[:limit]


class DeterministicHashEncoder(BaseEncoder):
    def __init__(self, name: str, modality: str, dim: int = 32, token_limit: int = 12) -> None:
        self.name = name
        self.modality = modality
        self._dim = dim
        self._token_limit = token_limit

    def encode(self, request: EncoderRequest) -> EncoderOutput:
        if self.modality == "text":
            text = request.title if request.overview is None else f"{request.title} {request.overview}"
        else:
            assets = request.assets.get(self.modality, [])
            text = " ".join(asset.uri for asset in assets) if assets else request.title
        embedding = hash_to_vector(text, self._dim)
        attributes = tokenize(text, self._token_limit)
        metadata = {
            "encoder": self.name,
            "modality": self.modality,
            "dim": self._dim,
            "token_limit": self._token_limit,
        }
        return EncoderOutput(embedding=embedding, attributes=attributes, metadata=metadata)


class TextSentenceEncoder(BaseEncoder):
    def __init__(
        self,
        name: str = "text_sentence_encoder",
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
    ) -> None:
        self.name = name
        self.modality = "text"
        self._model_id = model_id
        self._dim = dim
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id)

    def _mean_pool(self, hidden, mask) -> list[float]:
        mask_expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = (hidden * mask_expanded).sum(1)
        counts = mask_expanded.sum(1).clamp(min=1e-9)
        vector = (summed / counts).detach().cpu().numpy()[0].tolist()
        return _coerce_dim([float(x) for x in vector], self._dim)

    def encode(self, request: EncoderRequest) -> EncoderOutput:
        text = request.title if request.overview is None else f"{request.title}. {request.overview}"
        attributes = tokenize(text, 20)
        self._ensure_model()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = self._model(**inputs)
        embedding = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return EncoderOutput(
            embedding=embedding,
            attributes=attributes,
            metadata={"encoder": self.name, "modality": self.modality, "model_id": self._model_id},
        )


class Florence2ImageEncoder(BaseEncoder):
    def __init__(
        self,
        name: str = "florence2_image_encoder",
        model_id: str = "microsoft/Florence-2-base",
        task_prompt: str = "<DETAILED_CAPTION>",
        dim: int = 384,
    ) -> None:
        self.name = name
        self.modality = "image"
        self._model_id = model_id
        self._task_prompt = task_prompt
        self._dim = dim
        self._processor = None
        self._model = None
        self._text_encoder = TextSentenceEncoder(
            name=f"{name}_text_projection",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            dim=dim,
        )

    def _ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, trust_remote_code=True)

    def _caption(self, image) -> str:
        self._ensure_model()
        inputs = self._processor(text=self._task_prompt, images=image, return_tensors="pt")
        generated = self._model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=128,
            num_beams=3,
        )
        decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            decoded,
            task=self._task_prompt,
            image_size=(image.width, image.height),
        )
        payload = parsed.get(self._task_prompt)
        if isinstance(payload, str):
            return payload
        return str(payload)

    def encode(self, request: EncoderRequest) -> EncoderOutput:
        assets = request.assets.get("image", [])
        if not assets:
            caption = request.title
            vector = hash_to_vector(caption, self._dim)
            return EncoderOutput(
                embedding=vector,
                attributes=tokenize(caption, 20),
                metadata={"encoder": self.name, "modality": self.modality, "model_id": self._model_id, "mode": "fallback_no_asset"},
            )

        image = _load_image(assets[0].uri)
        caption = self._caption(image)
        text_out = self._text_encoder.encode(
            EncoderRequest(
                item_id=request.item_id,
                domain=request.domain,
                title=caption,
                overview=None,
                assets=request.assets,
            )
        )
        return EncoderOutput(
            embedding=text_out.embedding,
            attributes=tokenize(caption, 20),
            metadata={"encoder": self.name, "modality": self.modality, "model_id": self._model_id, "caption": caption},
        )


class HubertAudioEncoder(BaseEncoder):
    def __init__(self, name: str = "hubert_audio_encoder", dim: int = 768, model_id: str = "facebook/hubert-base-ls960") -> None:
        self.name = name
        self.modality = "audio"
        self._dim = dim
        self._model_id = model_id
        self._processor = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        from transformers import Wav2Vec2FeatureExtractor, HubertModel
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self._model_id)
        self._model = HubertModel.from_pretrained(self._model_id)

    def encode(self, request: EncoderRequest) -> EncoderOutput:
        assets = request.assets.get("audio", [])
        if not assets:
            text = request.title
            return EncoderOutput(
                embedding=hash_to_vector(text, self._dim),
                attributes=tokenize(text, 12),
                metadata={"encoder": self.name, "modality": self.modality, "mode": "fallback_no_asset"},
            )
        signal, sample_rate = _load_wav(assets[0].uri)
        if sample_rate != 16000:
            import librosa
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        self._ensure_model()
        import torch
        inputs = self._processor(signal, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze().tolist()
        
        attributes = [f"sr_{sample_rate}", f"len_{signal.size}", f"hubert_features_{len(embedding)}"]
        return EncoderOutput(
            embedding=_coerce_dim(embedding, self._dim),
            attributes=attributes,
            metadata={"encoder": self.name, "modality": self.modality, "model_id": self._model_id},
        )


class VideoMaeFrameEncoder(BaseEncoder):
    def __init__(self, name: str = "videomae_frame_encoder", dim: int = 768, model_id: str = "MCG-NJU/videomae-base") -> None:
        self.name = name
        self.modality = "video"
        self._dim = dim
        self._model_id = model_id
        self._processor = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        self._processor = VideoMAEImageProcessor.from_pretrained(self._model_id)
        self._model = VideoMAEModel.from_pretrained(self._model_id)

    def _sample_frames(self, uri: str, frame_count: int = 16) -> list[Any]:
        from PIL import Image
        import cv2

        if uri.startswith("http://") or uri.startswith("https://"):
            raise ValueError("remote video uri is not supported")
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise ValueError("video cannot be opened")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError("video has no frames")
        points = np.linspace(0, max(total - 1, 0), num=frame_count, dtype=int).tolist()
        frames: list[Any] = []
        for point in points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(point))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        cap.release()
        
        while len(frames) > 0 and len(frames) < frame_count:
            frames.append(frames[-1])
            
        return frames

    def encode(self, request: EncoderRequest) -> EncoderOutput:
        assets = request.assets.get("video", [])
        if not assets:
            return EncoderOutput(
                embedding=hash_to_vector(request.title, self._dim),
                attributes=tokenize(request.title, 12),
                metadata={"encoder": self.name, "modality": self.modality, "mode": "fallback_no_asset"},
            )
        frames = self._sample_frames(assets[0].uri, frame_count=16)
        if not frames or len(frames) != 16:
            return EncoderOutput(
                embedding=hash_to_vector(request.title, self._dim),
                attributes=tokenize(request.title, 12),
                metadata={"encoder": self.name, "modality": self.modality, "mode": "fallback_empty_video"},
            )
            
        self._ensure_model()
        import torch
        inputs = self._processor(list(frames), return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        
        return EncoderOutput(
            embedding=_coerce_dim(embedding, self._dim),
            attributes=tokenize(request.title, 24),
            metadata={"encoder": self.name, "modality": self.modality, "model_id": self._model_id},
        )


@dataclass(frozen=True)
class MultimodalBundle:
    embeddings: dict[str, list[float]]
    attributes: dict[str, list[str]]
    fused_embedding: list[float]
    provenance: dict[str, Any]


class MultimodalPipeline:
    def __init__(
        self,
        encoders: list[BaseEncoder],
        store: FeatureStore | None = None,
        feature_name: str = "multimodal_embedding",
        feature_version: str = "v1",
    ) -> None:
        self._encoders = encoders
        self._store = store
        self._feature_name = feature_name
        self._feature_version = feature_version

    def encode(self, request: EncoderRequest) -> MultimodalBundle:
        embeddings: dict[str, list[float]] = {}
        attributes: dict[str, list[str]] = {}
        provenance: dict[str, Any] = {
            "feature_name": self._feature_name,
            "feature_version": self._feature_version,
            "encoders": [],
        }

        entity_id = request.item_id or stable_item_id(request.domain, request.title.lower())

        for encoder in self._encoders:
            cache_key = None
            if self._store is not None:
                cache_key = FeatureKey(
                    entity_type="item",
                    entity_id=entity_id,
                    feature_name=self._feature_name,
                    feature_version=self._feature_version,
                    modality=encoder.modality,
                )
                cached = self._store.get(cache_key)
                if cached is not None:
                    embeddings[encoder.modality] = cached.vector
                    attributes[encoder.modality] = cached.metadata.get("attributes", [])
                    provenance["encoders"].append(cached.metadata.get("encoder"))
                    continue

            output = encoder.encode(request)
            embeddings[encoder.modality] = output.embedding
            attributes[encoder.modality] = output.attributes
            provenance["encoders"].append(output.metadata.get("encoder"))

            if self._store is not None and cache_key is not None:
                record = build_feature_record(
                    cache_key,
                    output.embedding,
                    {
                        "encoder": output.metadata.get("encoder"),
                        "attributes": output.attributes,
                    },
                )
                self._store.put(record)

        fused_embedding = self._fuse_embeddings(embeddings)
        return MultimodalBundle(
            embeddings=embeddings,
            attributes=attributes,
            fused_embedding=fused_embedding,
            provenance=provenance,
        )

    def _fuse_embeddings(self, embeddings: dict[str, list[float]]) -> list[float]:
        if not embeddings:
            return []
        dims = {len(vector) for vector in embeddings.values() if vector}
        if len(dims) != 1:
            return []
        dim = dims.pop()
        totals = [0.0] * dim
        for vector in embeddings.values():
            for index, value in enumerate(vector):
                totals[index] += float(value)
        return [value / len(embeddings) for value in totals]
