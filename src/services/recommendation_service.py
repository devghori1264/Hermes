from __future__ import annotations

from pathlib import Path

from src.config import load_settings
from src.data.catalog_repository import CatalogRepository
from src.domain.models import Candidate, RankedItem, RecommendationContext
from src.features.feature_store import build_feature_store
from src.features.multimodal_features import MultimodalFeatureService
from src.features.text_features import TextFeatureService
from src.policy.diversity import DiversityOptimizer
from src.policy.reranker import PolicyReranker
from src.ranking.pipeline import RankingBudget, RankingPipeline
from src.services.vector_index_service import VectorIndexService
from src.training.ranking_model import LinearRankingModel, load_ranking_model
from src.generative.llm_ranking import LLMShadowRanker
from src.retrieval.knowledge_graph import KnowledgeGraph
from src.retrieval.gnn_retrieval import GraphRetriever
from src.serving.cache_manager import CacheManager, build_cache_key
from src.model_registry.versioning import ModelVersionRegistry
from src.retrieval.cold_start import ColdStartRetrievalStrategy


class RecommendationService:
    def __init__(self, data_path: Path, cache_manager: CacheManager | None = None, model_registry: ModelVersionRegistry | None = None) -> None:
        settings = load_settings()
        snapshot_path = Path(settings.dataset_snapshot_path)
        self.repo = CatalogRepository(data_path, snapshot_path=snapshot_path)
        feature_store = build_feature_store(settings.feature_store_path)
        self.text_features = TextFeatureService()
        self.pipeline = RankingPipeline(RankingBudget())
        self.policy = PolicyReranker()
        self.diversity = DiversityOptimizer()
        self.multimodal = MultimodalFeatureService(store=feature_store)
        self.vector_index = VectorIndexService(Path(settings.vector_index_path)) if settings.vector_index_path else None
        self._fallback_model_path = settings.ranking_model_path
        self.llm_ranker = LLMShadowRanker(shadow_mode=True)
        self.knowledge_graph = KnowledgeGraph()
        kg_path = data_path.parent / "knowledge_graph.pkl"
        if not self.knowledge_graph.load(str(kg_path)):
            self.knowledge_graph.build_from_catalog(self.repo.load_catalog())
            self.knowledge_graph.save(str(kg_path))
        self.graph_retriever = GraphRetriever(self.knowledge_graph)
        self.cache_manager = cache_manager
        self.model_registry = model_registry
        self.cold_start = ColdStartRetrievalStrategy()
        
        from src.retrieval.index import CrossDomainAdapter
        self.domain_adapter = CrossDomainAdapter(
            source_domain="unknown", 
            target_domain="movies", 
            adapter_matrix=[[1.0 if i == j else 0.0 for j in range(384)] for i in range(384)]
        )
        
        from src.serving.bandits import ThompsonSamplingBandit
        self.bandit_policy = ThompsonSamplingBandit()

    def trained_ranker(self, context: RecommendationContext | None = None) -> LinearRankingModel | None:
        if self.model_registry is not None:
            model_version = context.experiment_group if context else "production"
            if model_version == "default" or model_version == "control":
                model_version = "production"
                
            prod_record = None
            if model_version != "production":
                prod_record = self.model_registry.get(model_version)
            if prod_record is None:
                prod_record = self.model_registry.active_production("ranking")
                
            if prod_record is not None:
                path = Path(prod_record.artifact_path)
                if path.exists():
                    try:
                        return load_ranking_model(path)
                    except Exception:
                        pass
        
        if not self._fallback_model_path:
            return None
        path = Path(self._fallback_model_path)
        if not path.exists():
            return None
        return load_ranking_model(path)

    def _title_for_item_id(self, item_id: str, data) -> str | None:
        if item_id.isdigit():
            index = int(item_id)
            if 0 <= index < len(data):
                return str(data.iloc[index]["movie_title"])
        normalized = str(item_id).strip().lower()
        matches = data.loc[data["movie_title"] == normalized]
        if not matches.empty:
            return str(matches.iloc[0]["movie_title"])
        return None

    def _merge_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for candidate in candidates:
            existing = merged.get(candidate.item_id)
            if existing is None or candidate.score > existing.score:
                merged[candidate.item_id] = candidate
        return list(merged.values())

    def _vector_candidates(self, query_title: str, data, context: RecommendationContext | None = None) -> list[Candidate]:
        if self.vector_index is None or not self.vector_index.available():
            return []
        
        target_domain = context.domain if context else "movies"
        cache_key = build_cache_key("vector_candidates", f"{query_title}_{target_domain}")
        if self.cache_manager is not None:
            cached = self.cache_manager.get("retrieval_vector", cache_key)
            if cached is not None:
                return cached

        try:
            from src.retrieval.types import RetrievalQuery
            query_vector = self.multimodal.extract(title=query_title, overview=None).fused_embedding
            
            query_obj = RetrievalQuery(embedding=query_vector, top_k=50)
            if target_domain != "movies":
                self.domain_adapter.source_domain = target_domain
                query_obj = self.domain_adapter.adapt(query_obj)
            
            matches = self.vector_index.search(query_obj.embedding, top_k=50, domain="movies")
        except Exception:
            return []

        candidates: list[Candidate] = []
        for match in matches:
            title = self._title_for_item_id(match.item_id, data)
            if title is None:
                continue
            candidates.append(
                Candidate(
                    item_id=match.item_id,
                    title=title,
                    score=float(match.score),
                    channel="vector_index",
                    metadata={
                        "signals": {
                            "text": 0.0,
                            "multimodal": float(match.score),
                            "popularity": 0.0,
                            "recency": 0.0,
                            "novelty": 0.0,
                        },
                        "source": match.source,
                    },
                )
            )
            
        if self.cache_manager is not None:
            self.cache_manager.set("retrieval_vector", cache_key, candidates)
            
        return candidates

    def _rank_candidates(self, candidates: list[Candidate]) -> list[str]:
        ranked_items = self._ranked_items(candidates)
        return [item.title for item in ranked_items[:10]]

    def _ranked_items(self, candidates: list[Candidate], context: RecommendationContext | None = None) -> list[RankedItem]:
        pre = self.pipeline.pre_rank(candidates)
        ranker = self.trained_ranker(context)
        if ranker is not None:
            ranked_candidates = ranker.rank(pre)
            final_ranked = [
                RankedItem(
                    item_id=candidate.item_id,
                    title=candidate.title,
                    score=float(ranker.score_candidate(candidate)),
                    explanation=f"trained_ranker|score={ranker.score_candidate(candidate):.6f}",
                    metadata={"ranker": "trained_linear", "signals": candidate.metadata.get("signals", {})},
                )
                for candidate in ranked_candidates[: self.pipeline.budget.final_k]
            ]
        else:
            final_ranked = self.pipeline.fine_rank(pre)
        reranked = self.policy.apply(final_ranked)
        diversified = self.diversity.rerank(reranked)
        shadow_ranked = self.llm_ranker.apply_shadow_ranking(query="", items=diversified)
        return shadow_ranked

    def suggestions(self) -> list[str]:
        return self.repo.titles()

    def recommend_titles(self, movie_title: str, context: RecommendationContext | None = None) -> list[str]:
        ranked = self.recommend_ranked(movie_title, context=context)
        return [item.title for item in ranked]

    def recommend_ranked(self, movie_title: str, context: RecommendationContext | None = None) -> list[RankedItem]:
        title = movie_title.lower().strip()
        
        if self.cache_manager is not None:
            cache_key = build_cache_key("recommend_ranked", title)
            cached = self.cache_manager.get("recommendations", cache_key)
            if cached is not None:
                return cached

        data = self.repo.load_catalog()
        if title not in data["movie_title"].unique():
            user_id = context.user_id if context else None
            candidates, decisions = self.cold_start.for_new_user_with_decisions(
                query_title=movie_title,
                catalog=data
            )
            import logging
            for decision in decisions[:10]:
                logging.info(f"Cold start decision: item={decision.title}, included={decision.included}, reason={decision.reason}")
            if not candidates:
                return []
            result = self._ranked_items(candidates, context=context)[:10]
            if self.cache_manager is not None:
                self.cache_manager.set("recommendations", cache_key, result)
            return result

        artifacts = self.text_features.build_similarity("v1", data)
        idx = data.loc[data["movie_title"] == title].index[0]
        raw = list(enumerate(artifacts.similarity[idx]))
        raw = sorted(raw, key=lambda x: x[1], reverse=True)[1:201]

        candidates: list[Candidate] = []
        for i, score in raw:
            row = data.iloc[i]
            mm = self.multimodal.extract(title=row["movie_title"], overview=None)
            bonus = 0.02 * len(mm.image_attributes)
            candidates.append(
                Candidate(
                    item_id=str(i),
                    title=str(row["movie_title"]),
                    score=float(score) + bonus,
                    channel="hybrid_text_multimodal",
                    metadata={
                        "signals": {
                            "text": float(score),
                            "multimodal": float(bonus),
                            "popularity": 0.0,
                            "recency": 0.0,
                            "novelty": 0.0,
                        }
                    },
                )
            )

        vector_candidates = self._vector_candidates(title, data, context=context)
        graph_retrieval_candidates = self.graph_retriever.retrieve(query_entity_id=title, top_k=20)
        graph_candidates = []
        for match in graph_retrieval_candidates:
            matched_title = self._title_for_item_id(match.item_id, data)
            if matched_title is None:
                continue
            graph_candidates.append(
                Candidate(
                    item_id=match.item_id,
                    title=matched_title,
                    score=float(match.score) * 0.3,
                    channel="graph",
                    metadata={
                        "signals": {
                            "text": 0.0,
                            "multimodal": 0.0,
                            "popularity": 0.0,
                            "recency": 0.0,
                            "novelty": 0.0,
                        },
                        "source": match.source,
                    },
                )
            )

        primary_candidates = self._merge_candidates([*candidates, *vector_candidates, *graph_candidates])

        cold_candidates = []
        if len(primary_candidates) < self.pipeline.budget.final_k:
            cold_candidates = self.cold_start.top_up(
                existing_item_ids={c.item_id for c in primary_candidates},
                query_title=movie_title,
                catalog=data,
                user_id=context.user_id if context else None,
                top_k=self.pipeline.budget.final_k - len(primary_candidates)
            )

        candidates = self._merge_candidates([*primary_candidates, *cold_candidates])
        candidates = self.bandit_policy.select_action(candidates)

        result = self._ranked_items(candidates, context=context)
        result = [item for item in result if item.title.lower() != title]
        result = result[:10]
        
        if self.cache_manager is not None:
            self.cache_manager.set("recommendations", cache_key, result)
            
        return result
