from __future__ import annotations
from collections import defaultdict
from src.retrieval.types import RetrievalCandidate
from src.retrieval.knowledge_graph import KnowledgeGraph

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

if nn is not None:
    class GraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
            super().__init__()
            self.agg1 = nn.Linear(in_dim * 2, hidden_dim)
            self.agg2 = nn.Linear(hidden_dim * 2, out_dim)

        def forward(self, features: torch.Tensor, adj: dict[int, list[int]]) -> torch.Tensor:
            num_nodes = features.shape[0]
            h1 = torch.zeros(num_nodes, self.agg1.out_features, device=features.device)
            for i in range(num_nodes):
                neighbors = adj.get(i, [])
                if neighbors:
                    neigh_feats = features[neighbors].mean(dim=0)
                else:
                    neigh_feats = torch.zeros_like(features[i])
                combined = torch.cat([features[i], neigh_feats])
                h1[i] = F.relu(self.agg1(combined))
            h2 = torch.zeros(num_nodes, self.agg2.out_features, device=features.device)
            for i in range(num_nodes):
                neighbors = adj.get(i, [])
                if neighbors:
                    neigh_feats = h1[neighbors].mean(dim=0)
                else:
                    neigh_feats = torch.zeros_like(h1[i])
                combined = torch.cat([h1[i], neigh_feats])
                h2[i] = F.normalize(self.agg2(combined), p=2, dim=0)
            return h2

class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}
        self.adj: dict[int, list[int]] = defaultdict(list)
        self._build_graph()
        self.embeddings = None
        if torch is not None and nn is not None:
            num_nodes = len(self.node_to_idx)
            in_dim = 32
            hidden_dim = 64
            out_dim = 32
            self.features = torch.randn(num_nodes, in_dim)
            self.gnn = GraphSAGE(in_dim, hidden_dim, out_dim)
        else:
            self.features = None
            self.gnn = None

    def _build_graph(self) -> None:
        for idx, entity_id in enumerate(self.kg.entities.keys()):
            self.node_to_idx[entity_id] = idx
            self.idx_to_node[idx] = entity_id
            
        for rel in self.kg.relations:
            if rel.source_id in self.node_to_idx and rel.target_id in self.node_to_idx:
                u = self.node_to_idx[rel.source_id]
                v = self.node_to_idx[rel.target_id]
                self.adj[u].append(v)
                self.adj[v].append(u)

    def _compute_embeddings(self) -> None:
        if torch is None or self.gnn is None or self.features is None:
            self.embeddings = None
            return
        with torch.no_grad():
            self.embeddings = self.gnn(self.features, self.adj)

    def retrieve(self, query_entity_id: str, top_k: int = 50) -> list[RetrievalCandidate]:
        if torch is None:
            return self._retrieve_structural(query_entity_id, top_k)
        if self.embeddings is None:
            self._compute_embeddings()
            
        if query_entity_id not in self.node_to_idx or self.embeddings is None:
            return []
            
        query_idx = self.node_to_idx[query_entity_id]
        query_emb = self.embeddings[query_idx]
        
        scores = torch.matmul(self.embeddings, query_emb)
        top_indices = torch.topk(scores, k=min(top_k + 1, len(scores))).indices.tolist()
        
        candidates = []
        for idx in top_indices:
            if idx == query_idx:
                continue
            entity_id = self.idx_to_node[idx]
            entity = self.kg.get_entity(entity_id)
            if entity and entity.type == "movie":
                score = float(scores[idx])
                candidates.append(
                    RetrievalCandidate(
                        item_id=entity_id,
                        score=score,
                        source="gnn_graph",
                        metadata={"graph_score": score}
                    )
                )
                if len(candidates) >= top_k:
                    break
        return candidates

    def _retrieve_structural(self, query_entity_id: str, top_k: int) -> list[RetrievalCandidate]:
        if query_entity_id not in self.node_to_idx:
            return []
        query_idx = self.node_to_idx[query_entity_id]
        query_neighbors = set(self.adj.get(query_idx, []))
        if not query_neighbors:
            return []
        candidates: list[RetrievalCandidate] = []
        for idx, entity_id in self.idx_to_node.items():
            if idx == query_idx:
                continue
            entity = self.kg.get_entity(entity_id)
            if entity is None or entity.type != "movie":
                continue
            neighbors = set(self.adj.get(idx, []))
            if not neighbors:
                continue
            overlap = len(query_neighbors & neighbors)
            union = len(query_neighbors | neighbors)
            if union == 0:
                continue
            score = float(overlap / union)
            if score <= 0:
                continue
            candidates.append(
                RetrievalCandidate(
                    item_id=entity_id,
                    score=score,
                    source="graph_structural",
                    metadata={"graph_score": score},
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
