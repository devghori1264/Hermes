from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    attributes: dict[str, str]

@dataclass(frozen=True)
class Relation:
    source_id: str
    target_id: str
    type: str
    weight: float

class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

    def save(self, filepath: str) -> None:
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump((self.entities, self.relations, self.adj), f)

    def load(self, filepath: str) -> bool:
        import pickle
        from pathlib import Path
        if not Path(filepath).exists():
            return False
        with open(filepath, "rb") as f:
            self.entities, self.relations, self.adj = pickle.load(f)
        return True

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)
        self.adj[relation.source_id].append((relation.target_id, relation.type, relation.weight))
        self.adj[relation.target_id].append((relation.source_id, relation.type, relation.weight))

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def get_neighbors(self, entity_id: str) -> list[tuple[str, str, float]]:
        return self.adj.get(entity_id, [])

    def random_walk(self, start_id: str, steps: int, seed: int = 42) -> list[str]:
        import random
        rng = random.Random(seed)
        path = [start_id]
        current = start_id
        for _ in range(steps):
            neighbors = self.adj.get(current, [])
            if not neighbors:
                break
            next_node, _, _ = rng.choice(neighbors)
            path.append(next_node)
            current = next_node
        return path

    def build_from_catalog(self, df) -> None:
        """Build the knowledge graph from the movie catalog DataFrame."""
        for idx, row in df.iterrows():
            movie_id = str(idx)
            movie_title = str(row.get("movie_title", "")).strip().lower()
            if not movie_title:
                continue
            
            # Add movie entity
            self.add_entity(Entity(id=movie_id, type="movie", attributes={"title": movie_title}))
            self.add_entity(Entity(id=movie_title, type="movie_title", attributes={"title": movie_title}))
            self.add_relation(Relation(source_id=movie_title, target_id=movie_id, type="is_movie", weight=1.0))
            
            # Add director
            director = str(row.get("director_name", "")).strip().lower()
            if director and director != "nan":
                self.add_entity(Entity(id=director, type="director", attributes={"name": director}))
                self.add_relation(Relation(source_id=movie_id, target_id=director, type="directed_by", weight=1.0))
                self.add_relation(Relation(source_id=movie_title, target_id=director, type="directed_by", weight=1.0))
                
            # Add actors
            for actor_col in ["actor_1_name", "actor_2_name", "actor_3_name"]:
                actor = str(row.get(actor_col, "")).strip().lower()
                if actor and actor != "nan":
                    self.add_entity(Entity(id=actor, type="actor", attributes={"name": actor}))
                    self.add_relation(Relation(source_id=movie_id, target_id=actor, type="acted_in", weight=1.0))
                    self.add_relation(Relation(source_id=movie_title, target_id=actor, type="acted_in", weight=1.0))
                    
            # Add genres
            genres_raw = str(row.get("genres", ""))
            if genres_raw and genres_raw != "nan":
                for genre in genres_raw.split():
                    genre = genre.strip().lower()
                    if genre:
                        self.add_entity(Entity(id=genre, type="genre", attributes={"name": genre}))
                        self.add_relation(Relation(source_id=movie_id, target_id=genre, type="has_genre", weight=1.0))
                        self.add_relation(Relation(source_id=movie_title, target_id=genre, type="has_genre", weight=1.0))
