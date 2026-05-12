import numpy as np
from typing import Dict, List, Tuple
from src.domain.models import Candidate

class BanditPolicy:
    def select_action(self, candidates: List[Candidate], context_features: np.ndarray = None) -> List[Candidate]:
        raise NotImplementedError

class ThompsonSamplingBandit(BanditPolicy):
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.alpha_map: Dict[str, float] = {}
        self.beta_map: Dict[str, float] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def update(self, item_id: str, reward: float) -> None:
        if item_id not in self.alpha_map:
            self.alpha_map[item_id] = self.prior_alpha
            self.beta_map[item_id] = self.prior_beta
            
        if reward > 0:
            self.alpha_map[item_id] += reward
        else:
            self.beta_map[item_id] += 1.0

    def select_action(self, candidates: List[Candidate], context_features: np.ndarray = None) -> List[Candidate]:
        if not self.alpha_map:
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        scored_candidates = []
        for candidate in candidates:
            alpha = self.alpha_map.get(candidate.item_id, self.prior_alpha)
            beta = self.beta_map.get(candidate.item_id, self.prior_beta)
            sampled_score = np.random.beta(alpha, beta)

            blended_score = 0.7 * candidate.score + 0.3 * sampled_score
            scored_candidates.append((blended_score, candidate))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates]

class UCBBandit(BanditPolicy):
    def __init__(self, exploration_weight: float = np.sqrt(2)) -> None:
        self.exploration_weight = exploration_weight
        self.counts: Dict[str, int] = {}
        self.values: Dict[str, float] = {}
        self.total_pulls = 0

    def update(self, item_id: str, reward: float) -> None:
        self.total_pulls += 1
        count = self.counts.get(item_id, 0)
        value = self.values.get(item_id, 0.0)
        
        self.counts[item_id] = count + 1
        self.values[item_id] = value + (reward - value) / (count + 1)

    def select_action(self, candidates: List[Candidate], context_features: np.ndarray = None) -> List[Candidate]:
        if self.total_pulls == 0:
            return candidates

        scored_candidates = []
        for candidate in candidates:
            count = self.counts.get(candidate.item_id, 0)
            if count == 0:
                ucb_score = float('inf')
            else:
                value = self.values[candidate.item_id]
                exploration = self.exploration_weight * np.sqrt(np.log(self.total_pulls) / count)
                ucb_score = value + exploration
                
            blended_score = 0.5 * candidate.score + 0.5 * min(ucb_score, 1.0)
            scored_candidates.append((blended_score, candidate))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates]
