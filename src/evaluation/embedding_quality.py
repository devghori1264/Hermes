import numpy as np
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class EmbeddingQualityReport:
    alignment: float
    uniformity: float
    anisotropy: float

class EmbeddingEvaluator:
    def evaluate(self, embeddings: Sequence[np.ndarray], positive_pairs: list[tuple[int, int]] = []) -> EmbeddingQualityReport:
        if not embeddings:
            return EmbeddingQualityReport(0.0, 0.0, 0.0)
            
        mat = np.array(embeddings)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        
        alignment = 0.0
        if positive_pairs:
            diffs = []
            for i, j in positive_pairs:
                if i < len(mat) and j < len(mat):
                    diffs.append(np.linalg.norm(mat[i] - mat[j])**2)
            if diffs:
                alignment = float(np.mean(diffs))

        N = len(mat)
        if N > 1:
            idx = np.random.choice(N, min(N, 1000), replace=False)
            sub_mat = mat[idx]
            dists = []
            for i in range(len(sub_mat)):
                for j in range(i+1, len(sub_mat)):
                    dists.append(np.exp(-2.0 * np.linalg.norm(sub_mat[i] - sub_mat[j])**2))
            uniformity = float(np.log(np.mean(dists))) if dists else 0.0
        else:
            uniformity = 0.0

        u, s, vh = np.linalg.svd(mat - np.mean(mat, axis=0), full_matrices=False)
        if len(s) > 0:
            anisotropy = float(s[0] / np.sum(s))
        else:
            anisotropy = 0.0

        return EmbeddingQualityReport(
            alignment=alignment,
            uniformity=uniformity,
            anisotropy=anisotropy
        )
