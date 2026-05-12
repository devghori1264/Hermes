from src.evaluation.offline_metrics import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_metrics() -> None:
    recommended = ["a", "b", "c"]
    relevant = {"a", "c"}
    assert ndcg_at_k(recommended, relevant, k=3) > 0
    assert recall_at_k(recommended, relevant, k=3) == 1.0
    assert mrr_at_k(recommended, relevant, k=3) > 0
    assert map_at_k(recommended, relevant, k=3) > 0
