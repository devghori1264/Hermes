from src.exploration.bandit import BanditAction, EpsilonGreedyPolicy


def test_epsilon_greedy_returns_actions() -> None:
    policy = EpsilonGreedyPolicy(epsilon=0.0, seed=7)
    actions = [BanditAction(item_id="a", score=0.9), BanditAction(item_id="b", score=0.1)]
    selected = policy.select(actions, top_k=1)
    assert selected[0].item_id == "a"
