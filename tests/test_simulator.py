from src.evaluation.simulator import BehaviorModel


def test_simulator_runs() -> None:
    model = BehaviorModel(base_click_rate=0.1, seed=5)
    outcomes = model.simulate([("a", 0.5), ("b", 0.2)])
    assert len(outcomes) == 2
