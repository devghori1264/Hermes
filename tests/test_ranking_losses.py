import pytest

from src.ranking.losses import (
    LISTWISE_SOFTMAX,
    PAIRWISE_HINGE,
    POINTWISE_LOGLOSS,
    listwise_softmax_loss,
    normalize_objective_name,
    pairwise_hinge_loss,
    pointwise_logloss,
    supported_objectives,
)


def test_pointwise_logloss() -> None:
    result = pointwise_logloss([1, 0], [0.9, 0.1])
    assert result.value > 0
    assert result.name == POINTWISE_LOGLOSS


def test_pairwise_hinge_loss() -> None:
    result = pairwise_hinge_loss([(1.0, 0.2)])
    assert result.value >= 0
    assert result.name == PAIRWISE_HINGE


def test_listwise_softmax_loss() -> None:
    result = listwise_softmax_loss([2.0, 1.0], [1, 0])
    assert result.value > 0
    assert result.name == LISTWISE_SOFTMAX


def test_supported_objectives() -> None:
    from src.ranking.losses import POINTWISE_LOGLOSS, POINTWISE_FOCAL, PAIRWISE_HINGE, LISTWISE_SOFTMAX, LISTWISE_LISTMLE, LISTWISE_LAMBDARANK
    assert supported_objectives() == (POINTWISE_LOGLOSS, POINTWISE_FOCAL, PAIRWISE_HINGE, LISTWISE_SOFTMAX, LISTWISE_LISTMLE, LISTWISE_LAMBDARANK)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pointwise", POINTWISE_LOGLOSS),
        ("pairwise", PAIRWISE_HINGE),
        ("listwise", LISTWISE_SOFTMAX),
        ("pairwise_hinge", PAIRWISE_HINGE),
    ],
)
def test_normalize_objective_name(raw: str, expected: str) -> None:
    assert normalize_objective_name(raw) == expected


def test_normalize_objective_name_invalid() -> None:
    with pytest.raises(ValueError, match="unsupported ranking objective"):
        normalize_objective_name("unknown")


def test_pointwise_logloss_mismatch_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        pointwise_logloss([1, 0], [0.1])


def test_listwise_softmax_loss_mismatch_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        listwise_softmax_loss([1.0], [1, 0])
