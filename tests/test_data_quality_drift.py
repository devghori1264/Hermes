import pandas as pd

from src.data.quality.drift import DriftThresholds, compare_profiles, profile_frame


def test_missing_fraction_drift():
    baseline = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3.0]})
    current = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, 3.0]})

    base_profile = profile_frame(baseline)
    current_profile = profile_frame(current)
    report = compare_profiles(base_profile, current_profile, DriftThresholds(missing_fraction_delta=0.2))

    assert report.is_drifted
    assert any(violation.metric == "missing_fraction" for violation in report.violations)


def test_numeric_mean_drift():
    baseline = pd.DataFrame({"score": [0.0, 0.1, 0.2]})
    current = pd.DataFrame({"score": [1.0, 1.1, 1.2]})

    base_profile = profile_frame(baseline)
    current_profile = profile_frame(current)
    report = compare_profiles(base_profile, current_profile, DriftThresholds(numeric_mean_delta=0.5))

    assert report.is_drifted
    assert any(violation.metric == "mean_delta" for violation in report.violations)


def test_no_drift():
    baseline = pd.DataFrame({"score": [1.0, 1.1, 1.2], "name": ["a", "b", "c"]})
    current = pd.DataFrame({"score": [1.0, 1.05, 1.2], "name": ["a", "b", "c"]})

    base_profile = profile_frame(baseline)
    current_profile = profile_frame(current)
    report = compare_profiles(base_profile, current_profile, DriftThresholds())

    assert report.is_drifted is False
    assert report.violations == []
