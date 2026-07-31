import numpy as np

from mmm_sports.transforms import geometric_adstock, hill_saturation


def test_adstock_decay():
    # Single $10 spike at t=0, alpha=0.5, l_max=3: hand-computed geometric
    # decay 10, 5, 2.5, 1.25, then nothing (lag exceeds l_max).
    spend = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    expected = np.array([10.0, 5.0, 2.5, 1.25, 0.0])
    result = geometric_adstock(spend, alpha=0.5, l_max=3)
    np.testing.assert_allclose(result, expected)


def test_adstock_decay_overlapping_spikes():
    # Two spikes at t=0 and t=1, alpha=0.5, l_max=3 -- hand-computed
    # superposition of two decaying kernels.
    spend = np.array([10.0, 20.0, 0.0, 0.0, 0.0])
    expected = np.array([10.0, 25.0, 12.5, 6.25, 2.5])
    result = geometric_adstock(spend, alpha=0.5, l_max=3)
    np.testing.assert_allclose(result, expected)


def test_adstock_truncates_at_l_max():
    # l_max=1 means only lag 0 and 1 contribute -- the spike's effect must be
    # gone by t=2.
    spend = np.array([10.0, 0.0, 0.0, 0.0])
    result = geometric_adstock(spend, alpha=0.5, l_max=1)
    np.testing.assert_allclose(result, [10.0, 5.0, 0.0, 0.0])


def test_hill_monotone_and_bounded():
    spend = np.linspace(0.0, 100_000.0, 50)
    result = hill_saturation(spend, k=10_000.0, s=2.0)

    assert np.all(np.diff(result) > 0)  # strictly increasing
    assert np.all((result >= 0.0) & (result < 1.0))  # bounded in [0, 1)
    assert result[0] == 0.0  # zero spend -> zero effect
    np.testing.assert_allclose(hill_saturation(np.array([10_000.0]), k=10_000.0, s=2.0), [0.5])
