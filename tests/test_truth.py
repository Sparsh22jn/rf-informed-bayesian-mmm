from mmm_sports.simulate.schedule import BROADCASTERS
from mmm_sports.simulate.truth import ALWAYS_ON, CHANNELS, EVENT_TARGETED, N_SEASONS, TRUTH


def test_all_channels_parameterized():
    for field in (TRUTH.alpha, TRUTH.k, TRUTH.s, TRUTH.beta):
        assert set(field) == set(CHANNELS)


def test_ranges_sane():
    for ch in CHANNELS:
        assert 0.0 <= TRUTH.alpha[ch] < 1.0
        assert 0.0 < TRUTH.k[ch] < 1_000_000.0
        assert 0.5 < TRUTH.s[ch] < 5.0
        assert TRUTH.beta[ch] >= 0.0
    assert TRUTH.sigma > 0.0
    assert len(TRUTH.season_intercept) == N_SEASONS


def test_event_targeted_channels_have_no_adstock():
    for ch in EVENT_TARGETED:
        assert TRUTH.alpha[ch] == 0.0
    for ch in ALWAYS_ON:
        assert TRUTH.alpha[ch] > 0.0


def test_display_channel_is_near_dead():
    other_always_on = [TRUTH.beta[ch] for ch in ALWAYS_ON if ch != "display"]
    assert TRUTH.beta["display"] < 0.1 * min(other_always_on)


def test_broadcaster_levels_match_schedule():
    assert set(TRUTH.broadcaster_beta) == set(BROADCASTERS)


def test_tentpole_coefficient_is_large_relative_to_other_controls():
    tentpole = abs(TRUTH.control_beta["tentpole_tier"])
    other_scalar_controls = [
        abs(v) for k, v in TRUTH.control_beta.items() if k != "tentpole_tier"
    ]
    broadcaster_spread = max(TRUTH.broadcaster_beta.values()) - min(
        TRUTH.broadcaster_beta.values()
    )
    assert tentpole > max(other_scalar_controls)
    assert tentpole > broadcaster_spread
