"""Tests for the absolute position solver used by the wlroots backend.

This is the only genuinely new algorithm in the Wayland path, and the only
place where a bug produces a visually-wrong-but-not-failing result.
"""

import pytest

from loose.backends import _logical_size, _resolve_positions


@pytest.fixture
def identifiers(device_edp1, device_hdmi1):
    return [device_edp1, device_hdmi1]


# --- _logical_size ---


class TestLogicalSize:
    def test_resolution_from_config(self, identifiers, logger):
        config = {'resolution': '3440x1440'}
        assert _logical_size('HDMI-1', config, identifiers, logger) == (
            3440,
            1440,
        )

    def test_falls_back_to_preferred_mode(self, identifiers, logger):
        assert _logical_size('eDP-1', {}, identifiers, logger) == (1920, 1080)

    def test_unknown_device_uses_default(self, identifiers, logger):
        assert _logical_size('DP-9', {}, identifiers, logger) == (1920, 1080)

    def test_scale_divides_both_axes(self, identifiers, logger):
        config = {'resolution': '3840x2160', 'scale': 2}
        assert _logical_size('eDP-1', config, identifiers, logger) == (
            1920,
            1080,
        )

    def test_fractional_scale_rounds(self, identifiers, logger):
        config = {'resolution': '1920x1200', 'scale': 1.5}
        assert _logical_size('eDP-1', config, identifiers, logger) == (
            1280,
            800,
        )

    @pytest.mark.parametrize('rotation', ['left', 'right'])
    def test_rotation_swaps_axes(self, identifiers, logger, rotation):
        config = {'resolution': '1920x1080', 'rotate': rotation}
        assert _logical_size('eDP-1', config, identifiers, logger) == (
            1080,
            1920,
        )

    @pytest.mark.parametrize('rotation', ['normal', 'inverted'])
    def test_flat_rotation_keeps_axes(self, identifiers, logger, rotation):
        config = {'resolution': '1920x1080', 'rotate': rotation}
        assert _logical_size('eDP-1', config, identifiers, logger) == (
            1920,
            1080,
        )

    def test_scale_applied_before_rotation_swap(self, identifiers, logger):
        config = {'resolution': '3840x2160', 'scale': 2, 'rotate': 'left'}
        assert _logical_size('eDP-1', config, identifiers, logger) == (
            1080,
            1920,
        )


# --- _resolve_positions ---


class TestResolvePositions:
    def test_single_device_at_origin(self, identifiers, logger):
        config = {'eDP-1': {'resolution': '1920x1080'}}
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0)
        }

    def test_right_of_shares_top_edge(self, identifiers, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'HDMI-1': (1920, 0),
        }

    def test_left_of_shifts_into_positive_quadrant(self, identifiers, logger):
        """left-of puts the device at a negative x, normalization fixes it up."""
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {'resolution': '3440x1440', 'left-of': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'HDMI-1': (0, 0),
            'eDP-1': (3440, 0),
        }

    def test_below_shares_left_edge(self, identifiers, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {'resolution': '3440x1440', 'below': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'HDMI-1': (0, 1080),
        }

    def test_above_shifts_into_positive_quadrant(self, identifiers, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {'resolution': '3440x1440', 'above': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'HDMI-1': (0, 0),
            'eDP-1': (0, 1440),
        }

    def test_chain_resolves_transitively(self, device_edp1, logger):
        """C right-of B right-of A, declared in reverse dependency order."""
        identifiers = [device_edp1]
        config = {
            'DP-2': {'resolution': '1000x1000', 'right-of': 'DP-1'},
            'DP-1': {'resolution': '1000x1000', 'right-of': 'eDP-1'},
            'eDP-1': {'resolution': '1920x1080'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'DP-1': (1920, 0),
            'DP-2': (2920, 0),
        }

    def test_mixed_axes(self, identifiers, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {'resolution': '1920x1080', 'right-of': 'eDP-1'},
            'DP-1': {'resolution': '1920x1080', 'below': 'HDMI-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'HDMI-1': (1920, 0),
            'DP-1': (1920, 1080),
        }

    def test_rotated_anchor_uses_swapped_width(self, identifiers, logger):
        """A portrait anchor is 1080 wide, so the neighbour starts at 1080."""
        config = {
            'eDP-1': {'resolution': '1920x1080', 'rotate': 'left'},
            'HDMI-1': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'HDMI-1': (1080, 0),
        }

    def test_scaled_anchor_uses_logical_width(self, identifiers, logger):
        config = {
            'eDP-1': {'resolution': '3840x2160', 'scale': 2},
            'HDMI-1': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'HDMI-1': (1920, 0),
        }

    def test_preferred_mode_used_when_resolution_omitted(
        self, identifiers, logger
    ):
        """eDP-1 has no resolution, so its preferred 1920x1080 sets the offset."""
        config = {
            'eDP-1': {},
            'HDMI-1': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0),
            'HDMI-1': (1920, 0),
        }

    def test_disabled_device_excluded(self, identifiers, logger):
        config = {
            'eDP-1': {'disabled': True},
            'HDMI-1': {'resolution': '3440x1440'},
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'HDMI-1': (0, 0)
        }

    def test_reserved_keys_ignored(self, identifiers, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'hooks': {'pre': ['true']},
            'comment': 'laptop only',
        }
        assert _resolve_positions(config, identifiers, logger) == {
            'eDP-1': (0, 0)
        }

    def test_empty_config(self, identifiers, logger):
        assert _resolve_positions({}, identifiers, logger) == {}

    def test_multiple_roots_overlap_and_warn(
        self, identifiers, logger, caplog
    ):
        """xrandr stacks unpositioned outputs too, but we say so out loud."""
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {'resolution': '3440x1440'},
        }
        with caplog.at_level('WARNING'):
            positions = _resolve_positions(config, identifiers, logger)

        assert positions == {'eDP-1': (0, 0), 'HDMI-1': (0, 0)}
        assert 'overlap' in caplog.text

    def test_reference_to_disabled_device_degrades_to_root(
        self, identifiers, logger, caplog
    ):
        config = {
            'eDP-1': {'disabled': True},
            'HDMI-1': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        with caplog.at_level('WARNING'):
            positions = _resolve_positions(config, identifiers, logger)

        assert positions == {'HDMI-1': (0, 0)}
        assert 'eDP-1' in caplog.text

    def test_reference_to_absent_device_degrades_to_root(
        self, identifiers, logger, caplog
    ):
        config = {'HDMI-1': {'resolution': '3440x1440', 'right-of': 'DP-9'}}
        with caplog.at_level('WARNING'):
            positions = _resolve_positions(config, identifiers, logger)

        assert positions == {'HDMI-1': (0, 0)}
        assert 'DP-9' in caplog.text

    def test_first_positioning_key_wins(self, identifiers, logger):
        """Only one directive is honored, matching _build_monitor_grid()."""
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-1': {
                'resolution': '1920x1080',
                'left-of': 'eDP-1',
                'right-of': 'eDP-1',
            },
        }
        positions = _resolve_positions(config, identifiers, logger)
        # left-of comes first in POSITION_KEYS, so HDMI-1 ends up on the left
        assert positions == {'HDMI-1': (0, 0), 'eDP-1': (1920, 0)}
