"""Tests for the wlroots backend: probe normalization and command building."""

import json
import subprocess

import pytest

from loose.backends import (
    BackendError,
    WlrootsBackend,
    _find_scaled_mode,
    _resolve_exact_frequency,
    _wlr_output_mode,
    _wlr_product_id,
)


@pytest.fixture
def backend(mocker):
    """A WlrootsBackend with wlr-randr pretend-installed."""
    mocker.patch('loose.backends.which', return_value='/usr/bin/wlr-randr')
    return WlrootsBackend()


@pytest.fixture
def probed(backend, mocker, wlr_randr_json):
    """Identifiers as produced from the wlr-randr fixture."""
    mocker.patch(
        'loose.backends.subprocess.check_output',
        return_value=json.dumps(wlr_randr_json),
    )
    return backend.probe()


# --- probe ---


class TestProbe:
    def test_sorted_by_product_id(self, probed):
        assert [d['device_name'] for d in probed] == ['HDMI-A-1', 'eDP-1']

    def test_serial_wins_as_product_id(self, probed):
        hdmi = next(d for d in probed if d['device_name'] == 'HDMI-A-1')
        assert hdmi['product_id'] == 'ABC123'

    def test_make_model_fallback_when_serial_empty(self, probed):
        edp = next(d for d in probed if d['device_name'] == 'eDP-1')
        assert edp['product_id'] == 'AU Optronics 0x1234'

    def test_everything_reported_is_connected(self, probed):
        assert all(d['is_connected'] for d in probed)

    def test_enabled_maps_to_is_active(self, probed):
        active = {d['device_name']: d['is_active'] for d in probed}
        assert active == {'eDP-1': True, 'HDMI-A-1': False}

    def test_modes_grouped_by_resolution(self, probed):
        edp = next(d for d in probed if d['device_name'] == 'eDP-1')
        assert [
            (m['resolution_width'], m['resolution_height'])
            for m in edp['resolution_modes']
        ] == [(1920, 1080), (1280, 720)]

    def test_frequencies_kept_per_resolution(self, probed):
        edp = next(d for d in probed if d['device_name'] == 'eDP-1')
        fullhd = edp['resolution_modes'][0]
        assert [f['frequency'] for f in fullhd['frequencies']] == [
            59.951,
            48.001,
        ]

    def test_current_and_preferred_flags(self, probed):
        edp = next(d for d in probed if d['device_name'] == 'eDP-1')
        first = edp['resolution_modes'][0]['frequencies'][0]
        assert first['is_current'] is True
        assert first['is_preferred'] is True

    def test_mhz_refresh_normalized_to_hz(self, backend, mocker):
        """Some wlr-randr builds report mHz, we want Hz downstream."""
        mocker.patch(
            'loose.backends.subprocess.check_output',
            return_value=json.dumps(
                [
                    {
                        'name': 'eDP-1',
                        'make': '',
                        'model': '',
                        'serial': '',
                        'enabled': True,
                        'modes': [
                            {
                                'width': 1920,
                                'height': 1080,
                                'refresh': 59951,
                                'preferred': True,
                                'current': True,
                            }
                        ],
                    }
                ]
            ),
        )
        identifiers = backend.probe()
        frequency = identifiers[0]['resolution_modes'][0]['frequencies'][0]
        assert frequency['frequency'] == 59.951

    def test_missing_binary_raises(self, mocker):
        mocker.patch('loose.backends.which', return_value=None)
        with pytest.raises(BackendError, match='wlr-randr'):
            WlrootsBackend().probe()

    def test_non_json_output_raises(self, backend, mocker):
        mocker.patch(
            'loose.backends.subprocess.check_output',
            return_value='eDP-1 "AU Optronics"\n  Enabled: yes\n',
        )
        with pytest.raises(BackendError, match='0.3.0'):
            backend.probe()

    def test_failing_command_raises(self, backend, mocker):
        mocker.patch(
            'loose.backends.subprocess.check_output',
            side_effect=subprocess.CalledProcessError(1, 'wlr-randr'),
        )
        with pytest.raises(BackendError, match='WAYLAND_DISPLAY'):
            backend.probe()


# --- product id ---


class TestProductId:
    def test_falls_back_to_name(self):
        output = {'name': 'DP-3', 'make': '', 'model': '', 'serial': ''}
        assert _wlr_product_id(output) == 'DP-3'

    def test_handles_missing_keys(self):
        assert _wlr_product_id({'name': 'DP-3'}) == 'DP-3'


# --- frequency resolution ---


class TestFrequencyResolution:
    def test_rounded_config_value_resolves_to_real_refresh(
        self, probed, logger
    ):
        assert (
            _resolve_exact_frequency('eDP-1', '1920x1080', 60, probed, logger)
            == 59.951
        )

    def test_no_frequency_takes_fastest_mode(self, probed, logger):
        assert (
            _resolve_exact_frequency(
                'eDP-1', '1920x1080', None, probed, logger
            )
            == 59.951
        )

    def test_unavailable_frequency_returns_none(self, probed, logger):
        assert (
            _resolve_exact_frequency('eDP-1', '1920x1080', 144, probed, logger)
            is None
        )

    def test_unknown_device_returns_none(self, probed, logger):
        assert (
            _resolve_exact_frequency('DP-9', '1920x1080', 60, probed, logger)
            is None
        )

    def test_mode_string_uses_real_refresh(self, probed, logger):
        config = {'resolution': '1920x1080', 'frequency': 60}
        assert _wlr_output_mode('eDP-1', config, probed, logger) == (
            '1920x1080@59.951Hz',
            None,
        )

    def test_mode_string_passes_config_value_through_when_unresolvable(
        self, probed, logger
    ):
        config = {'resolution': '1920x1080', 'frequency': 144}
        assert _wlr_output_mode('eDP-1', config, probed, logger) == (
            '1920x1080@144Hz',
            None,
        )

    def test_no_resolution_means_no_mode(self, probed, logger):
        assert _wlr_output_mode(
            'eDP-1', {'frequency': 60}, probed, logger
        ) == (
            None,
            None,
        )


# --- build_command ---


class TestBuildCommand:
    def test_single_device_full_config(self, backend, probed, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080', 'frequency': 60},
            'HDMI-A-1': {'disabled': True},
        }
        assert backend.build_command(config, probed, logger) == [
            '/usr/bin/wlr-randr',
            '--output',
            'eDP-1',
            '--on',
            '--mode',
            '1920x1080@59.951Hz',
            '--transform',
            'normal',
            '--pos',
            '0,0',
            '--output',
            'HDMI-A-1',
            '--off',
        ]

    def test_preferred_when_no_resolution(self, backend, probed, logger):
        config = {'eDP-1': {'primary': True}, 'HDMI-A-1': {'disabled': True}}
        command = backend.build_command(config, probed, logger)
        assert '--preferred' in command
        # primary has no wlr-randr equivalent and must not leak into the args
        assert '--primary' not in command

    def test_positions_come_from_the_solver(self, backend, probed, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-A-1': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        assert backend.build_command(config, probed, logger) == [
            '/usr/bin/wlr-randr',
            '--output',
            'eDP-1',
            '--on',
            '--mode',
            '1920x1080@59.951Hz',
            '--transform',
            'normal',
            '--pos',
            '0,0',
            '--output',
            'HDMI-A-1',
            '--on',
            '--mode',
            '3440x1440@99.982Hz',
            '--transform',
            'normal',
            '--pos',
            '1920,0',
        ]

    @pytest.mark.parametrize(
        'rotation,transform',
        [
            ('normal', 'normal'),
            ('left', '90'),
            ('right', '270'),
            ('inverted', '180'),
        ],
    )
    def test_transform_mapping(
        self, backend, probed, logger, rotation, transform
    ):
        config = {
            'eDP-1': {'resolution': '1920x1080', 'rotate': rotation},
            'HDMI-A-1': {'disabled': True},
        }
        command = backend.build_command(config, probed, logger)
        assert command[command.index('--transform') + 1] == transform

    def test_unknown_rotation_falls_back_to_normal(
        self, backend, probed, logger, caplog
    ):
        config = {
            'eDP-1': {'resolution': '1920x1080', 'rotate': 'sideways'},
            'HDMI-A-1': {'disabled': True},
        }
        with caplog.at_level('WARNING'):
            command = backend.build_command(config, probed, logger)

        assert command[command.index('--transform') + 1] == 'normal'
        assert 'sideways' in caplog.text

    def test_scale_is_emitted(self, backend, probed, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080', 'scale': 1.5},
            'HDMI-A-1': {'disabled': True},
        }
        command = backend.build_command(config, probed, logger)
        assert command[command.index('--scale') + 1] == '1.5'

    def test_unconfigured_devices_are_turned_off(
        self, backend, probed, logger
    ):
        config = {'eDP-1': {'resolution': '1920x1080'}}
        command = backend.build_command(config, probed, logger)
        assert command[-3:] == ['--output', 'HDMI-A-1', '--off']

    def test_hooks_and_comment_are_not_devices(self, backend, probed, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080'},
            'HDMI-A-1': {'disabled': True},
            'hooks': {'pre': ['true']},
            'comment': 'laptop only',
        }
        command = backend.build_command(config, probed, logger)
        assert 'hooks' not in command
        assert 'comment' not in command

    def test_missing_binary_raises(self, probed, logger, mocker):
        mocker.patch('loose.backends.which', return_value=None)
        with pytest.raises(BackendError, match='wlr-randr'):
            WlrootsBackend().build_command({}, probed, logger)


# --- auto-scaling ---


@pytest.fixture
def hidpi_panel():
    """A 3200x2000 laptop panel, exactly like a real eDP one: native modes only.

    Wayland exposes only what the panel reports, while the X server also
    advertises synthesized scaled modes. That difference is what auto-scaling
    exists to paper over.
    """
    return {
        'device_name': 'eDP-1',
        'product_id': 'Samsung Display Corp. 0x41A2',
        'is_active': False,
        'is_connected': True,
        'resolution_modes': [
            {
                'resolution_width': 3200,
                'resolution_height': 2000,
                'frequencies': [
                    {
                        'frequency': 120.0,
                        'is_current': False,
                        'is_preferred': True,
                    },
                    {
                        'frequency': 60.0,
                        'is_current': False,
                        'is_preferred': False,
                    },
                ],
            },
        ],
    }


class TestFindScaledMode:
    def test_matching_aspect_resolves(self, hidpi_panel):
        """3200x2000 driven at scale 1.6667 gives 1920x1200 logical."""
        assert _find_scaled_mode(hidpi_panel, 1920, 1200, 60) == (
            3200,
            2000,
            60.0,
            1.666667,
        )

    def test_mismatched_aspect_rejected(self, hidpi_panel):
        """16:9 on a 16:10 panel is unreachable, there is only one scale."""
        assert _find_scaled_mode(hidpi_panel, 1920, 1080, 60) is None

    def test_unavailable_frequency_rejected(self, hidpi_panel):
        assert _find_scaled_mode(hidpi_panel, 1920, 1200, 75) is None

    def test_no_frequency_takes_fastest(self, hidpi_panel):
        assert _find_scaled_mode(hidpi_panel, 1920, 1200, None)[2] == 120.0

    def test_never_scales_up(self, hidpi_panel):
        """A request larger than every mode cannot be satisfied by scaling."""
        assert _find_scaled_mode(hidpi_panel, 3840, 2400, 60) is None

    def test_equal_resolution_is_not_a_scaled_match(self, hidpi_panel):
        assert _find_scaled_mode(hidpi_panel, 3200, 2000, 60) is None

    def test_highest_resolution_wins(self, hidpi_panel):
        """Two aspect-matching candidates, the panel keeps running at native."""
        hidpi_panel['resolution_modes'].append(
            {
                'resolution_width': 2560,
                'resolution_height': 1600,
                'frequencies': [
                    {
                        'frequency': 60.0,
                        'is_current': False,
                        'is_preferred': False,
                    },
                ],
            }
        )
        assert _find_scaled_mode(hidpi_panel, 1920, 1200, 60)[0] == 3200

    def test_x11_never_auto_scales(self, hidpi_panel):
        """The X server already synthesizes these modes itself."""
        from loose.backends import XrandrBackend

        assert (
            XrandrBackend.find_scaled_mode(hidpi_panel, 1920, 1200, 60) is None
        )


class TestAutoScaleCommand:
    def test_command_drives_native_mode_with_scale(
        self, backend, hidpi_panel, logger
    ):
        config = {'eDP-1': {'resolution': '1920x1200', 'frequency': 60}}
        assert backend.build_command(config, [hidpi_panel], logger) == [
            '/usr/bin/wlr-randr',
            '--output',
            'eDP-1',
            '--on',
            '--mode',
            '3200x2000@60.0Hz',
            '--transform',
            'normal',
            '--scale',
            '1.666667',
            '--pos',
            '0,0',
        ]

    def test_explicit_scale_disables_auto_scaling(
        self, backend, hidpi_panel, logger
    ):
        """An explicit scale means the user knows what they want."""
        config = {
            'eDP-1': {
                'resolution': '1920x1200',
                'frequency': 60,
                'scale': 2,
            }
        }
        command = backend.build_command(config, [hidpi_panel], logger)
        assert command[command.index('--mode') + 1] == '1920x1200@60Hz'
        assert command[command.index('--scale') + 1] == '2'

    def test_exact_mode_gets_no_scale(self, backend, probed, logger):
        config = {
            'eDP-1': {'resolution': '1920x1080', 'frequency': 60},
            'HDMI-A-1': {'disabled': True},
        }
        assert '--scale' not in backend.build_command(config, probed, logger)

    def test_positions_use_the_requested_logical_size(
        self, backend, hidpi_panel, logger
    ):
        """The neighbour starts at 1920, not at the panel's native 3200."""
        external = {
            'device_name': 'DP-2',
            'product_id': '47PXNH3',
            'is_active': True,
            'is_connected': True,
            'resolution_modes': [
                {
                    'resolution_width': 3440,
                    'resolution_height': 1440,
                    'frequencies': [
                        {
                            'frequency': 99.982,
                            'is_current': True,
                            'is_preferred': True,
                        },
                    ],
                },
            ],
        }
        config = {
            'eDP-1': {'resolution': '1920x1200', 'frequency': 60},
            'DP-2': {'resolution': '3440x1440', 'right-of': 'eDP-1'},
        }
        command = backend.build_command(
            config, [hidpi_panel, external], logger
        )
        assert command[-1] == '1920,0'
