import pytest

from loose.backends import WlrootsBackend, XrandrBackend
from loose.loose import _validate_device_compatibility


class TestValidateDeviceCompatibility:
    def test_matching_resolution(self, device_edp1, logger):
        config = {'resolution': '1920x1080'}
        assert (
            _validate_device_compatibility(device_edp1, config, '_1', logger)
            is True
        )

    def test_mismatching_resolution(self, device_edp1, logger):
        config = {'resolution': '3440x1440'}
        assert (
            _validate_device_compatibility(device_edp1, config, '_1', logger)
            is False
        )

    def test_matching_resolution_and_frequency(self, device_hdmi1, logger):
        config = {'resolution': '3440x1440', 'frequency': 100}
        assert (
            _validate_device_compatibility(device_hdmi1, config, '_1', logger)
            is True
        )

    def test_matching_resolution_wrong_frequency(self, device_hdmi1, logger):
        config = {'resolution': '3440x1440', 'frequency': 144}
        assert (
            _validate_device_compatibility(device_hdmi1, config, '_1', logger)
            is False
        )

    def test_no_resolution_specified(self, device_edp1, logger):
        """When no resolution is requested, any device is compatible."""
        config = {}
        assert (
            _validate_device_compatibility(device_edp1, config, '_1', logger)
            is True
        )

    def test_frequency_only_no_resolution(self, device_edp1, logger):
        """Frequency without resolution: resolution check skipped (needed_x is None)."""
        config = {'frequency': 60}
        assert (
            _validate_device_compatibility(device_edp1, config, '_1', logger)
            is True
        )

    def test_second_resolution_mode(self, device_hdmi1, logger):
        """HDMI-1 also supports 1920x1200@60Hz as a secondary mode."""
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert (
            _validate_device_compatibility(device_hdmi1, config, '_1', logger)
            is True
        )

    def test_fractional_frequency_rounds_to_match(self, logger):
        """Real xrandr reports e.g. 59.95 Hz — config with 60 should match."""
        device = {
            'device_name': 'eDP-1',
            'product_id': '0x1234',
            'is_active': True,
            'is_connected': True,
            'resolution_modes': [
                {
                    'resolution_width': 1920,
                    'resolution_height': 1200,
                    'frequencies': [
                        {
                            'frequency': 59.95,
                            'is_current': True,
                            'is_preferred': True,
                        },
                        {
                            'frequency': 59.88,
                            'is_current': False,
                            'is_preferred': False,
                        },
                    ],
                },
            ],
        }
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert (
            _validate_device_compatibility(device, config, '_1', logger)
            is True
        )

    def test_fractional_frequency_no_false_positive(self, logger):
        """59.3 Hz should NOT round to 60."""
        device = {
            'device_name': 'eDP-1',
            'product_id': '0x1234',
            'is_active': True,
            'is_connected': True,
            'resolution_modes': [
                {
                    'resolution_width': 1920,
                    'resolution_height': 1200,
                    'frequencies': [
                        {
                            'frequency': 59.3,
                            'is_current': True,
                            'is_preferred': True,
                        },
                    ],
                },
            ],
        }
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert (
            _validate_device_compatibility(device, config, '_1', logger)
            is False
        )

    def test_empty_resolution_modes(self, device_dp2_disconnected, logger):
        """Disconnected device with no modes: resolution check fails."""
        config = {'resolution': '1920x1080'}
        assert (
            _validate_device_compatibility(
                device_dp2_disconnected, config, '_1', logger
            )
            is False
        )


# --- backend-assisted scaling ---


class TestScalableCompatibility:
    """A resolution the panel cannot do natively may still be reachable.

    Without a backend the check stays a literal mode lookup, which is what
    the x11 path wants anyway.
    """

    @pytest.fixture
    def hidpi_panel(self):
        return {
            'device_name': 'eDP-1',
            'product_id': '0x41A2',
            'is_active': False,
            'is_connected': True,
            'resolution_modes': [
                {
                    'resolution_width': 3200,
                    'resolution_height': 2000,
                    'frequencies': [
                        {
                            'frequency': 60.0,
                            'is_current': False,
                            'is_preferred': True,
                        },
                    ],
                },
            ],
        }

    def test_rejected_without_a_backend(self, hidpi_panel, logger):
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert (
            _validate_device_compatibility(hidpi_panel, config, '_1', logger)
            is False
        )

    def test_rejected_by_x11_backend(self, hidpi_panel, logger):
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert (
            _validate_device_compatibility(
                hidpi_panel, config, '_1', logger, XrandrBackend()
            )
            is False
        )

    def test_accepted_by_wlroots_backend(self, hidpi_panel, logger):
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert (
            _validate_device_compatibility(
                hidpi_panel, config, '_1', logger, WlrootsBackend()
            )
            is True
        )

    def test_aspect_mismatch_still_rejected(self, hidpi_panel, logger):
        config = {'resolution': '1920x1080', 'frequency': 60}
        assert (
            _validate_device_compatibility(
                hidpi_panel, config, '_1', logger, WlrootsBackend()
            )
            is False
        )

    def test_explicit_scale_disables_the_fallback(self, hidpi_panel, logger):
        """With an explicit scale the resolution has to be a real mode."""
        config = {'resolution': '1920x1200', 'frequency': 60, 'scale': 2}
        assert (
            _validate_device_compatibility(
                hidpi_panel, config, '_1', logger, WlrootsBackend()
            )
            is False
        )
