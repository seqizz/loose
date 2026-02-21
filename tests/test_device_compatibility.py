from loose.loose import _validate_device_compatibility


class TestValidateDeviceCompatibility:
    def test_matching_resolution(self, device_edp1, logger):
        config = {'resolution': '1920x1080'}
        assert _validate_device_compatibility(device_edp1, config, '_1', logger) is True

    def test_mismatching_resolution(self, device_edp1, logger):
        config = {'resolution': '3440x1440'}
        assert _validate_device_compatibility(device_edp1, config, '_1', logger) is False

    def test_matching_resolution_and_frequency(self, device_hdmi1, logger):
        config = {'resolution': '3440x1440', 'frequency': 100}
        assert _validate_device_compatibility(device_hdmi1, config, '_1', logger) is True

    def test_matching_resolution_wrong_frequency(self, device_hdmi1, logger):
        config = {'resolution': '3440x1440', 'frequency': 144}
        assert _validate_device_compatibility(device_hdmi1, config, '_1', logger) is False

    def test_no_resolution_specified(self, device_edp1, logger):
        """When no resolution is requested, any device is compatible."""
        config = {}
        assert _validate_device_compatibility(device_edp1, config, '_1', logger) is True

    def test_frequency_only_no_resolution(self, device_edp1, logger):
        """Frequency without resolution: resolution check skipped (needed_x is None)."""
        config = {'frequency': 60}
        assert _validate_device_compatibility(device_edp1, config, '_1', logger) is True

    def test_second_resolution_mode(self, device_hdmi1, logger):
        """HDMI-1 also supports 1920x1200@60Hz as a secondary mode."""
        config = {'resolution': '1920x1200', 'frequency': 60}
        assert _validate_device_compatibility(device_hdmi1, config, '_1', logger) is True

    def test_empty_resolution_modes(self, device_dp2_disconnected, logger):
        """Disconnected device with no modes: resolution check fails."""
        config = {'resolution': '1920x1080'}
        assert (
            _validate_device_compatibility(device_dp2_disconnected, config, '_1', logger)
            is False
        )
