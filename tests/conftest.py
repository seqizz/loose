import logging
from copy import deepcopy

import pytest

from loose.loose import VERSION


@pytest.fixture
def logger():
    """A real logger at DEBUG level for test inspection."""
    log = logging.getLogger('loose_test')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    return log


@pytest.fixture
def device_edp1():
    """Connected, active laptop display: 1920x1080@60Hz."""
    return {
        'device_name': 'eDP-1',
        'product_id': '0x1234',
        'is_active': True,
        'is_connected': True,
        'resolution_modes': [
            {
                'resolution_width': 1920,
                'resolution_height': 1080,
                'frequencies': [
                    {'frequency': 60, 'is_current': True, 'is_preferred': True},
                ],
            },
        ],
        'aliases': [],
    }


@pytest.fixture
def device_hdmi1():
    """Connected, active external display: 3440x1440@100Hz + 1920x1200@60Hz."""
    return {
        'device_name': 'HDMI-1',
        'product_id': '0x5678',
        'is_active': True,
        'is_connected': True,
        'resolution_modes': [
            {
                'resolution_width': 3440,
                'resolution_height': 1440,
                'frequencies': [
                    {'frequency': 100, 'is_current': True, 'is_preferred': True},
                ],
            },
            {
                'resolution_width': 1920,
                'resolution_height': 1200,
                'frequencies': [
                    {'frequency': 60, 'is_current': False, 'is_preferred': False},
                ],
            },
        ],
        'aliases': [],
    }


@pytest.fixture
def device_dp1():
    """Connected, inactive display: 1920x1200@60Hz."""
    return {
        'device_name': 'DP-1',
        'product_id': '0x9ABC',
        'is_active': False,
        'is_connected': True,
        'resolution_modes': [
            {
                'resolution_width': 1920,
                'resolution_height': 1200,
                'frequencies': [
                    {'frequency': 60, 'is_current': False, 'is_preferred': True},
                ],
            },
        ],
        'aliases': [],
    }


@pytest.fixture
def device_dp2_disconnected():
    """Disconnected display, no modes."""
    return {
        'device_name': 'DP-2',
        'product_id': None,
        'is_active': False,
        'is_connected': False,
        'resolution_modes': [],
    }


@pytest.fixture
def make_main_dict():
    """Factory fixture: call with identifiers list and active_config list."""

    def _make(identifiers, active_config, raw_config=None):
        return {
            'identifiers': deepcopy(identifiers),
            'active_config': deepcopy(active_config),
            'VERSION': VERSION,
            'raw_config': raw_config or {},
        }

    return _make
