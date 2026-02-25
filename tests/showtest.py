#!/usr/bin/env python3
"""Visual demo of the show() ASCII monitor layout rendering.

Not a pytest test — run directly: python tests/showtest.py
"""

from loose.loose import _render_config_layout

# Fake xrandr identifiers for preferred resolution lookup
IDENTIFIERS = [
    {
        'device_name': 'eDP-1',
        'is_connected': True, 'is_active': True,
        'resolution_modes': [
            {'resolution_width': 1920, 'resolution_height': 1080,
             'frequencies': [
                 {'frequency': 60, 'is_current': True, 'is_preferred': True},
             ]},
        ],
    },
    {
        'device_name': 'HDMI-1',
        'is_connected': True, 'is_active': True,
        'resolution_modes': [
            {'resolution_width': 2560, 'resolution_height': 1440,
             'frequencies': [
                 {'frequency': 144, 'is_current': True, 'is_preferred': True},
             ]},
        ],
    },
    {
        'device_name': 'DP-1',
        'is_connected': True, 'is_active': True,
        'resolution_modes': [
            {'resolution_width': 3440, 'resolution_height': 1440,
             'frequencies': [
                 {'frequency': 100, 'is_current': True, 'is_preferred': True},
             ]},
        ],
    },
    {
        'device_name': 'DP-2',
        'is_connected': True, 'is_active': True,
        'resolution_modes': [
            {'resolution_width': 1920, 'resolution_height': 1200,
             'frequencies': [
                 {'frequency': 60, 'is_current': True, 'is_preferred': True},
             ]},
        ],
    },
]


def _print_layout(title, config, identifiers=None):
    print(f'\n{title}:')
    lines, used_pref, used_pri = _render_config_layout(
        config, identifiers=identifiers or IDENTIFIERS
    )
    for line in lines:
        print(line)
    if used_pri:
        print(' \u00b9 primary display')
    if used_pref:
        print(' * preferred display defaults')


# --- Scenario 1: Monitor above another ---
_print_layout(
    'Scenario 1 — HDMI-1 above eDP-1',
    {
        'eDP-1': {
            'resolution': '1920x1080', 'frequency': 60, 'primary': True,
        },
        'HDMI-1': {
            'resolution': '2560x1440', 'frequency': 144, 'above': 'eDP-1',
        },
    },
)

# --- Scenario 2: Three monitors — above + left-of ---
_print_layout(
    'Scenario 2 — HDMI-1 above eDP-1, DP-1 left-of eDP-1',
    {
        'eDP-1': {
            'resolution': '1920x1080', 'frequency': 60, 'primary': True,
        },
        'HDMI-1': {
            'resolution': '2560x1440', 'frequency': 144, 'above': 'eDP-1',
        },
        'DP-1': {
            'resolution': '3440x1440', 'frequency': 100, 'left-of': 'eDP-1',
        },
    },
)

# --- Scenario 3: Vertical + horizontal side by side ---
_print_layout(
    'Scenario 3 — DP-2 rotated left, eDP-1 right-of DP-2',
    {
        'DP-2': {
            'resolution': '1920x1200', 'frequency': 60,
            'rotate': 'left', 'primary': True,
        },
        'eDP-1': {
            'resolution': '1920x1080', 'frequency': 60,
            'right-of': 'DP-2',
        },
    },
)

# --- Scenario 4: Dock setup — laptop disabled, two externals + preferred ---
_print_layout(
    'Scenario 4 — eDP-1 disabled, DP-1 primary, HDMI-1 right-of DP-1 (preferred)',
    {
        'eDP-1': {'disabled': True},
        'DP-1': {
            'resolution': '3440x1440', 'frequency': 100, 'primary': True,
        },
        'HDMI-1': {
            'right-of': 'DP-1',
        },
    },
)

# --- Scenario 5: Four monitors — grid layout ---
_print_layout(
    'Scenario 5 — 2x2 grid: eDP-1 top-left, HDMI-1 right, DP-1 below eDP-1, DP-2 below HDMI-1',
    {
        'eDP-1': {
            'resolution': '1920x1080', 'frequency': 60, 'primary': True,
        },
        'HDMI-1': {
            'resolution': '2560x1440', 'frequency': 144,
            'right-of': 'eDP-1',
        },
        'DP-1': {
            'resolution': '3440x1440', 'frequency': 100,
            'below': 'eDP-1',
        },
        'DP-2': {
            'resolution': '1920x1200', 'frequency': 60,
            'below': 'HDMI-1',
        },
    },
)

# --- Scenario 6: Single fallback with alias (no xrandr match) ---
_print_layout(
    'Scenario 6 — Global failback with alias',
    {'_1': {}},
)
