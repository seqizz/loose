#!/usr/bin/env python3

"""Display server backends for loose.

Everything that actually talks to a display server lives here. The rest of loose
only ever sees the normalized "identifier" dicts produced by ``Backend.probe()``:

    {
        'device_name': str,
        'product_id': str | int | None,   # stable identity: sorting + state diff
        'is_active': bool,
        'is_connected': bool,
        'resolution_modes': [
            {'resolution_width': int, 'resolution_height': int,
             'frequencies': [
                 {'frequency': float, 'is_current': bool, 'is_preferred': bool},
             ]},
        ],
    }

Any backend producing that shape works without touching downstream logic.
"""

import json
import logging
import subprocess
from os import environ
from shutil import which
from typing import Protocol

import jc

# Config keys that are not device names
RESERVED_KEYS = ('hooks', 'comment', 'is_current')
POSITION_KEYS = ('left-of', 'right-of', 'above', 'below')

# Fallback used by the position solver when a device has no resolvable mode
DEFAULT_LOGICAL_WIDTH = 1920
DEFAULT_LOGICAL_HEIGHT = 1080

# Auto-scaling only works when the requested and the native aspect ratios match,
# since the protocol has a single scalar scale. The tolerance absorbs the rounding
# in advertised mode sizes (e.g. 1366x768 is not exactly 16:9).
ASPECT_TOLERANCE = 0.002
# wlr-randr itself prints scales with 6 decimals, so match that
SCALE_DECIMALS = 6

# xrandr rotates the *content*, so "left" means counter-clockwise, which is
# transform 90 in the wlr-output-management protocol.
WLR_TRANSFORMS = {
    'normal': 'normal',
    'left': '90',
    'right': '270',
    'inverted': '180',
}


class BackendError(RuntimeError):
    """Raised when a backend cannot run: missing binary, unusable version, ..."""


class Backend(Protocol):
    """Contract every display server adapter has to fulfill"""

    name: str

    @staticmethod
    def is_available() -> bool:
        """Whether this backend could be used on this machine at all"""
        ...

    def probe(self) -> list[dict]:
        """Query the display server, return normalized identifiers"""
        ...

    def build_command(
        self,
        replaced_config: dict,
        identifiers: list[dict],
        logger: logging.Logger,
    ) -> list[str]:
        """Build the command line applying the given (alias-free) config"""
        ...

    @staticmethod
    def find_scaled_mode(
        device: dict, width: int, height: int, frequency: float | None
    ) -> tuple[int, int, float, float] | None:
        """A larger native mode that yields the requested logical size, if any.

        Returns (mode_width, mode_height, refresh, scale), or None when the
        backend cannot reach that size by scaling.
        """
        ...


def _sort_identifiers(identifiers: list[dict]) -> list[dict]:
    """Sorts by connection status (connected first), then by product_id/name

    Shared by all backends so the state file diff in main() compares
    like-for-like regardless of which backend produced the list.
    """
    identifiers.sort(
        key=lambda x: (
            not x['is_connected'],
            x['product_id'] or x['device_name'],
        )
    )
    return identifiers


def _find_device(identifiers: list[dict], device_name: str) -> dict | None:
    return next(
        (d for d in identifiers if d['device_name'] == device_name), None
    )


def _has_exact_mode(
    device: dict, width: int, height: int, frequency: float | None
) -> bool:
    """Whether the device advertises this resolution (at this rate) natively"""
    for mode in device.get('resolution_modes', []):
        if (
            mode['resolution_width'] != width
            or mode['resolution_height'] != height
        ):
            continue
        if frequency is None:
            return True
        if any(
            round(f['frequency']) == round(frequency)
            for f in mode['frequencies']
        ):
            return True
    return False


def _find_scaled_mode(
    device: dict, width: int, height: int, frequency: float | None
) -> tuple[int, int, float, float] | None:
    """Finds a larger native mode that scales down to the requested size.

    Wayland compositors advertise only the modes the panel actually reports,
    while the X server also synthesizes scaled ones. So a config asking a
    3200x2000 panel for 1920x1200 has no matching mode under wlroots — but
    driving the native mode at scale 1.6667 produces exactly that logical size.

    The aspect ratios have to match: the protocol has one scalar scale, so a
    16:9 request on a 16:10 panel is unreachable at any scale factor.
    """
    best = None
    for mode in device.get('resolution_modes', []):
        mode_width = mode['resolution_width']
        mode_height = mode['resolution_height']
        # Strictly larger only, an equal mode would have matched exactly
        if mode_width <= width or mode_height <= height:
            continue

        aspect_error = abs(mode_width * height - mode_height * width) / (
            mode_width * height
        )
        if aspect_error > ASPECT_TOLERANCE:
            continue

        frequencies = [f['frequency'] for f in mode['frequencies']]
        if frequency is not None:
            frequencies = [
                f for f in frequencies if round(f) == round(frequency)
            ]
        if not frequencies:
            continue

        # Highest resolution wins, so the panel keeps running at native
        if best is None or mode_width > best[0]:
            best = (
                mode_width,
                mode_height,
                max(frequencies),
                round(mode_width / width, SCALE_DECIMALS),
            )

    return best


def auto_scale_for(
    device_name: str,
    config: dict,
    identifiers: list[dict],
    backend,
) -> float | None:
    """The scale loose would apply on its own for this device, if any.

    An explicit "scale" in the config always wins, and a natively supported
    resolution never needs scaling.
    """
    if backend is None or 'scale' in config:
        return None

    resolution = config.get('resolution')
    if not resolution:
        return None
    try:
        width, height = (int(x) for x in resolution.split('x'))
    except ValueError:
        return None

    device = _find_device(identifiers, device_name)
    if device is None:
        return None

    frequency = config.get('frequency')
    if _has_exact_mode(device, width, height, frequency):
        return None

    scaled = backend.find_scaled_mode(device, width, height, frequency)
    return scaled[3] if scaled else None


def _get_preferred_mode(device_name: str, identifiers: list):
    """Find preferred resolution/frequency for a device from probe data."""
    for ident in identifiers:
        if ident['device_name'] == device_name:
            for mode in ident.get('resolution_modes', []):
                for freq in mode.get('frequencies', []):
                    if freq.get('is_preferred'):
                        res = (
                            f'{mode["resolution_width"]}'
                            f'x{mode["resolution_height"]}'
                        )
                        return res, freq['frequency']
    return None, None


# --- X11 / xrandr ---------------------------------------------------------


def parse_xrandr(props: bool = False) -> dict:
    """Parses the output of xrandr command and returns as dictionary"""

    command = ['xrandr']
    if props:
        command.append('--properties')
    try:
        outta = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as e:
        raise BackendError(
            f'xrandr failed (exit code {e.returncode}) — '
            'is DISPLAY set? (the x11 backend requires X11)'
        ) from e
    except FileNotFoundError:
        raise BackendError('xrandr not found — is it installed?')

    # It was horror trying to parse that ^bull(?:l+)?shit$ with regex myself
    # Kudos to jc: https://github.com/kellyjonbrazil/jc
    parsed_data = jc.parse('xrandr', outta)

    assert isinstance(parsed_data, dict)
    return parsed_data


class XrandrBackend:
    """X11 adapter, driving xrandr"""

    name = 'x11'

    @staticmethod
    def is_available() -> bool:
        return bool(which('xrandr'))

    @staticmethod
    def find_scaled_mode(
        device: dict, width: int, height: int, frequency: float | None
    ) -> tuple[int, int, float, float] | None:
        """Never auto-scales.

        The X server already synthesizes scaled modes, so a resolution missing
        here is genuinely unsupported. xrandr --scale would also mean the
        inverse of a Wayland scale factor.
        """
        return None

    def probe(self) -> list[dict]:
        xrandr_output = parse_xrandr()
        identifiers = []
        active_devices = []

        # First pass to get active devices
        for screen in xrandr_output['screens']:
            for device in screen['devices']:
                for resolution in device['resolution_modes']:
                    for frequency in resolution['frequencies']:
                        if frequency['is_current']:
                            active_devices.append(device['device_name'])

        # Get full device information including EDIDs
        parsed_props_xrandr = parse_xrandr(props=True)

        # Create a set of all device names from the basic xrandr output
        all_devices = {
            device['device_name']
            for screen in xrandr_output['screens']
            for device in screen['devices']
        }

        # Process connected devices first
        for screen in parsed_props_xrandr['screens']:
            for device in screen['devices']:
                if device['is_connected']:
                    resolution_modes = next(
                        (
                            d['resolution_modes']
                            for s in xrandr_output['screens']
                            for d in s['devices']
                            if d['device_name'] == device['device_name']
                        ),
                        [],
                    )

                    device_info = {
                        'device_name': device['device_name'],
                        'product_id': device.get('props', {})
                        .get('EdidModel', {})
                        .get('product_id'),
                        'is_active': device['device_name'] in active_devices,
                        'is_connected': True,
                        'resolution_modes': resolution_modes,
                    }
                    identifiers.append(device_info)
                    all_devices.remove(device['device_name'])

        # Now add disconnected devices, to disable later
        for device_name in all_devices:
            device_info = {
                'device_name': device_name,
                'product_id': None,
                'is_active': False,
                'is_connected': False,
                'resolution_modes': [],
            }
            identifiers.append(device_info)

        return _sort_identifiers(identifiers)

    def build_command(
        self,
        replaced_config: dict,
        identifiers: list[dict],
        logger: logging.Logger,
    ) -> list[str]:
        xrandr_binary = which('xrandr')
        if not xrandr_binary:
            raise BackendError('xrandr command could not be found in PATH!')

        xrandr_command = [xrandr_binary]
        # Configure devices mentioned in the config
        for device, config in replaced_config.items():
            if device in RESERVED_KEYS:
                continue
            xrandr_command += ['--output', device]
            if 'disabled' in config:
                xrandr_command += ['--off']
                continue

            if 'resolution' in config:
                xrandr_command += ['--mode', config['resolution']]
            else:
                xrandr_command += ['--auto']

            if 'primary' in config:
                xrandr_command += ['--primary']

            if 'rotate' in config:
                xrandr_command += ['--rotate', config['rotate']]
            else:
                xrandr_command += ['--rotate', 'normal']

            for position in POSITION_KEYS:
                if position in config:
                    xrandr_command += ['--' + position, config[position]]

            if 'frequency' in config:
                xrandr_command += ['--rate', str(config['frequency'])]

            if 'scale' in config:
                # xrandr does have --scale, but it means the opposite of what a
                # Wayland scale means: it renders a larger framebuffer and
                # downscales it onto the panel instead of doing HiDPI. Honoring
                # it here would make the same config behave inversely per backend.
                logger.warning(
                    f'"scale" is not supported by the x11 backend, ignoring it '
                    f'for device "{device}"'
                )

        # Turn off any device not explicitly configured
        unconfigured_devices = [
            device['device_name']
            for device in identifiers
            if device['device_name'] not in replaced_config
        ]
        for device in unconfigured_devices:
            xrandr_command.extend(['--output', device, '--off'])

        return xrandr_command


# --- wlroots / wlr-randr --------------------------------------------------


def _wlr_product_id(output: dict) -> str:
    """Builds a stable identity string for a wlr-randr output.

    This will never match the numeric EDID product code the X11 path gets from
    pyedid, and it does not need to: it is only used for sorting and for the
    "did the connected devices change" comparison in main().
    """
    serial = (output.get('serial') or '').strip()
    if serial:
        return serial

    make = (output.get('make') or '').strip()
    model = (output.get('model') or '').strip()
    if make or model:
        return f'{make} {model}'.strip()

    return output['name']


def _wlr_resolution_modes(modes: list) -> list[dict]:
    """Groups the flat wlr-randr mode list by resolution, preserving order"""
    grouped: dict = {}
    for mode in modes:
        key = (mode['width'], mode['height'])
        entry = grouped.setdefault(
            key,
            {
                'resolution_width': mode['width'],
                'resolution_height': mode['height'],
                'frequencies': [],
            },
        )
        refresh = mode['refresh']
        if refresh > 1000:
            # Some wlr-randr builds report mHz instead of Hz
            refresh = refresh / 1000
        entry['frequencies'].append(
            {
                'frequency': refresh,
                'is_current': bool(mode.get('current', False)),
                'is_preferred': bool(mode.get('preferred', False)),
            }
        )
    return list(grouped.values())


def _resolve_exact_frequency(
    device_name: str,
    resolution: str,
    wanted: float | None,
    identifiers: list[dict],
    logger: logging.Logger,
) -> float | None:
    """Resolves a configured refresh rate to the value the panel reports.

    A config saying "60" has to become the real "59.951" before it goes into a
    wlr-randr mode string. wlr-randr's own matching tolerance is undocumented,
    so resolving here keeps the result deterministic. Matching uses the same
    round() rule as _validate_device_compatibility() in loose.py.

    Returns None when there is nothing to resolve against, which makes the
    caller fall back to passing the configured value through.
    """
    try:
        width, height = (int(x) for x in resolution.split('x'))
    except ValueError:
        return None

    for device in identifiers:
        if device['device_name'] != device_name:
            continue
        for mode in device.get('resolution_modes', []):
            if (
                mode['resolution_width'] != width
                or mode['resolution_height'] != height
            ):
                continue
            frequencies = [f['frequency'] for f in mode['frequencies']]
            if not frequencies:
                return None
            if wanted is None:
                # No preference given, mirror "xrandr --mode" and take the fastest
                return max(frequencies)
            for frequency in frequencies:
                if round(frequency) == round(wanted):
                    return frequency
            logger.warning(
                f'Device "{device_name}" does not report {wanted}Hz at '
                f'{resolution}, passing the configured value through'
            )
            return None

    return None


def _wlr_output_mode(
    device_name: str,
    config: dict,
    identifiers: list[dict],
    logger: logging.Logger,
) -> tuple[str | None, float | None]:
    """Builds the --mode argument and the scale it needs, for one device.

    Returns (mode, auto_scale). A mode of None means "use --preferred", and an
    auto_scale of None means the mode is driven at its native size.
    """
    resolution = config.get('resolution')
    if not resolution:
        return None, None

    wanted = config.get('frequency')
    passthrough = (
        f'{resolution}@{wanted}Hz' if wanted is not None else resolution
    )

    try:
        width, height = (int(x) for x in resolution.split('x'))
    except ValueError:
        return passthrough, None

    device = _find_device(identifiers, device_name)
    if device is None:
        # Nothing to resolve against, hand the configured values over as-is
        return passthrough, None

    if _has_exact_mode(device, width, height, wanted):
        frequency = _resolve_exact_frequency(
            device_name=device_name,
            resolution=resolution,
            wanted=wanted,
            identifiers=identifiers,
            logger=logger,
        )
        if frequency is not None:
            return f'{resolution}@{frequency}Hz', None
        return passthrough, None

    if 'scale' not in config:
        scaled = _find_scaled_mode(device, width, height, wanted)
        if scaled:
            mode_width, mode_height, refresh, scale = scaled
            logger.info(
                f'Device "{device_name}" has no {resolution} mode, driving it '
                f'at {mode_width}x{mode_height}@{refresh}Hz with scale {scale} '
                'to reach that logical size'
            )
            return f'{mode_width}x{mode_height}@{refresh}Hz', scale

    logger.warning(
        f'Device "{device_name}" does not report a {resolution} mode and it '
        'cannot be reached by scaling, passing the configured value through'
    )
    return passthrough, None


def _logical_size(
    device_name: str,
    config: dict,
    identifiers: list[dict],
    logger: logging.Logger,
) -> tuple[int, int]:
    """Effective logical size of a device, as the compositor lays it out"""
    resolution = config.get('resolution')
    if not resolution:
        resolution, _ = _get_preferred_mode(device_name, identifiers)

    width, height = None, None
    if resolution:
        try:
            width, height = (int(x) for x in resolution.split('x'))
        except ValueError:
            width, height = None, None

    if not width or not height:
        logger.warning(
            f'Could not determine a resolution for "{device_name}", assuming '
            f'{DEFAULT_LOGICAL_WIDTH}x{DEFAULT_LOGICAL_HEIGHT} while calculating positions'
        )
        width, height = DEFAULT_LOGICAL_WIDTH, DEFAULT_LOGICAL_HEIGHT

    # wlroots lays outputs out in logical pixels, so scaling shrinks the footprint
    scale = config.get('scale') or 1
    width = round(width / scale)
    height = round(height / scale)

    if config.get('rotate') in ('left', 'right'):
        width, height = height, width

    return width, height


def _resolve_positions(
    replaced_config: dict,
    identifiers: list[dict],
    logger: logging.Logger,
) -> dict[str, tuple[int, int]]:
    """Turns relative positioning directives into absolute logical coordinates.

    xrandr resolves "--left-of eDP-1" itself, wlr-randr only takes "--pos X,Y",
    so we have to do the layout ourselves. Edge alignment matches xrandr's:
    neighbours share an edge and keep the other axis of their anchor.
    """
    devices = {}
    for name, config in replaced_config.items():
        if name in RESERVED_KEYS or not isinstance(config, dict):
            continue
        if 'disabled' in config:
            # A disabled output has no place in the layout
            continue
        devices[name] = config

    if not devices:
        return {}

    sizes = {
        name: _logical_size(name, config, identifiers, logger)
        for name, config in devices.items()
    }

    relations = {}
    for name, config in devices.items():
        for key in POSITION_KEYS:
            if key in config:
                relations[name] = (key, config[key])
                break

    roots = [name for name in devices if name not in relations]

    # A directive can point at a device that is disabled or absent from this
    # config. xrandr would error out, we degrade to an unpositioned root.
    for name, (key, target) in list(relations.items()):
        if target not in devices:
            logger.warning(
                f'Device "{name}" is placed "{key}" of "{target}", which is not '
                'an enabled device in this config. Placing it at 0,0 instead.'
            )
            del relations[name]
            roots.append(name)

    if len(roots) > 1:
        logger.warning(
            'Multiple devices without a positioning directive '
            f'({", ".join(sorted(roots))}), they will overlap at 0,0'
        )

    positions: dict[str, tuple[int, int]] = {name: (0, 0) for name in roots}

    # validate_config() already ran has_loops(), so this converges. The bound is
    # a safety net so a malformed graph fails visibly instead of hanging.
    for _ in range(len(devices)):
        if len(positions) == len(devices):
            break
        for name, (key, target) in relations.items():
            if name in positions or target not in positions:
                continue
            target_x, target_y = positions[target]
            target_width, target_height = sizes[target]
            width, height = sizes[name]
            if key == 'right-of':
                positions[name] = (target_x + target_width, target_y)
            elif key == 'left-of':
                positions[name] = (target_x - width, target_y)
            elif key == 'below':
                positions[name] = (target_x, target_y + target_height)
            elif key == 'above':
                positions[name] = (target_x, target_y - height)

    for name in devices:
        if name not in positions:
            logger.warning(
                f'Could not resolve a position for device "{name}", placing it at 0,0'
            )
            positions[name] = (0, 0)

    # Negative coordinates are legal in the protocol, but several compositors
    # handle them poorly, so shift everything into the positive quadrant.
    min_x = min(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())

    return {name: (x - min_x, y - min_y) for name, (x, y) in positions.items()}


class WlrootsBackend:
    """Wayland adapter for wlroots compositors, driving wlr-randr"""

    name = 'wlroots'

    @staticmethod
    def is_available() -> bool:
        return bool(which('wlr-randr'))

    @staticmethod
    def find_scaled_mode(
        device: dict, width: int, height: int, frequency: float | None
    ) -> tuple[int, int, float, float] | None:
        return _find_scaled_mode(device, width, height, frequency)

    @staticmethod
    def _binary() -> str:
        binary = which('wlr-randr')
        if not binary:
            raise BackendError(
                'wlr-randr could not be found in PATH! It is required for the '
                'wlroots backend, install it or force another backend with '
                'LOOSE_BACKEND=x11'
            )
        return binary

    def probe(self) -> list[dict]:
        try:
            output = subprocess.check_output(
                [self._binary(), '--json'], text=True
            )
        except subprocess.CalledProcessError as e:
            raise BackendError(
                f'wlr-randr failed (exit code {e.returncode}) — is '
                'WAYLAND_DISPLAY set and does the compositor implement '
                'wlr-output-management?'
            ) from e

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            raise BackendError(
                'Could not parse "wlr-randr --json" output — wlr-randr 0.3.0 '
                'or newer is required'
            ) from e

        if not isinstance(parsed, list):
            raise BackendError(
                'Unexpected "wlr-randr --json" output, expected a list of outputs'
            )

        identifiers = [
            {
                'device_name': output['name'],
                'product_id': _wlr_product_id(output),
                'is_active': bool(output.get('enabled', False)),
                # wlr-randr only ever lists connected heads
                'is_connected': True,
                'resolution_modes': _wlr_resolution_modes(
                    output.get('modes', [])
                ),
            }
            for output in parsed
        ]

        return _sort_identifiers(identifiers)

    def build_command(
        self,
        replaced_config: dict,
        identifiers: list[dict],
        logger: logging.Logger,
    ) -> list[str]:
        binary = self._binary()
        positions = _resolve_positions(
            replaced_config=replaced_config,
            identifiers=identifiers,
            logger=logger,
        )

        command = [binary]
        for device, config in replaced_config.items():
            if device in RESERVED_KEYS:
                continue
            command += ['--output', device]
            if 'disabled' in config:
                command += ['--off']
                continue

            command += ['--on']

            mode, auto_scale = _wlr_output_mode(
                device_name=device,
                config=config,
                identifiers=identifiers,
                logger=logger,
            )
            if mode:
                command += ['--mode', mode]
            else:
                command += ['--preferred']

            if 'primary' in config:
                logger.debug(
                    'wlr-randr has no concept of a primary output, ignoring '
                    f'"primary" for device "{device}"'
                )

            rotate = config.get('rotate', 'normal')
            transform = WLR_TRANSFORMS.get(rotate)
            if transform is None:
                logger.warning(
                    f'Unknown rotate value "{rotate}" for device "{device}", '
                    'using "normal"'
                )
                transform = 'normal'
            command += ['--transform', transform]

            if 'scale' in config:
                command += ['--scale', str(config['scale'])]
            elif auto_scale:
                command += ['--scale', str(auto_scale)]

            if device in positions:
                x, y = positions[device]
                command += ['--pos', f'{x},{y}']

        # Turn off any device not explicitly configured
        unconfigured_devices = [
            device['device_name']
            for device in identifiers
            if device['device_name'] not in replaced_config
        ]
        for device in unconfigured_devices:
            command.extend(['--output', device, '--off'])

        return command


# --- Selection ------------------------------------------------------------

BACKENDS = {
    'x11': XrandrBackend,
    'xrandr': XrandrBackend,
    'wlroots': WlrootsBackend,
    'wayland': WlrootsBackend,
}


def detect_backend(logger: logging.Logger) -> Backend:
    """Picks the backend for the current session.

    LOOSE_BACKEND is the escape hatch, otherwise the session environment decides.
    """
    forced = environ.get('LOOSE_BACKEND')
    if forced:
        backend_class = BACKENDS.get(forced.lower())
        if not backend_class:
            raise BackendError(
                f'Unknown LOOSE_BACKEND value "{forced}", expected one of: '
                f'{", ".join(sorted(BACKENDS))}'
            )
        logger.debug(f'Backend forced to "{forced}" via LOOSE_BACKEND')
        return backend_class()

    # WAYLAND_DISPLAY has to win: under Wayland, DISPLAY is usually also set by
    # XWayland, and xrandr there happily reports a single synthetic output
    # instead of failing. Checking DISPLAY first would silently apply nonsense.
    if environ.get('WAYLAND_DISPLAY'):
        logger.debug('WAYLAND_DISPLAY is set, using the wlroots backend')
        return WlrootsBackend()

    if environ.get('DISPLAY'):
        if WlrootsBackend.is_available():
            logger.warning(
                'DISPLAY is set but WAYLAND_DISPLAY is not, while wlr-randr is '
                'installed. Falling back to the x11 backend. If this is a '
                'wlroots session (e.g. loose started from a systemd user unit), '
                'the environment is incomplete — try "systemctl --user '
                'import-environment WAYLAND_DISPLAY".'
            )
        logger.debug('DISPLAY is set, using the x11 backend')
        return XrandrBackend()

    raise BackendError(
        'No display server detected, neither WAYLAND_DISPLAY nor DISPLAY is set'
    )
