"""Tests for backend selection from the session environment.

The critical case is a Wayland session where XWayland also set DISPLAY:
picking X11 there would silently apply a config against a single synthetic
xrandr output instead of failing.
"""

import pytest

from loose.backends import (
    BackendError,
    WlrootsBackend,
    XrandrBackend,
    detect_backend,
)

DISPLAY_VARS = ('LOOSE_BACKEND', 'WAYLAND_DISPLAY', 'DISPLAY')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a session with no display environment at all."""
    for var in DISPLAY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_wlr_randr(mocker):
    """Pretend wlr-randr is not installed (the common X11-only machine)."""
    mocker.patch('loose.backends.which', return_value=None)


class TestDetectBackend:
    def test_wayland_only(self, monkeypatch, logger, no_wlr_randr):
        monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-1')
        assert isinstance(detect_backend(logger), WlrootsBackend)

    def test_display_only(self, monkeypatch, logger, no_wlr_randr):
        monkeypatch.setenv('DISPLAY', ':0')
        assert isinstance(detect_backend(logger), XrandrBackend)

    def test_both_set_prefers_wayland(self, monkeypatch, logger, no_wlr_randr):
        """XWayland sets DISPLAY too, so WAYLAND_DISPLAY has to win."""
        monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-1')
        monkeypatch.setenv('DISPLAY', ':0')
        assert isinstance(detect_backend(logger), WlrootsBackend)

    def test_neither_set_raises(self, logger, no_wlr_randr):
        with pytest.raises(BackendError, match='No display server'):
            detect_backend(logger)

    def test_display_with_wlr_randr_warns(
        self, monkeypatch, mocker, logger, caplog
    ):
        """Likely a wlroots session whose unit never got WAYLAND_DISPLAY."""
        mocker.patch('loose.backends.which', return_value='/usr/bin/wlr-randr')
        monkeypatch.setenv('DISPLAY', ':0')
        with caplog.at_level('WARNING'):
            backend = detect_backend(logger)

        assert isinstance(backend, XrandrBackend)
        assert 'import-environment' in caplog.text

    def test_display_without_wlr_randr_is_quiet(
        self, monkeypatch, logger, caplog, no_wlr_randr
    ):
        monkeypatch.setenv('DISPLAY', ':0')
        with caplog.at_level('WARNING'):
            detect_backend(logger)

        assert caplog.text == ''

    @pytest.mark.parametrize('value', ['x11', 'X11', 'xrandr'])
    def test_forced_x11(self, monkeypatch, logger, value, no_wlr_randr):
        monkeypatch.setenv('LOOSE_BACKEND', value)
        monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-1')
        assert isinstance(detect_backend(logger), XrandrBackend)

    @pytest.mark.parametrize('value', ['wlroots', 'wayland', 'WLROOTS'])
    def test_forced_wlroots(self, monkeypatch, logger, value, no_wlr_randr):
        monkeypatch.setenv('LOOSE_BACKEND', value)
        monkeypatch.setenv('DISPLAY', ':0')
        assert isinstance(detect_backend(logger), WlrootsBackend)

    def test_forced_works_without_any_display_var(
        self, monkeypatch, logger, no_wlr_randr
    ):
        monkeypatch.setenv('LOOSE_BACKEND', 'x11')
        assert isinstance(detect_backend(logger), XrandrBackend)

    def test_unknown_value_is_a_hard_error(self, monkeypatch, logger):
        """Not a silent fallthrough: a typo here would pick the wrong tool."""
        monkeypatch.setenv('LOOSE_BACKEND', 'mutter')
        monkeypatch.setenv('DISPLAY', ':0')
        with pytest.raises(BackendError, match='mutter'):
            detect_backend(logger)
