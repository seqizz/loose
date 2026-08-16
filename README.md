# Loose 🫠

![dall-e is stupid as hell](https://paste.gurkan.in/merry-impala.com-resize.jpg)

A display layout wrapper that lets you define layouts by monitor capabilities, makes flexible configurations possible. Works on X11 (xrandr) and wlroots compositors (wlr-randr).

### Use case (or why I wrote this)

> I wrote this as a reaction to all other xrandr tools/wrappers which requires you to exactly define your setup (e.g. which monitor has which damn fingerprint, what input name it has etc.) and then failing to set up, since you plugged a cable on a different port.

Come on, I just want to be able to define some "preferences". Something like this:

- If I plug 1 extra monitor, I want either:
    1. Switch off the laptop's monitor and only use plugged monitor
    2. Put the new monitor to right of my laptop's screen
    3. Put the new monitor to left of my laptop's screen and rotate it 90 degrees
- If I plug 2 monitors, I want either:
    1. Switch off the laptop's monitor and use monitor 1 on the left, 2 on the right
    2. Switch off the laptop's monitor and use monitor 2 on the left, 1 on the right
- And so on...

With loose, I can do it!

### Features

- Configured with a yaml file (I know, I know..)
- Accepts both explicit monitor names and aliases/placeholders to define "intentions" (e.g. "3 monitors" or "4k monitor")
- Supports setting the resolution, rotation, (relative) position, refresh rate and primary monitor
- If you don't define a setting (e.g. resolution or refresh rate), it will apply the "preferred" one from Xrandr
- Supports multiple pre/post commands to run before/after applying the configuration (e.g. to set the DPI or restart WM)
- Supports multiple configurations for multiple screen counts so you can "rotate" if you don't like the first applied one
- It will try to detect and match suitable devices for given aliases consistently (thus you can declare self-correcting preferences like "2 monitors" example above)
- Something failed bad? It will try the global failback option defined in the config
- Interactive rotation mode (`loose rotate --interactive`): previews each config's `comment` via a configurable notification command and waits a bit before applying, so you can skip ahead by rotating again during the preview window
- Runs on X11 and on wlroots-based Wayland compositors, with the same config file

### Usage

Just check the [example config](loose/example_config.yaml) for real-life examples and explanations.

### Backends

loose picks a backend from the session environment. `WAYLAND_DISPLAY` is checked
before `DISPLAY`, because XWayland sets `DISPLAY` too and xrandr there reports a
single synthetic output instead of failing.

| Backend | Tool | Selected when |
|---|---|---|
| `x11` | `xrandr` | `DISPLAY` is set and `WAYLAND_DISPLAY` is not |
| `wlroots` | `wlr-randr` (>= 0.3.0) | `WAYLAND_DISPLAY` is set |

Set `LOOSE_BACKEND=x11` or `LOOSE_BACKEND=wlroots` to override the detection.

Not every config key exists on both sides. Keys a backend cannot honor are skipped
with a log message, so one config file stays portable:

| Key | x11 | wlroots |
|---|---|---|
| `resolution`, `frequency`, `rotate`, `disabled` | yes | yes |
| positioning (`left-of`, `right-of`, `above`, `below`) | yes | yes, resolved into absolute coordinates |
| `primary` | yes | ignored (no such concept in wlr-output-management) |
| `scale` | ignored with a warning | yes (HiDPI scaling) |

`scale` is deliberately not mapped onto `xrandr --scale`: xrandr renders a larger
framebuffer and downscales it onto the panel, which is the inverse of what a Wayland
scale factor does.

Two notes for wlroots setups:

- `wlr-randr` is not a hard dependency of the default Nix package, so X11-only
  machines don't pull in the Wayland closure. Either install it yourself or use the
  `loose-wayland` package output, which wraps loose with `wlr-randr` on `PATH`.
- If loose runs from a systemd user unit, that unit needs `WAYLAND_DISPLAY` in its
  environment (`systemctl --user import-environment WAYLAND_DISPLAY`). Without it,
  loose warns and falls back to X11.

The first run after switching backends on the same machine will report a device
change and start from scratch: the two backends derive `product_id` differently (EDID
product code vs. serial/make/model). That is expected, not a bug.

### Installation

Since I am using NixOS, I am using this with dark magic ([systemd service](https://git.gurkan.in/gurkan/nixos-system-flake/src/commit/914d4f0ae730780c5240befa3bb9b746c46dc1ad/home-manager/lib/xserver.nix#L18), [udev rules](https://git.gurkan.in/gurkan/nixos-system-flake/src/commit/914d4f0ae730780c5240befa3bb9b746c46dc1ad/nixos/lib/laptop/loose.nix#L8)).
Instructions for other distros are welcome, since I don't have enough incentive to write them (plus this tool is only really useful if it's integrated with an init service + udev).

For the testing/development purposes, you can use uv:
- Clone this repo
- Install uv
- Run `uv sync` in the repo
