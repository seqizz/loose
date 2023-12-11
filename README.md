# Loose 🫠

This is a helper tool to apply vague Xrandr configurations

> This is still an alpha software, use at your own risk. I am planning to add some features and polish it a bit more before calling it stable.

### Why?
(Warning: Includes rant)

I wrote this as a reaction to all other xrandr automation tools which requires you to exactly define your setup (e.g. which monitor has which damn fingerprint, what input name it has etc.) and then failing to set up since you plugged a cable on different port.

Come on dude, I just want to define some non-specific settings. It should not be that hard to say something like:

- If I plug 1 extra monitor, my preference list:
    1. Switch off the laptop's monitor and only use plugged monitor
    2. Put the new monitor to right of my laptop's screen
    3. Put the new monitor to left of my laptop's screen and rotate it 90 degrees
- If I plug 2 monitors:
    1. Switch off the laptop's monitor and use monitor 1 on the left, 2 on the right
    2. Switch off the laptop's monitor and use monitor 2 on the left, 1 on the right

### Installation (TODO?)

Since I am using NixOS, I will be using this with dark magic. Instructions for other distros are welcome, since this tool will be maximally useful if it is installed system-wide and triggered by udev automatically.

For the testing/development purposes, you can use poetry:
- Clone this repo
- Install poetry
- Run `poetry install` in the repo

### Usage

- Set up your desired configuration in a yaml file after reading the [example config](loose/example_config.yaml)
- You can declare either explicit monitor names or just alises which starts with underscore while writing your configuration
- Check if the config is valid with `loose show`
- Once dust settles after screen connections, trigger `loose rotate`
- It doesn't care if external monitor referred by the name "DP-1", "DP1", "DP-1-2" or "duck99"
- It will detect and apply first preference
- You didn't like this preference? Just trigger `rotate` command again (preferably bound to a keyboard shortcut), and it will switch to next declared preference
- For non-explicit names, it will try to be consistent with detection of screen names (thus you can declare self-correcting preferences like "2 monitors" example above)
- If you don't give specific resolution, it'll use the preferred one (which comes from `xrandr` itself)
- Something failed bad? Then it will try the global failback (which is defined in config file)

Please see [example config](loose/example_config.yaml) for examples and explanations.

(This was _not_ easy to write btw)

![protip: it was hard](https://paste.gurkan.in/good-sculpin.jpg)

### TODO

- Implement pre/post commands (good for setting DPI etc.)
- Add tests
- Add gotchas about aliases (I had a lot of assumptions while writing this, which I need to document later)
