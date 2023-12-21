# Loose 🫠

![dall-e is stupid as hell](https://paste.gurkan.in/merry-impala.com-resize.jpg)

This is a helper tool to apply vague Xrandr configurations

> This is still an alpha software, use at your own risk. I am planning to add some features and polish it a bit more before recommending it.

### Why?
(Warning: Includes rant)

I wrote this as a reaction to all other xrandr automation tools which requires you to exactly define your setup (e.g. which monitor has which damn fingerprint, what input name it has etc.) and then failing to set up, since you plugged a cable on a different port.

Come on, I just want to define some non-specific settings. Something like:

- If I plug 1 extra monitor, my preference list:
    1. Switch off the laptop's monitor and only use plugged monitor
    2. Put the new monitor to right of my laptop's screen
    3. Put the new monitor to left of my laptop's screen and rotate it 90 degrees
- If I plug 2 monitors:
    1. Switch off the laptop's monitor and use monitor 1 on the left, 2 on the right
    2. Switch off the laptop's monitor and use monitor 2 on the left, 1 on the right

### Features

- Configured with a yaml file (I know, I know..)
- Accepts both explicit monitor names and aliases to define purposefully vague configurations
- Supports setting the resolution, rotation, (relative) position, refresh rate and primary monitor
- If you don't define a setting (e.g. resolution or refresh rate), it will apply the "preferred" one from Xrandr
- Supports multiple pre/post commands to run before/after applying the configuration (e.g. to set the DPI)
- Supports multiple configurations for multiple screen counts so you can "rotate" if you don't like the first applied one
- It will try to detect the suitable devices for given aliases consistently (thus you can declare self-correcting preferences like "2 monitors" example above)
- Something failed bad? Then it will try the global failback option defined in the config

I highly recommend checking the [example config](loose/example_config.yaml) for examples and explanations.

### Installation (What installation?)

Since I am using NixOS, I will be using this with dark magic. Instructions for other distros are welcome, since this tool will be maximally useful if it is installed system-wide and triggered by udev automatically.

For the testing/development purposes, you can use poetry:
- Clone this repo
- Install poetry
- Run `poetry install` in the repo

### TODO

Todo is tracked on my [issues](https://git.gurkan.in/gurkan/loose/issues) page.
