#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from importlib.metadata import version as pkg_version
from importlib.util import find_spec
from os import environ
from os.path import (
    abspath,
    dirname,
    exists,
    join as path_join,
)
from pathlib import Path
from pprint import pprint
from shutil import which
import jc
import yamale
from filelock import FileLock, Timeout
from xdg_base_dirs import xdg_config_home, xdg_state_home
from yaml import safe_load

DEFAULT_CONFIG_FILE = f'{xdg_config_home()}/loose/config.yaml'
PY_MAJOR_VERSION = 3
PY_MINOR_VERSION = 10
RUN_TIMEOUT_SEC = 30  # In case of a stuck process
DEFAULT_INTERACTIVE_SLEEP_SEC = 3
DEFAULT_INTERACTIVE_COMMAND = 'notify-send loose {comment} -t {timeout_ms}'
VERSION = pkg_version('loose')


def build_main_dict(config: dict) -> dict:
    xrandr_output = parse_xrandr()
    main_dict = {'identifiers': []}
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
                main_dict['identifiers'].append(device_info)
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
        main_dict['identifiers'].append(device_info)

    # Sort identifiers first by connection status (connected first), then by product_id/name
    main_dict['identifiers'].sort(
        key=lambda x: (
            not x['is_connected'],
            x['product_id'] or x['device_name'],
        )
    )
    main_dict['VERSION'] = VERSION
    main_dict['raw_config'] = deepcopy(config)

    return main_dict


def run_command(
    command: str,
    logger: logging.Logger,
) -> int:
    """Runs given command and returns the return code

    :param command: The command to run
    :param logger: The logger object

    :return: The return code of the command

    """

    try:
        result = subprocess.run(
            command,
            timeout=RUN_TIMEOUT_SEC,
            env=environ,  # Pass the current environment
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
    except subprocess.TimeoutExpired:
        logger.critical(
            f'Command took more than {RUN_TIMEOUT_SEC} seconds, aborted: {command}'
        )
        return 1

    logger.debug(f'Return code was: {result.returncode}')

    if result.returncode == 0:
        return 0

    logger.warning(f'Command failed: {command}')
    logger.info(f'Error output: {result.stderr.decode("utf-8")}')
    logger.info(f'Standard output: {result.stdout.decode("utf-8")}')
    return result.returncode


class MyFormatter(
    argparse.RawTextHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    # Small hack to make use both formatters at the same time
    pass


def get_parser(print_help: bool) -> argparse.Namespace:
    """Returns the argument parser object

    :param print_help: Whether to print help and early exit

    :return: The argument parser object
    """

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description='\n'.join(
            [
                'Welcome to loose 🫠',
                '',
                'loose is a tool to manage your multi-monitor setup on Linux.',
                'It allows you to define your desired setup in a YAML file ',
                'in a flexible way and rotate between them.',
                '',
                'Feel free to use --help toggle for each subcommand below',
            ]
        ),
    )
    # We will add shtab, but only if it is installed
    # Used to include shell completion in the package
    if find_spec('shtab'):
        import shtab

        shtab.add_argument_to(parser, ['-s', '--shell-completion'])
    parser.add_argument(
        '-V',  # Capital to not conflict with --verbose
        '--version',
        action='store_true',
        help='Print version and exit',
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='\n'.join(
            [
                'Location of the config file',
                f'(Defaults to {DEFAULT_CONFIG_FILE})',
            ]
        ),
        default=DEFAULT_CONFIG_FILE,
    )
    sub = parser.add_subparsers(dest='command')
    common_options = argparse.ArgumentParser()
    common_options.add_argument(
        '-n',
        '--dry-run',
        action='store_true',
        help='Do not apply any hooks or xrandr commands, just print them',
    )
    common_options.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Print debug messages to stdout also',
    )
    common_options.add_argument(
        '-d',
        '--dump-state',
        action='store_true',
        help='Dump saved state (if any) to stdout and exit',
    )
    rotate_parser = sub.add_parser(
        'rotate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Switch to the next definition on configuation',
        description='\n'.join(
            [
                'Switch to the next definition on configuation.',
                '',
                'If there is no currently applied configuration,',
                'the first one will be applied.',
                '',
                'Example usage:',
                '  loose rotate',
            ]
        ),
        parents=[common_options],
        add_help=False,
    )
    rotate_group = rotate_parser.add_mutually_exclusive_group()
    rotate_group.add_argument(
        '-r',
        '--reset',
        action='store_true',
        help='Do not check for the next config, apply the first one',
    )
    rotate_group.add_argument(
        '-e',
        '--ensure',
        action='store_true',
        help='Only apply changes if connected devices or config changed',
    )
    rotate_group.add_argument(
        '--interactive',
        action='store_true',
        help='Preview next config via notification before applying, skip with subsequent rotations',
    )
    rotate_parser.add_argument(
        '-i',
        '--ignore-failing-hooks',
        action='store_true',
        help='Ignore failing pre-hooks and continue applying xrandr commands, useful for initial runs',
    )
    sub.add_parser(
        'show',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Print the active configuration',
        description='\n'.join(
            [
                'Print the active configuration.',
                '',
                'Useful to check validated configuration and aliases,',
                'as well as next configuration(s) to be applied.',
                '',
                'Example usage:',
                '  loose show',
            ]
        ),
        parents=[common_options],
        add_help=False,
    )
    sub.add_parser(
        'generate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Generate a sample configuration file',
        description='\n'.join(
            [
                'Generate a sample configuration file.',
                '',
                'Useful to pipe it into config location and start as a template.',
                '',
                'Example usages:',
                '  loose generate',
                '  loose generate > ~/.config/loose/config.yaml',
            ]
        ),
        parents=[common_options],
        add_help=False,
    )
    args = parser.parse_args()

    # Let's handle one-shot actions first, so we can exit early
    if print_help:
        parser.print_help()
        exit(0)

    if args.version:
        print(f'loose 🫠 version: {VERSION}')
        exit(0)

    if args.command == 'generate':
        # Example config is in the same folder as the script
        current_folder = dirname(abspath(__file__))
        schema_file = path_join(current_folder, 'example_config.yaml')
        with open(schema_file) as file_stream:
            print(file_stream.read())
        exit(0)

    return args


def enforce_python_version():
    # 3.7+ because of the ordered dicts
    # 3.10+ because of xdg-base-dirs
    if (
        sys.version_info.major != PY_MAJOR_VERSION
        or sys.version_info.minor < PY_MINOR_VERSION
    ):
        print(
            f'This script requires Python {PY_MAJOR_VERSION}.{PY_MINOR_VERSION}+'
        )
        sys.exit(1)


def _safe_file_operation(
    file_path: str, operation: str, logger: logging.Logger, data=None
):
    """Safely handle file operations with consistent error handling"""
    try:
        if operation == 'read_yaml':
            with open(file_path) as file_stream:
                return safe_load(file_stream)
        elif operation == 'read_text':
            with open(file_path) as file_stream:
                return file_stream.read()
        elif operation == 'write_pickle':
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
        elif operation == 'read_pickle':
            if not exists(file_path):
                return None
            with open(file_path, 'rb') as file:
                return pickle.load(file)
    except FileNotFoundError:
        if operation == 'read_yaml':
            logger.error(f'Config file not found at: {file_path} !')
            exit(1)
        return None
    except Exception as e:
        logger.error(f'Error during {operation} on {file_path}: {str(e)}')
        return None


def save_to_disk(
    applied_config: dict,
    full_config: dict,
    logger: logging.Logger,
    old_dict: dict,
    save_path: str,
):
    # Save the dictionary to a file using pickle
    config_rotation = deepcopy(old_dict['active_config'])
    for config in config_rotation:
        if 'is_current' in config and config['is_current']:
            if config != applied_config:
                # Remove the current tag from the old config
                del config['is_current']
        if config == applied_config:
            config['is_current'] = True

    # Instead of creating a new main_dict, let's update the existing one
    new_dict = deepcopy(old_dict)
    new_dict['active_config'] = config_rotation
    new_dict['raw_config'] = full_config

    logger.debug('Saving new active config to disk')
    _safe_file_operation(save_path, 'write_pickle', logger, new_dict)


def load_from_disk(filename: str, logger: logging.Logger):
    # Load the dictionary from the pickle file
    return _safe_file_operation(filename, 'read_pickle', logger)


def parse_xrandr(props: bool = False) -> dict:
    """Parses the output of xrandr command and returns as dictionary"""

    command = ['xrandr']
    if props:
        command.append('--properties')
    try:
        outta = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f'xrandr failed (exit code {e.returncode}) — '
            'is DISPLAY set? (loose requires X11)'
        ) from e
    except FileNotFoundError:
        raise RuntimeError('xrandr not found — is it installed?')

    # It was horror trying to parse that ^bull(?:l+)?shit$ with regex myself
    # Kudos to jc: https://github.com/kellyjonbrazil/jc
    parsed_data = jc.parse('xrandr', outta)

    assert isinstance(parsed_data, dict)
    return parsed_data


def assert_unique_primary(data):
    """Asserts that there is only one primary screen for each config section"""
    for screen_count, config_list in data['on_screen_count'].items():
        for single_config in config_list:
            primary_count = sum(
                1
                for _, config in single_config.items()
                if isinstance(config, dict) and config.get('primary', False)
            )
            if primary_count > 1:
                raise ValueError(
                    f'Multiple "primary" entries found within the same '
                    f'configuration list under on_screen_count->{screen_count} '
                    f'(index {config_list.index(single_config)})'
                )


def validate_config(
    config_dict: dict, config_file: str, logger: logging.Logger
) -> None:
    """Validates the config file

    We use yamale for schema validation, but we also have to check for loops

    :param config: The config dictionary
    :param config_file: The config file path
    :param logger: The logger object
    """

    # Check for logical loops in the config
    loop_fail, message = has_loops(config_dict['on_screen_count'])
    if loop_fail:
        logger.error(
            "Config file has loops, please remember we don't allow "
            '"below/above/left-of/right-of" definitions bi-directionally '
            'between screens (or self-references). Your detected issue was:\n\n'
            f'{message}'
        )
        exit(1)

    # Validate the data against the correct schema
    current_folder = dirname(abspath(__file__))
    schema_file = path_join(current_folder, 'config_schema.yaml')

    schema = yamale.make_schema(schema_file)
    data = yamale.make_data(config_file)
    try:
        yamale.validate(schema, data)
        assert_unique_primary(config_dict)
        logger.debug('Validation of configuration successful.')
    except Exception as e:
        logger.error('Validation of configuration failed. Details:')
        print(str(e))
        exit(1)


def has_loops(on_screen_config) -> tuple[bool, str]:
    """Detects whether there is a loop in the graph, for position references

    We don't allow:
        - Self-reference (e.g. _X: {below: _X})
        - Level-1 reference loops for "any" directives (e.g. _X: {below: _Y}, _Y: {above: _X})

    Credit: ChatGPT4 (No way in hell I can write this myself)
    """

    # Function to build graph from configuration and check for loops
    def build_graph_and_check_loops(screens):
        # Create an adjacency list to represent the graph
        graph = defaultdict(list)
        # Iterate through each screen ID and their properties
        for screen_id, properties in screens.items():
            # Iterate through properties to find directional references
            for direction, ref_id in properties.items():
                # Check only directional properties
                if direction in ['above', 'below', 'left-of', 'right-of']:
                    # Check for self-reference
                    if ref_id == screen_id:
                        # Return True for loop detected and a message
                        return (
                            True,
                            'There is a self-reference to screen object itself',
                        )
                    # Add a directed edge from current screen to referenced screen
                    graph[screen_id].append(ref_id)

        # Function for performing Depth-First Search
        def dfs(node_id, graph, visited, rec_stack):
            # If the node is in the recursion stack, a loop is detected
            if node_id in rec_stack:
                return (
                    True,
                    'There are objects referring to each other within same config section',
                )
            # If the node was already visited, skip it
            if node_id in visited:
                return False, ''
            # Mark the node as visited and add to recursion stack
            visited.add(node_id)
            rec_stack.add(node_id)
            # Recursively visit all adjacent nodes
            for neighbor_id in graph[node_id]:
                has_loop, message = dfs(neighbor_id, graph, visited, rec_stack)
                if has_loop:
                    # If a loop is detected in the DFS, propagate the result up
                    return True, message
            # Remove the current node from recursion stack after DFS completes
            rec_stack.remove(node_id)
            return False, ''

        # Sets to keep track of visited nodes and the recursion stack
        visited, rec_stack = set(), set()
        # Obtain a list of the nodes to iterate over without changing the dict's size
        nodes = list(graph.keys())
        # Perform DFS on each node
        for node_id in nodes:
            if node_id not in visited:
                has_loop, message = dfs(node_id, graph, visited, rec_stack)
                if has_loop:
                    # If a loop is found, return True and the accompanying message
                    return True, message.format(config=node_id)
        # If no loops are found in the graph, return False with an empty message
        return False, ''

    # Iterate over each screen configuration index and the corresponding sections
    for _, screens_list in on_screen_config.items():
        for _, screen_section in enumerate(screens_list):
            # Ensure the section items are dictionaries before processing
            screens = {
                k: v for k, v in screen_section.items() if isinstance(v, dict)
            }
            # Use the helper function to check for loops in the current section
            has_loop, message = build_graph_and_check_loops(screens)
            if has_loop:
                # If a loop is detected, return the information immediately
                return True, message

    # Return False and an empty message if no loops are found in any configuration
    return False, ''


def _replace_none_with_dict(d):
    """Replaces None values with empty dicts in a nested dictionary"""
    for k, v in d.items():
        if isinstance(v, dict):  # If the item is a dict, recurse into it
            _replace_none_with_dict(v)
        elif v is None:  # Replace None with an empty dict
            d[k] = {}


def read_config(config_file: str, logger: logging.Logger) -> dict:
    config = _safe_file_operation(config_file, 'read_yaml', logger)

    assert isinstance(config, dict)
    if 'on_screen_count' in config:
        for _, level in config['on_screen_count'].items():
            if isinstance(level, list):
                for item in level:
                    if item is None:
                        # If the item itself is None, replace it with an empty dict
                        index = level.index(item)
                        level[index] = {}
                    elif isinstance(item, dict):
                        _replace_none_with_dict(item)

    return config


def find_real_device_name(
    alias: str,
    identifiers: list,
    logger: logging.Logger,
) -> str:
    """Returns the real device name for the given alias"""

    for device in identifiers:
        if device['is_connected'] and alias in device.get(
            'aliases', [device['device_name']]
        ):
            return device['device_name']

    logger.error(f'No real device found for alias "{alias}"')
    exit(1)


def replace_aliases_with_real_names(
    main_dict: dict,
    config_to_convert: dict,
    logger: logging.Logger,
) -> dict:
    """Replaces the aliases in the config with the real device names"""
    replaced_config = {}

    # First rename the aliases to the actual device names
    for alias, config in config_to_convert.items():
        if alias == 'is_current':
            # This is a special key, ignore it
            continue

        interim_config = {}
        if alias == 'hooks':
            replaced_config[alias] = config
            continue
        if alias == 'comment':
            continue
        if alias.startswith('_'):
            # This is an alias, replace it with the actual device name
            real_name = find_real_device_name(
                alias=alias,
                identifiers=main_dict['identifiers'],
                logger=logger,
            )
        else:
            real_name = alias
        interim_config[real_name] = config

        # There might also be positioning directives, we have to replace them too
        for key, value in config.items():
            if key in [
                'left-of',
                'right-of',
                'above',
                'below',
            ] and value.startswith('_'):
                interim_config[real_name][key] = find_real_device_name(
                    alias=value,
                    identifiers=main_dict['identifiers'],
                    logger=logger,
                )
        replaced_config.update(interim_config)

    return replaced_config


def apply_xrandr_command(
    main_dict: dict,
    config_to_apply: dict,
    logger: logging.Logger,
    dry_run: bool,
    ignore_failing_hooks: bool = False,
) -> bool:
    """Applies the given config to the xrandr output"""

    xrandr_binary = which('xrandr')
    if not xrandr_binary:
        logger.error('xrandr command could not be found in PATH!')
        return False

    replaced_config = replace_aliases_with_real_names(
        main_dict=main_dict,
        config_to_convert=config_to_apply,
        logger=logger,
    )

    xrandr_command = [xrandr_binary]
    # Configure devices mentioned in the config
    for device, config in replaced_config.items():
        if device in ('hooks', 'comment'):
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

        for position in ['left-of', 'right-of', 'above', 'below']:
            if position in config:
                xrandr_command += ['--' + position, config[position]]

        if 'frequency' in config:
            xrandr_command += ['--rate', str(config['frequency'])]

    # Turn off any device not explicitly configured
    unconfigured_devices = [
        device['device_name']
        for device in main_dict['identifiers']
        if device['device_name'] not in replaced_config
    ]
    for device in unconfigured_devices:
        xrandr_command.extend(['--output', device, '--off'])

    # Execute pre-hooks, xrandr command, and post-hooks
    if not _execute_hooks(
        config=replaced_config,
        hook_type='pre',
        logger=logger,
        dry_run=dry_run,
        ignore_failing_hooks=ignore_failing_hooks,
    ):
        return False

    if not _execute_xrandr(xrandr_command, replaced_config, logger, dry_run):
        return False

    _execute_hooks(
        config=replaced_config,
        hook_type='post',
        logger=logger,
        dry_run=dry_run,
        ignore_failing_hooks=ignore_failing_hooks,
    )
    return True


def _execute_hooks(
    config: dict,
    hook_type: str,
    logger: logging.Logger,
    dry_run: bool,
    ignore_failing_hooks: bool = False,
) -> bool:
    """Execute pre or post hooks, returns False only for pre-hook failures"""
    if 'hooks' not in config or hook_type not in config['hooks']:
        return True

    for hook in config['hooks'][hook_type]:
        if dry_run:
            logger.info(f'DRY RUN: Would run {hook_type}-hook: {hook}')
            continue

        logger.info(f'Running {hook_type}-hook: {hook}')
        result = run_command(command=hook, logger=logger)
        if result != 0:
            logger.error(
                f'{hook_type.capitalize()}-hook "{hook}" failed! {"Continuing anyway..." if hook_type == "post" else ""}'
            )
            if hook_type == 'pre' and not ignore_failing_hooks:
                return False
    return True


def _execute_xrandr(
    command: list, config: dict, logger: logging.Logger, dry_run: bool
) -> bool:
    """Execute the xrandr command"""
    if dry_run:
        logger.info(f'DRY RUN: Would run command: {" ".join(command)}')
        logger.debug(f'Config for command: {config}')
        return True
    logger.info(f'Running command: {" ".join(command)}')
    logger.debug(f'Config for command: {config}')
    result = run_command(command=' '.join(command), logger=logger)
    if result != 0:
        logger.error('xrandr command failed!')
        return False
    return True


def get_logger(verbose: bool = False) -> logging.Logger:
    """Creates and returns logger from logging lib"""

    logger = logging.getLogger('loose')

    formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def clear_impossible_configs(main_dict: dict, logger: logging.Logger) -> dict:
    """Removes the configs that are impossible to apply"""

    temp_dict_active_config = main_dict['active_config'].copy()

    # Remove configs for disconnected devices
    for device in [
        d for d in main_dict['identifiers'] if not d['is_connected']
    ]:
        for config in temp_dict_active_config:
            if (
                config in main_dict['active_config']
            ):  # Check if not already removed
                for device_name in config:
                    if device_name == device['device_name']:
                        logger.debug(
                            f'Ignoring config "{config}" since device "{device_name}" is not connected'
                        )
                        main_dict['active_config'].remove(config)

    # Remove configs with non-existent devices
    connected_device_names = [
        d['device_name'] for d in main_dict['identifiers'] if d['is_connected']
    ]
    for config in main_dict['active_config'].copy():
        for device in config:
            if (
                device != 'hooks'  # Skip hooks section
                and device != 'comment'  # Skip comment
                and not device.startswith('_')  # Skip aliases
                and device not in connected_device_names
            ):
                # Maybe just removed because of another nonexistent device
                if config in main_dict['active_config']:
                    logger.debug(
                        f'Ignoring config "{config}" because of missing device "{device}"'
                    )
                    main_dict['active_config'].remove(config)

    return main_dict


def _validate_device_compatibility(
    device: dict, config: dict, alias: str, logger: logging.Logger
) -> bool:
    """Validates if a device is compatible with the given config section"""
    needed_x, needed_y = None, None
    if 'resolution' in config:
        needed_x, needed_y = (int(x) for x in config['resolution'].split('x'))
    needed_frequency = config.get('frequency')

    # Validate resolution and frequency together when both are specified
    if needed_x and needed_frequency:
        if not any(
            mode['resolution_width'] == needed_x
            and mode['resolution_height'] == needed_y
            and any(
                round(f['frequency']) == needed_frequency
                for f in mode['frequencies']
            )
            for mode in device['resolution_modes']
        ):
            logger.debug(
                f'Config with alias "{alias}" is not applicable to device "{device["device_name"]}" due to resolution+frequency mismatch'
            )
            return False
    elif needed_x:
        # Validate resolution only
        if not any(
            mode['resolution_width'] == needed_x
            and mode['resolution_height'] == needed_y
            for mode in device['resolution_modes']
        ):
            logger.debug(
                f'Config with alias "{alias}" is not applicable to device "{device["device_name"]}" due to resolution mismatch'
            )
            return False
    elif needed_frequency:
        # Validate frequency only
        if not any(
            round(frequency['frequency']) == needed_frequency
            for mode in device['resolution_modes']
            for frequency in mode['frequencies']
        ):
            logger.debug(
                f'Config with alias "{alias}" is not applicable to device "{device["device_name"]}" due to frequency mismatch'
            )
            return False

    return True


def assign_aliases(main_dict: dict, logger: logging.Logger) -> dict:
    """
    Assigns aliases from the config to connected devices based on compatibility
    and ensures a 1:1 mapping during the assignment process.
    Filters down main_dict['active_config'] to only include configs
    whose keys can be mapped 1:1 to the assigned aliases/devices.
    """
    connected_devices = [
        d for d in main_dict['identifiers'] if d['is_connected']
    ]
    connected_device_names = {d['device_name'] for d in connected_devices}

    # Clear previous alias assignments in the identifiers list
    for device in main_dict['identifiers']:
        if device['is_connected']:
            device['aliases'] = []  # Reset aliases list

    # clear_impossible_configs removes configs referencing non-connected explicit device names
    # It's good to keep this initial filter.
    main_dict = clear_impossible_configs(main_dict=main_dict, logger=logger)

    # Get all potential keys (aliases/names) used in the remaining active_config list
    unassigned_keys_pool = set()
    for item in main_dict['active_config']:
        for key in item.keys():
            if key != 'hooks' and key != 'is_current' and key != 'comment':
                unassigned_keys_pool.add(key)

    # Track which physical devices have been claimed by a key (explicit name or alias)
    claimed_devices = set()
    # Track which keys from the pool have been successfully assigned to a device
    assigned_keys_from_pool = set()

    logger.debug(
        f'Potential keys from config pool: {sorted(list(unassigned_keys_pool))}'
    )

    # Pass 1: Prioritize explicit device names used as keys in config
    # These keys map directly to a connected device by name. We just need to check compatibility.
    explicit_device_name_keys = sorted(
        [k for k in unassigned_keys_pool if k in connected_device_names]
    )
    logger.debug(
        f'Pass 1: Checking explicit device names used in config pool: {explicit_device_name_keys}'
    )

    for key in explicit_device_name_keys:
        # Find the device matching this explicit name key
        device = next(
            (d for d in connected_devices if d['device_name'] == key), None
        )

        # Should always find the device if key was in connected_device_names,
        # and ensure it hasn't been claimed by another explicit name already (unlikely)
        if device and device['device_name'] not in claimed_devices:
            # Check compatibility: Does this device match requirements in *any* config section using this key?
            is_compatible = False
            for config_item in main_dict['active_config']:
                if key in config_item:
                    # Check compatibility of this device against the config section using this key
                    if _validate_device_compatibility(
                        device, config_item[key], key, logger
                    ):
                        is_compatible = True
                        break  # Found compatibility

            if is_compatible:
                logger.debug(
                    f'Assigning explicit device name "{key}" to device "{device["device_name"]}"'
                )
                device['aliases'].append(
                    key
                )  # Assign the key as an alias to the device
                claimed_devices.add(device['device_name'])  # Claim this device
                assigned_keys_from_pool.add(key)  # Mark this key as assigned
                # Move to the next explicit name key.

    # Pass 2: Assign _X aliases to remaining unclaimed devices
    # Sort aliases to give _1 priority over _2 etc.
    alias_keys = sorted([k for k in unassigned_keys_pool if k.startswith('_')])
    logger.debug(
        f'Pass 2: Checking alias keys used in config pool: {alias_keys}'
    )

    for key in alias_keys:
        if key in assigned_keys_from_pool:
            continue  # This alias key has already been assigned

        # Find the first available connected device compatible with this alias key
        assigned_device = None
        for device in connected_devices:
            if device['device_name'] in claimed_devices:
                continue  # Device already claimed by an explicit name or previous alias

            # Check compatibility: Does this device match requirements in *any* config section using this key?
            is_compatible = False
            for config_item in main_dict['active_config']:
                if key in config_item:
                    if _validate_device_compatibility(
                        device, config_item[key], key, logger
                    ):
                        is_compatible = True
                        break  # Found compatibility

            if is_compatible:
                # This is the first compatible, unclaimed device found for this alias key
                assigned_device = device
                break  # Found a device for this alias key, stop checking other devices

        if assigned_device:
            # Assign this alias key to the found device
            logger.debug(
                f'Assigning alias "{key}" to device "{assigned_device["device_name"]}"'
            )
            assigned_device['aliases'].append(
                key
            )  # Assign the key as an alias
            claimed_devices.add(
                assigned_device['device_name']
            )  # Claim the device
            assigned_keys_from_pool.add(key)  # Mark the alias key as assigned
            # Move to the next alias key.

    # --- Filtering applicable configs based on assignment results ---
    # Keep only the config dictionaries from the original active_config list
    # where all their required keys (excluding hooks and is_current)
    # were successfully assigned to unique devices based on the global assignment done above.

    original_active_configs = deepcopy(
        main_dict['active_config']
    )  # Work on a copy for filtering
    main_dict[
        'active_config'
    ] = []  # Reset the active_config list to build the filtered list

    logger.debug('Filtering config candidates based on alias assignments')
    for config_candidate_original in original_active_configs:
        # Need a temporary dict to work with keys easily, preserving original for the final list
        config_candidate_keys_only = {
            k: v
            for k, v in config_candidate_original.items()
            if k != 'is_current'
        }

        required_keys = {
            k for k in config_candidate_keys_only.keys()
            if k not in ('hooks', 'comment')
        }

        # For a candidate config to be applicable, every required_key must be in the assigned_keys_from_pool
        # AND the devices these keys were assigned to must be unique *within this candidate config*.
        candidate_claimed_devices_for_mapping = set()
        is_candidate_applicable = True

        for required_key in required_keys:
            # Check if the required key was assigned at all during the global assignment phase
            if required_key not in assigned_keys_from_pool:
                logger.debug(
                    f'Config candidate requires key "{required_key}" which was not assigned to any device. Candidate not applicable.'
                )
                is_candidate_applicable = False
                break

            # Find the unique device assigned to this key globally
            assigned_device = next(
                (
                    d
                    for d in main_dict['identifiers']
                    if required_key in d.get('aliases', [])
                ),
                None,
            )

            # assigned_device should always be found if required_key is in assigned_keys_from_pool, but check for robustness
            if not assigned_device:
                logger.warning(
                    f'Internal logic issue: Key "{required_key}" in assigned_keys_from_pool but no device found with this alias. Candidate not applicable.'
                )
                is_candidate_applicable = False
                break

            # Check if this assigned device is already claimed by another key *within this specific candidate config's requirements*
            if (
                assigned_device['device_name']
                in candidate_claimed_devices_for_mapping
            ):
                logger.debug(
                    f'Config candidate requires key "{required_key}" which maps to device "{assigned_device["device_name"]}", but device is already claimed by another key in this config. Candidate not applicable.'
                )
                is_candidate_applicable = False
                break

            # Device is unique for this candidate config's requirements. Claim it for this candidate's mapping check.
            candidate_claimed_devices_for_mapping.add(
                assigned_device['device_name']
            )

        # If the candidate is applicable after checking all its required keys:
        if is_candidate_applicable:
            # Add the original config dictionary back to the filtered list.
            # 'is_current' and other potential non-key fields are preserved.
            main_dict['active_config'].append(config_candidate_original)
            logger.debug('Config candidate deemed applicable.')

    # Log determined aliases (these are the global potential assignments based on the 1:1 rule)
    # This output will now reflect that each device gets at most one _X alias.
    for device in main_dict['identifiers']:
        if device['is_connected']:
            logger.debug(
                f'Determined aliases for device "{device["device_name"]}": {", ".join(device.get("aliases", []))}'
            )

    # Log which keys from the initial pool were not assigned globally.
    # Any config requiring these keys will have been filtered out.
    all_assigned_aliases = set()
    for device in main_dict['identifiers']:
        if device['is_connected']:
            all_assigned_aliases.update(device.get('aliases', []))

    unassigned_keys_after_assignment = (
        unassigned_keys_pool - all_assigned_aliases
    )
    if unassigned_keys_after_assignment:
        # This is expected if a config uses a key (alias or name), but no compatible device
        # was available or claimed by a higher priority key during assignment.
        # Config candidates requiring these keys have already been filtered out.
        logger.debug(
            f'Keys from config pool that could not be assigned to any device: {sorted(list(unassigned_keys_after_assignment))}'
        )

    return main_dict  # Return updated main_dict


def get_next_config(
    active_config: list,
    logger: logging.Logger,
) -> dict:
    """Get the xrandr output, return the next config in the list"""
    # Check if there is a currently applied config
    # If there is, rotate to the next one
    # If there isn't, apply the first one

    for config in active_config:
        if 'is_current' in config:
            next_config = active_config[
                (active_config.index(config) + 1) % len(active_config)
            ]
            logger.debug(f'Rotating to the next config: {next_config}')
            return next_config

    logger.debug(
        f'No active configuration found, applying the first config: {active_config[0]}'
    )

    return active_config[0]


# Used for debugging purposes, sometimes
def _print_and_exit(anyobject):
    pprint(anyobject, width=1)
    exit(0)


def apply_global_failback(
    main_dict: dict,
    config: dict,
    logger: logging.Logger,
    dry_run: bool,
):
    """Applies the global failback directive

    Meant to be called when there is no possible config found to apply
    """
    logger.warning(
        f'{"Would apply" if dry_run else "Applying"} global failback directive.'
    )
    if 'global_failback' not in config:
        logger.error(
            "Can't even find global_failback directive in the config, "
            f'{"would exit" if dry_run else "exiting"}!'
        )
        exit(bool(not dry_run))

    # Can't even check the return of this, what are we going to do, exit? 😒
    apply_xrandr_command(
        main_dict=main_dict,
        config_to_apply=config['global_failback'],
        logger=logger,
        dry_run=dry_run,
    )

    # Failback implies error
    exit(bool(not dry_run))


def get_active_config(
    main_dict: dict, config: dict, logger: logging.Logger, dry_run: bool
) -> dict:
    connected_count = len(
        [x for x in main_dict['identifiers'] if x['is_connected']]
    )
    if connected_count not in config['on_screen_count']:
        logger.warning(
            f'No config found for {connected_count} connected screens!'
        )
        apply_global_failback(
            main_dict=main_dict,
            config=config,
            logger=logger,
            dry_run=dry_run,
        )

    return config['on_screen_count'][connected_count]


def _show_notification(
    command: str, comment: str, sleep_sec: float, logger: logging.Logger
) -> None:
    """Show the preview using the configured command template.

    {comment} is replaced with the shell-quoted comment text,
    {timeout_ms} with the sleep duration in milliseconds.
    """
    rendered = command.replace('{comment}', shlex.quote(comment)).replace(
        '{timeout_ms}', str(int(sleep_sec * 1000))
    )
    run_command(command=rendered, logger=logger)


def rotate_interactive(
    args: argparse.Namespace,
    full_config: dict,
    logger: logging.Logger,
    main_dict: dict,
    save_file: str,
    ignore_failing_hooks: bool = False,
) -> None:
    """Interactive rotation with comment preview and SIGUSR1-based skipping."""
    pid_path = path_join(dirname(save_file), 'loose.preview_pid')
    active = main_dict['active_config']

    # Find current index in rotation
    current_index = 0
    for i, config in enumerate(active):
        if 'is_current' in config:
            current_index = i
            break

    interactive_opts = full_config.get('interactive') or {}
    sleep_sec = interactive_opts.get('sleep_sec', DEFAULT_INTERACTIVE_SLEEP_SEC)
    notify_command = interactive_opts.get(
        'command', DEFAULT_INTERACTIVE_COMMAND
    )

    advance_requested = False

    def handle_sigusr1(signum, frame):
        nonlocal advance_requested
        advance_requested = True

    signal.signal(signal.SIGUSR1, handle_sigusr1)

    # Write PID so other loose processes can signal us
    try:
        with open(pid_path, 'w') as f:
            f.write(str(os.getpid()))
    except OSError as e:
        logger.debug(f'Could not write preview PID file: {e}')

    try:
        while True:
            # Advance to next config in rotation
            current_index = (current_index + 1) % len(active)
            next_config = active[current_index]

            comment = next_config.get(
                'comment', 'Next option (no comment found)'
            )
            logger.info(f'Preview: {comment}')
            _show_notification(
                command=notify_command,
                comment=comment,
                sleep_sec=sleep_sec,
                logger=logger,
            )

            # Wait up to sleep_sec for SIGUSR1 (user wants to skip)
            advance_requested = False
            deadline = time.time() + sleep_sec
            while time.time() < deadline:
                if advance_requested:
                    advance_requested = False
                    break
                time.sleep(0.1)
            else:
                # Timeout reached, apply this config
                run_result = apply_xrandr_command(
                    main_dict=main_dict,
                    config_to_apply=next_config,
                    logger=logger,
                    dry_run=args.dry_run,
                    ignore_failing_hooks=ignore_failing_hooks,
                )
                if not run_result:
                    logger.error('Failed to apply the config, exiting!')
                    exit(1)
                if not args.dry_run:
                    save_to_disk(
                        applied_config=next_config,
                        full_config=full_config,
                        logger=logger,
                        old_dict=main_dict,
                        save_path=save_file,
                    )
                return
    finally:
        try:
            os.remove(pid_path)
        except FileNotFoundError:
            pass


def rotate(
    args: argparse.Namespace,
    full_config: dict,
    logger: logging.Logger,
    main_dict: dict,
    save_file: str,
    ignore_failing_hooks: bool = False,
):
    """Rotates the current config to the next one"""
    logger.debug(
        f'Got request to rotate.{" (DRY RUN)" if args.dry_run else ""}'
    )

    if getattr(args, 'interactive', False):
        rotate_interactive(
            args=args,
            full_config=full_config,
            logger=logger,
            main_dict=main_dict,
            save_file=save_file,
            ignore_failing_hooks=ignore_failing_hooks,
        )
        return

    # Find next configuration to apply
    next_config = get_next_config(
        active_config=main_dict['active_config'],
        logger=logger,
    )

    # Apply the next config, shit gets real here, unless dry run is requested
    run_result = apply_xrandr_command(
        main_dict=main_dict,
        config_to_apply=next_config,
        logger=logger,
        dry_run=args.dry_run,
        ignore_failing_hooks=ignore_failing_hooks,
    )

    if not run_result:
        logger.error('Failed to apply the config, exiting!')
        exit(1)

    if not args.dry_run:
        # Looks like success, save the state to disk with new current tag
        save_to_disk(
            applied_config=next_config,
            full_config=full_config,
            logger=logger,
            old_dict=main_dict,
            save_path=save_file,
        )


def _build_monitor_grid(config: dict) -> dict:
    """Assigns (row, col) grid coordinates to each device based on positioning.

    Returns {device_name: (row, col)} mapping.
    """
    positions = {}
    positioning_keys = {'left-of', 'right-of', 'above', 'below'}

    # Collect positioning relationships
    relations = {}  # device -> (directive, target)
    devices = [d for d in config if d not in ('hooks', 'comment')]
    for device in devices:
        props = config[device]
        if not isinstance(props, dict):
            continue
        for key in positioning_keys:
            if key in props:
                relations[device] = (key, props[key])
                break

    # Place devices that are not positioned relative to others first
    # (i.e. "anchor" devices that others reference)
    placed = set()
    col_counter = 0

    # Find anchors: devices not in relations (no positioning directive)
    anchors = [d for d in devices if d not in relations]
    if not anchors and devices:
        # All devices have positioning; pick the first one as anchor
        anchors = [devices[0]]

    for anchor in anchors:
        positions[anchor] = (0, col_counter)
        placed.add(anchor)
        col_counter += 1

    # Iteratively place devices that reference already-placed devices
    max_iterations = len(devices) * 2
    for _ in range(max_iterations):
        if len(placed) == len(devices):
            break
        for device in devices:
            if device in placed:
                continue
            if device not in relations:
                positions[device] = (0, col_counter)
                placed.add(device)
                col_counter += 1
                continue
            directive, target = relations[device]
            if target not in positions:
                continue
            tr, tc = positions[target]
            if directive == 'right-of':
                positions[device] = (tr, tc + 1)
            elif directive == 'left-of':
                positions[device] = (tr, tc - 1)
            elif directive == 'below':
                positions[device] = (tr + 1, tc)
            elif directive == 'above':
                positions[device] = (tr - 1, tc)
            placed.add(device)

    # Place any remaining unresolved devices
    for device in devices:
        if device not in placed:
            positions[device] = (0, col_counter)
            col_counter += 1

    # Normalize so minimum row/col is 0
    if positions:
        min_r = min(r for r, c in positions.values())
        min_c = min(c for r, c in positions.values())
        positions = {
            d: (r - min_r, c - min_c) for d, (r, c) in positions.items()
        }

    return positions


def _get_preferred_mode(device_name: str, identifiers: list):
    """Find preferred resolution/frequency for a device from xrandr data."""
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


def _get_device_content_lines(
    name: str, props: dict, identifiers: list = None
) -> tuple:
    """Builds the text lines to display inside a monitor box.

    Returns (lines, used_preferred, is_primary).
    """
    is_primary = isinstance(props, dict) and props.get('primary', False)
    display_name = f'{name}\u00b9' if is_primary else name
    lines = [display_name]
    used_preferred = False

    if not isinstance(props, dict) or not props:
        # No config props — resolve preferred from xrandr if possible
        if identifiers:
            pref_res, pref_freq = _get_preferred_mode(name, identifiers)
            if pref_res and pref_freq:
                lines.append(f'{pref_res} @ {pref_freq}Hz*')
                used_preferred = True
                return lines, used_preferred, is_primary
        lines.append('(preferred)')
        return lines, used_preferred, is_primary

    disabled = props.get('disabled', False)
    if disabled:
        lines.append('(disabled)')
        return lines, used_preferred, is_primary

    res = props.get('resolution')
    freq = props.get('frequency')
    if res and freq:
        lines.append(f'{res} @ {freq}Hz')
    elif res:
        lines.append(res)
    elif freq:
        lines.append(f'@ {freq}Hz')
    else:
        # No resolution or frequency — resolve preferred
        if identifiers:
            pref_res, pref_freq = _get_preferred_mode(name, identifiers)
            if pref_res and pref_freq:
                lines.append(f'{pref_res} @ {pref_freq}Hz*')
                used_preferred = True
            else:
                lines.append('(preferred)')
        else:
            lines.append('(preferred)')

    if 'rotate' in props and props['rotate'] != 'normal':
        lines.append(f'rotate {props["rotate"]}')

    for pos_key in ('left-of', 'right-of', 'above', 'below'):
        if pos_key in props:
            lines.append(f'{pos_key} {props[pos_key]}')

    return lines, used_preferred, is_primary


def _render_monitor_box(
    content_lines: list, width: int, height: int, disabled: bool = False
) -> list:
    """Renders a single monitor as a list of strings of uniform width.

    Active monitors use solid box-drawing borders (─, │, ┌, ┐, └, ┘).
    Disabled monitors use dashed borders (╌, ┆, ┌, ┐, └, ┘).
    """
    if disabled:
        h_char, v_char = '╌', '┆'
    else:
        h_char, v_char = '─', '│'

    top_border = '┌' + h_char * (width - 2) + '┐'
    bot_border = '└' + h_char * (width - 2) + '┘'
    inner_width = width - 4  # v_char + space + content + space + v_char

    # Vertically center the content
    inner_height = height - 2  # minus top and bottom border
    pad_top = max(0, (inner_height - len(content_lines)) // 2)
    pad_bottom = max(0, inner_height - len(content_lines) - pad_top)

    rows = [top_border]
    for _ in range(pad_top):
        rows.append(f'{v_char}{" " * (width - 2)}{v_char}')
    for line in content_lines:
        centered = line.center(inner_width)
        rows.append(f'{v_char} {centered} {v_char}')
    for _ in range(pad_bottom):
        rows.append(f'{v_char}{" " * (width - 2)}{v_char}')
    rows.append(bot_border)

    return rows


def _render_config_layout(config: dict, identifiers: list = None) -> tuple:
    """Renders a full config section as monitor layout with box-drawing chars.

    Returns (lines, used_preferred) where used_preferred indicates
    whether any device used the * marker for display defaults.
    """
    # Filter out non-device keys
    devices = {k: v for k, v in config.items() if k not in ('hooks', 'comment')}
    if not devices:
        return [], False, False

    # Build grid positions
    grid = _build_monitor_grid(config)

    # Compute content lines and minimum box sizes for each device
    any_preferred = False
    any_primary = False
    device_info = {}
    for name, props in devices.items():
        content, used_pref, is_primary = _get_device_content_lines(
            name, props, identifiers=identifiers
        )
        if used_pref:
            any_preferred = True
        if is_primary:
            any_primary = True
        disabled = isinstance(props, dict) and props.get('disabled', False)
        # Minimum width: longest content line + 4 (borders + padding)
        min_width = max(len(line) for line in content) + 4
        # Parse resolution for proportional sizing
        pw, ph = 1920, 1080  # default
        if isinstance(props, dict) and 'resolution' in props:
            parts = props['resolution'].split('x')
            if len(parts) == 2:
                try:
                    pw, ph = int(parts[0]), int(parts[1])
                except ValueError:
                    pass
        # Swap dimensions for rotated (portrait) monitors
        if isinstance(props, dict) and props.get('rotate') in (
            'left',
            'right',
        ):
            pw, ph = ph, pw
        device_info[name] = {
            'content': content,
            'disabled': disabled,
            'px_w': pw,
            'px_h': ph,
            'min_width': min_width,
        }

    # Scale box widths proportionally
    # Target: widest monitor ~24 chars, minimum from content
    max_px_w = max(d['px_w'] for d in device_info.values())
    target_max_width = 24
    for name, info in device_info.items():
        proportional = int(target_max_width * info['px_w'] / max_px_w)
        scaled_w = max(proportional, info['min_width'])
        info['box_width'] = scaled_w

    # Scale box heights proportionally
    max_px_h = max(d['px_h'] for d in device_info.values())
    for name, info in device_info.items():
        content_height = len(info['content']) + 2  # +2 for borders
        # Proportional height: scale relative to tallest, min 6 lines total
        scaled_h = max(int(8 * info['px_h'] / max_px_h), content_height)
        info['box_height'] = scaled_h

    # Render individual boxes
    boxes = {}
    for name, info in device_info.items():
        boxes[name] = _render_monitor_box(
            content_lines=info['content'],
            width=info['box_width'],
            height=info['box_height'],
            disabled=info['disabled'],
        )

    # Determine grid dimensions
    max_row = max(r for r, c in grid.values())
    max_col = max(c for r, c in grid.values())

    # Calculate column widths and row heights from placed boxes
    col_widths = [0] * (max_col + 1)
    row_heights = [0] * (max_row + 1)
    for name, (r, c) in grid.items():
        if name in boxes:
            col_widths[c] = max(col_widths[c], len(boxes[name][0]))
            row_heights[r] = max(row_heights[r], len(boxes[name]))

    # Compose onto a 2D canvas
    gap = 2  # gap between columns
    total_width = sum(col_widths) + gap * max_col
    total_height = sum(row_heights)

    # Build canvas as list of lists of spaces
    canvas = [[' '] * total_width for _ in range(total_height)]

    for name, (r, c) in grid.items():
        if name not in boxes:
            continue
        box_lines = boxes[name]
        # Calculate x offset (column start)
        x_offset = sum(col_widths[:c]) + gap * c
        # Calculate y offset (row start), vertically center within row cell
        y_start = sum(row_heights[:r])
        y_pad = (row_heights[r] - len(box_lines)) // 2
        y_offset = y_start + y_pad

        for i, line in enumerate(box_lines):
            for j, ch in enumerate(line):
                if y_offset + i < total_height and x_offset + j < total_width:
                    canvas[y_offset + i][x_offset + j] = ch

    return (
        [''.join(row).rstrip() for row in canvas],
        any_preferred,
        any_primary,
    )


def show(
    main_dict: dict,
    config: dict,
    logger: logging.Logger,
):
    """Pretty-prints the current config as monitor layout with box-drawing"""
    # If no config has is_current, treat the first one as active
    has_current = any('is_current' in c for c in main_dict['active_config'])
    show_preferred = False
    show_primary = False
    identifiers = main_dict.get('identifiers', [])

    print()  # Blank line for separation

    for idx, conf in enumerate(main_dict['active_config'], 1):
        if has_current:
            current = 'is_current' in conf
        else:
            current = idx == 1
        converted_config = replace_aliases_with_real_names(
            main_dict=main_dict,
            config_to_convert={
                k: v for k, v in conf.items() if k != 'is_current'
            },
            logger=logger,
        )
        header = f'Config {idx}'
        if current:
            # print ansi to make "active" green
            header += ' (\u001b[32mactive\u001b[0m)'
        print(f'{header}:')

        lines, used_pref, used_pri = _render_config_layout(
            converted_config, identifiers=identifiers
        )
        if used_pref:
            show_preferred = True
        if used_pri:
            show_primary = True
        for line in lines:
            print(line)
        print()

    if 'global_failback' in config:
        print('Global failback:')
        failback_config = replace_aliases_with_real_names(
            main_dict=main_dict,
            config_to_convert=config['global_failback'],
            logger=logger,
        )
        lines, used_pref, used_pri = _render_config_layout(
            failback_config, identifiers=identifiers
        )
        if used_pref:
            show_preferred = True
        if used_pri:
            show_primary = True
        for line in lines:
            print(line)
        print()

    # Print footnotes if any markers were used
    if show_primary:
        print(' \u00b9 primary display')
    if show_preferred:
        print(' * preferred display defaults')


def fresh_start(
    args: argparse.Namespace,
    config: dict,
    logger: logging.Logger,
) -> dict:
    """Creates a fresh state file in case of a new config or new devices"""

    main_dict = build_main_dict(config=config)

    main_dict['active_config'] = deepcopy(
        get_active_config(
            main_dict=main_dict,
            config=config,
            logger=logger,
            dry_run=args.dry_run,
        )
    )
    main_dict = assign_aliases(main_dict=main_dict, logger=logger)

    # Quick sanity check, if there is no active config, we can't continue
    if len(main_dict['active_config']) == 0:
        logger.error('No active config can be determined with current rules!')
        apply_global_failback(
            main_dict=main_dict,
            config=config,
            logger=logger,
            dry_run=args.dry_run,
        )

    return main_dict


def main(save_path: str, args: argparse.Namespace):
    # First check if the script is run with the correct Python version
    enforce_python_version()

    # Get the logger
    logger = get_logger(verbose=args.verbose)

    # Read the config file, if changed, we will start from scratch
    config = read_config(config_file=args.config, logger=logger)

    # Validate the config file
    # This does schema validation and checks for logical loops in the config
    validate_config(config_dict=config, config_file=args.config, logger=logger)

    # We will sleep a bit to let hardware settle down
    # (e.g. when you plug-in a dock, it takes a bit to recognize multiple screens etc.)
    if args.command == 'rotate':
        time.sleep(1)

    # Construct the main dictionary, will use it for comparison
    new_main_dict = fresh_start(args=args, config=config, logger=logger)

    connected_count = len(
        [x for x in new_main_dict['identifiers'] if x['is_connected']]
    )
    active_count = len(
        [
            device
            for device in new_main_dict['identifiers']
            if device['is_active']
        ]
    )

    logger.info(
        f'Found {connected_count} connected screen'
        f'{"" if connected_count == 1 else "s"} '
        f'(active count: {active_count})'
    )

    try:
        # First check if reset flag is set
        if args.command == 'rotate' and args.reset:
            logger.info('Reset flag is set, ignoring previous config.')
            raise FileNotFoundError

        # Now load the previous state from disk
        previous_dict = load_from_disk(
            filename=path_join(save_path, 'loose.statefile'),
            logger=logger,
        )
        if args.dump_state:
            if not previous_dict:
                logger.error('No previous state found to dump!')
                exit(1)
            # Debugging request, print the state and exit
            _print_and_exit(previous_dict)

        # If no previous state found, we will start from scratch
        if not previous_dict:
            logger.info('No previous state found, starting from scratch.')
            raise FileNotFoundError
        # If loose itself is updated, we will start from scratch
        if (
            'VERSION' not in previous_dict
            or previous_dict['VERSION'] != VERSION
        ):
            logger.info('Config version mismatch. Ignoring the old config.')
            raise FileNotFoundError

        # Compare loaded xrandr output with the current one
        # If they don't have same device hash, we will start from scratch
        previously_connected = [
            f'{x["device_name"]}({x["product_id"]})'
            for x in previous_dict['identifiers']
            if x['is_connected']
        ]
        currently_connected = [
            f'{x["device_name"]}({x["product_id"]})'
            for x in new_main_dict['identifiers']
            if x['is_connected']
        ]
        if previously_connected != currently_connected:
            logger.info(
                'Config mismatch due to (dis)connected and/or (de)activated devices. Ignoring the old config.'
            )
            logger.debug(
                f'Previous: {previously_connected}, Current: {currently_connected}'
            )
            raise FileNotFoundError

        logger.debug('Devices match with previously saved identifiers')

        if 'raw_config' not in previous_dict or (
            previous_dict['raw_config'] != config
        ):
            logger.info(
                'Config changed since last save, ignoring the old config.'
            )
            raise FileNotFoundError

        if args.command == 'rotate' and args.ensure:
            logger.info(
                'Ensure flag is set & no changes detected, exiting peacefully'
            )
            exit(0)

        logger.debug('Config also match with previously saved data, using it')
        main_dict = previous_dict

    except FileNotFoundError:
        main_dict = new_main_dict

    if args.command == 'rotate':
        rotate(
            args=args,
            full_config=config,
            logger=logger,
            main_dict=main_dict,
            save_file=path_join(save_path, 'loose.statefile'),
            ignore_failing_hooks=args.ignore_failing_hooks,
        )
    elif args.command == 'show':
        show(
            main_dict=main_dict,
            config=config,
            logger=logger,
        )


def main_wrapper():
    """Just a dumb wrapper to satisfy the poetry entry point"""

    # Ensure our state folder exists
    save_path = path_join(Path(xdg_state_home(), 'loose'))
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Simple file lock to prevent multiple instances
    lock = FileLock(
        path_join(save_path, 'loose.lock'),
        timeout=0.1,
        poll_interval=0.05,
    )

    # Parse args early to check for interactive mode before locking
    args = get_parser(print_help=True if len(sys.argv) == 1 else False)

    try:
        with lock:
            main(save_path=save_path, args=args)
    except Timeout:
        if getattr(args, 'interactive', False):
            # Interactive mode: signal the preview process to advance
            pid_path = path_join(save_path, 'loose.preview_pid')
            try:
                with open(pid_path) as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGUSR1)
            except (FileNotFoundError, ValueError, ProcessLookupError):
                pass
            exit(0)
        else:
            logger = get_logger()
            logger.info('Another instance of loose is already running, skipping.')
            exit(0)
    except Exception as e:
        # Something else, dunno, log it and fail
        logger = get_logger()
        logger.error(f'An error occurred: {str(e)}')
        exit(1)


if __name__ == '__main__':
    main_wrapper()
