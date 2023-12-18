#!/usr/bin/env python3

import argparse
import jc
import logging
import pickle
import sys
import subprocess
import yamale
from shutil import which
from copy import deepcopy
from collections import defaultdict
from os import (
    get_terminal_size,
    environ,
)
from os.path import (
    dirname,
    abspath,
    join as path_join,
)
from pathlib import Path
from pprint import pprint
from typing import (
    Dict,
    List,
    Tuple,
)
from xdg_base_dirs import (
    xdg_state_home,
    xdg_config_home,
)
from yaml import safe_load, dump
from pyedid import (
    parse_edid,
    get_edid_from_xrandr_verbose,
)


CONFIG_FILE = f'{xdg_config_home()}/loose/config.yaml'
PY_MAJOR_VERSION = 3
PY_MINOR_VERSION = 10
RUN_TIMEOUT = 30  # In case of a stuck process, we will kill it after this many seconds
# Can't believe I don't have a portable way to do get the real version
# Poetry™ bullshit, has to be synced with pyproject.toml
VERSION = '0.0.11'


def get_identifiers(xrandr_output) -> List:
    """Returns the EDID product_id's of the connected devices"""

    edid_list = get_edid_from_xrandr_verbose(xrandr_output)
    return [parse_edid(device).product_id for device in edid_list]


def run_command(
        command: str,
        logger: logging.Logger,
) -> int:
    """Runs given command and returns the return code"""

    try:
        result = subprocess.run(
            command,
            timeout=RUN_TIMEOUT,
            env=environ,  # Pass the current environment
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
    except subprocess.TimeoutExpired:
        logger.critical(f'Command took more than {RUN_TIMEOUT} seconds, aborted: {command}')
        return 1

    logger.debug(f'Return code was: {result.returncode}')

    if result.returncode == 0:
        return 0

    logger.warning(f'Command failed: {command}')
    logger.info(f'Error output: {result.stderr.decode("utf-8")}')
    logger.info(f'Standard output: {result.stdout.decode("utf-8")}')
    return result.returncode


def get_parser(print_help: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='\n'.join([
            'Welcome to loose 🫠',
            '',
            'loose is a tool to manage your multi-monitor setup on Linux.',
            'It allows you to define your desired setup in a YAML file ',
            'in a flexible way and rotate between them.',
            '',
            'Feel free to use --help toggle for each subcommand below',
        ])
    )
    parser.add_argument(
        '-V',  # Capital to not conflict with --verbose
        '--version',
        action='store_true',
        help='Print version and exit'
    )
    sub = parser.add_subparsers(dest='command')
    common_options = argparse.ArgumentParser()
    common_options.add_argument(
        '-n',
        '--dry-run',
        action='store_true',
        help='Do not apply any hooks or xrandr commands, just print them'
    )
    common_options.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Print debug messages to stdout also'
    )
    rotate_parser = sub.add_parser(
        'rotate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Switch to the next definition on configuation',
        description='\n'.join([
            'Switch to the next definition on configuation.',
            '',
            'If there is no currently applied configuration,',
            'the first one will be applied.',
            '',
            'Example usage:',
            '  loose rotate',
        ]),
        parents=[common_options],
        add_help=False,
    )
    rotate_group = rotate_parser.add_mutually_exclusive_group()
    rotate_group.add_argument(
        '-r',
        '--reset',
        action='store_true',
        help='Do not check for the next config, apply the first one'
    )
    rotate_group.add_argument(
        '-e',
        '--ensure',
        action='store_true',
        help='Only apply changes if connected devices or config changed',
    )
    sub.add_parser(
        'show',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Print the active configuration',
        description='\n'.join([
            'Print the active configuration.',
            '',
            'Useful to check validated configuration and aliases,',
            'as well as next configuration(s) to be applied.',
            '',
            'Example usage:',
            '  loose show',
        ]),
        parents=[common_options],
        add_help=False,
    )
    sub.add_parser(
        'generate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Generate a sample configuration file',
        description='\n'.join([
            'Generate a sample configuration file.',
            '',
            'Useful to pipe it into config location and start as a template.',
            '',
            'Example usages:',
            '  loose generate',
            '  loose generate > ~/.config/loose/config.yaml',
        ]),
        parents=[common_options],
        add_help=False,
    )
    args = parser.parse_args()

    if print_help:
        parser.print_help()
        exit(0)

    return args


def enforce_python_version():
    # 3.7+ because of the ordered dicts
    # 3.10+ because of xdg-base-dirs
    if sys.version_info.major != PY_MAJOR_VERSION or sys.version_info.minor < PY_MINOR_VERSION:
        print("This script requires Python 3.7 or later.")
        sys.exit(1)


def save_to_disk(
    main_dict: Dict,
    save_path: str,
    logger: logging.Logger,
    current_config=None,
):
    # Save the dictionary to a file using pickle
    if current_config is not None:
        for config in main_dict['active_config']:
            if 'is_current' in config and config['is_current']:
                if config != current_config:
                    # Remove the current tag from the old config
                    del config['is_current']
            if config == current_config:
                config['is_current'] = True

    if current_config is None:
        logger.debug('Saving initial config to disk')
    else:
        logger.debug(f'Saving new active config to disk')

    with open(save_path, 'wb') as file:
        pickle.dump(main_dict, file)


def load_from_disk(filename):
    # Load the dictionary from the pickle file
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def parse_xrandr() -> Dict:
    """Parses the output of xrandr command and returns as dictionary"""

    outta = subprocess.check_output('xrandr', text=True)

    # It was horror trying to parse that ^bull(?:l+)?shit$ with regex myself
    # Kudos to jc: https://github.com/kellyjonbrazil/jc
    parsed_data = jc.parse('xrandr', outta)

    assert isinstance(parsed_data, dict)
    return parsed_data


def assert_unique_primary(data):
    """Asserts that there is only one primary screen for each config section"""
    for screen_count, config_list in data['on_screen_count'].items():
        for single_config in config_list:
            primary_count = sum(1 for _, config in single_config.items() if config.get('primary', False))
            if primary_count > 1:
                raise ValueError((
                    f'Multiple "primary" entries found within the same '
                    f'configuration list under on_screen_count->{screen_count} '
                    f'(index {config_list.index(single_config)})'
                ))


def validate_config(config: Dict, logger: logging.Logger):
    """Validates the config file"""

    loop_fail, message = has_loops(config['on_screen_count'])
    if loop_fail:
        logger.error(
            'Config file has loops, please remember we don\'t allow '
            '"below/above/left-of/right-of" definitions bi-directionally '
            'between screens (or self-references). Your detected issue was:\n\n'
            f'{message}'
        )
        exit(1)

    # Validate the data against the schema
    current_folder = dirname(abspath(__file__))
    schema_file = path_join(current_folder, 'config_schema.yaml')

    schema = yamale.make_schema(schema_file)
    data = yamale.make_data(CONFIG_FILE)
    try:
        yamale.validate(schema, data)
        assert_unique_primary(config)
        logger.debug('Validation of configuration successful.')
    except Exception as e:
        logger.error('Validation of configuration failed. Details:')
        print(str(e))
        exit(1)


def has_loops(on_screen_config) -> Tuple[bool, str]:
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
                        return True, f"There is a self-reference to screen object itself"
                    # Add a directed edge from current screen to referenced screen
                    graph[screen_id].append(ref_id)

        # Function for performing Depth-First Search
        def dfs(node_id, graph, visited, rec_stack):
            # If the node is in the recursion stack, a loop is detected
            if node_id in rec_stack:
                return True, "There are objects referring to each other within same config section"
            # If the node was already visited, skip it
            if node_id in visited:
                return False, ""
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
            return False, ""

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
        return False, ""

    # Iterate over each screen configuration index and the corresponding sections
    for _, screens_list in on_screen_config.items():
        for _, screen_section in enumerate(screens_list):
            # Ensure the section items are dictionaries before processing
            screens = {k: v for k, v in screen_section.items() if isinstance(v, dict)}
            # Use the helper function to check for loops in the current section
            has_loop, message = build_graph_and_check_loops(screens)
            if has_loop:
                # If a loop is detected, return the information immediately
                return True, message

    # Return False and an empty message if no loops are found in any configuration
    return False, ""


def _replace_none_with_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):  # If the item is a dict, recurse into it
            _replace_none_with_dict(v)
        elif v is None:  # Replace None with an empty dict
            d[k] = {}


def read_config() -> Dict:
    try:
        with open(f'{CONFIG_FILE}', 'r') as file_stream:
            config = safe_load(file_stream)
    except FileNotFoundError:
        print(f'Config file not found at: {CONFIG_FILE} !')
        exit(1)

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
    connected_devices: Dict,
    logger: logging.Logger,
) -> str:
    """Returns the real device name for the given alias"""

    for device, properties in connected_devices.items():
        if alias in properties['aliases']:
            return device

    logger.error(f'No real device found for alias "{alias}"')
    exit(1)


def replace_aliases_with_real_names(
    main_dict: Dict,
    config_to_convert: Dict,
    logger: logging.Logger,
) -> Dict:
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
        elif alias.startswith('_'):
            # This is an alias, replace it with the actual device name
            real_name = find_real_device_name(
                alias=alias,
                connected_devices=main_dict['connected_devices'],
                logger=logger
            )
        else:
            real_name = alias
        interim_config[real_name] = config

        # There might also be positioning directives, we have to replace them too
        for key, value in config.items():
            if key in ['left-of', 'right-of', 'above', 'below'] and value.startswith('_'):
                interim_config[real_name][key] = find_real_device_name(value, main_dict['connected_devices'], logger)
        replaced_config.update(interim_config)

    return replaced_config


def apply_xrandr_command(
    main_dict: Dict,
    config_to_apply: Dict,
    logger: logging.Logger,
    dry_run: bool,
) -> bool:
    """Applies the given config to the xrandr output

    Returns False only if the xrandr command itself fails
    we don't care that much about the pre/post hooks
    """

    replaced_config = replace_aliases_with_real_names(
        main_dict=main_dict,
        config_to_convert=config_to_apply,
        logger=logger,
    )

    xrandr_binary = which('xrandr')
    if not xrandr_binary:
        logger.error('xrandr command could not be found in PATH!')
        return False

    xrandr_command = [xrandr_binary]
    for device, config in replaced_config.items():
        xrandr_command += ['--output', device]
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

    # Turn off the disconnected screens
    for disconnected in [
        x['device_name'] for x in main_dict['screens'][0]['devices'] if not x['is_connected']
    ]:
        xrandr_command.extend(['--output', disconnected, '--off'])

    # Turn off connected but unused screens
    for connected in [
        x['device_name'] for x in main_dict['screens'][0]['devices'] if x['is_connected']
    ]:
        if connected not in replaced_config:
            xrandr_command.extend(['--output', connected, '--off'])

    if 'hooks' in replaced_config and 'pre' in replaced_config['hooks']:
        for hook in replaced_config['hooks']['pre']:
            if dry_run:
                logger.info(f'DRY RUN: Would run pre-hook: {hook}')
            else:
                logger.info(f'Running pre-hook: {hook}')
                pre_result = run_command(
                    command=hook,
                    logger=logger
                )
                if pre_result != 0:
                    logger.error(f'Pre-hook "{hook}" failed! Continuing anyway...')

    if dry_run:
        logger.info(f'DRY RUN: Would run command: {" ".join(xrandr_command)}')
        logger.debug(f'Config of command: {replaced_config}')
    else:
        logger.info(f'Running command: {" ".join(xrandr_command)}')
        logger.debug(f'Config of command: {replaced_config}')
        xrandr_result = run_command(
            command=' '.join(xrandr_command),
            logger=logger,
        )
        if xrandr_result != 0:
            logger.error('xrandr command failed!')
            return False

    if 'hooks' in replaced_config and 'post' in replaced_config['hooks']:
        for hook in replaced_config['hooks']['post']:
            if dry_run:
                logger.info(f'DRY RUN: Would run post-hook: {hook}')
            else:
                logger.info(f'Running post-hook: {hook}')
                post_result = run_command(
                    command=hook,
                    logger=logger
                )
                if post_result != 0:
                    logger.error(f'Post-hook "{hook}" failed! Continuing anyway...')

    return True


def get_logger(verbose: bool):
    """Creates and returns logger from logging lib"""

    logger = logging.getLogger('loose')
    formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler()
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def clear_impossible_configs(main_dict: Dict, logger: logging.Logger) -> Dict:
    """Removes the configs that are impossible to apply"""

    temp_dict_screens = main_dict['screens'].copy()
    temp_dict_active_config = main_dict['active_config'].copy()

    # Loop connected screens and find them in the active_config
    for screen in temp_dict_screens:
        for device in screen['devices']:
            # Remove the devices that are not connected
            if not device['is_connected']:
                for config in temp_dict_active_config:
                    for device_name in config:
                        if device_name == device['device_name']:
                            # Maybe just removed because of another disconnected device
                            if config in main_dict['active_config']:
                                logger.debug(f'Ignoring config "{config}" since device "{device_name}" is not connected')
                                main_dict['active_config'].remove(config)

    for config in main_dict['active_config']:
        for device in config:
            if device == 'hooks':
                # This is a special key, ignore it
                continue
            if not device.startswith('_') and device not in main_dict['connected_devices']:
                # Maybe just removed because of another nonexistent device
                if config in main_dict['active_config']:
                    logger.debug(f'Ignoring config "{config}" because of missing device "{device}"')
                    main_dict['active_config'].remove(config)

    return main_dict


def assign_aliases(main_dict: Dict, logger: logging.Logger) -> Dict:
    """Assigns given aliases to the screens in the xrandr output"""

    # First get all connected device names, we will use them as a reference
    # while comparing old and new configs. Also we'll assign tokens to them.
    connected_devices = {}
    for screen in main_dict['screens']:
        for device in screen['devices']:
            if device['is_connected']:
                connected_devices[device['device_name']] = {
                    'modes': device['modes'],
                    'aliases': [],
                }

    main_dict['connected_devices'] = connected_devices

    main_dict = clear_impossible_configs(main_dict=main_dict, logger=logger)

    # # Get all remaining aliases (fun fact, set() was not reliable here)
    unassigned_aliases = []
    for item in main_dict['active_config']:
        for key in item.keys():
            if key not in unassigned_aliases and key != 'hooks':
                unassigned_aliases.append(key)

    for device, properties in main_dict['connected_devices'].items():
        if properties['aliases'] == []:
            for section in main_dict['active_config']:
                if device not in section:
                    # This alias is not same as connected device name, skip
                    continue
                # Check if the real device can supply these defined resolutions
                needed_x, needed_y = None, None
                # If there is no resolution defined in config, that means we can use any resolution, nice
                if 'resolution' in section[device]:
                    needed_x, needed_y = (int(x) for x in section[device]['resolution'].split('x'))
                # Also check for supported frequencies
                needed_frequency = None
                if 'frequency' in section[device]:
                    needed_frequency = section[device]['frequency']

                # Now validate if there is a required resolution and/or frequency
                if needed_x and not any(
                    mode['resolution_width'] == needed_x and mode['resolution_height'] == needed_y for mode in properties['modes']
                ):
                    logger.debug(f'Config "{section}" is not applicable to device "{device}" due to resolution mismatch')
                    main_dict['active_config'].remove(section)
                    continue
                if needed_frequency and not any(
                    frequency['frequency'] == needed_frequency for mode in properties['modes'] for frequency in mode['frequencies']
                ):
                    logger.debug(f'Config "{section}" is not applicable to device "{device}" due to frequency mismatch')
                    main_dict['active_config'].remove(section)
                    continue

                # Re-check if any applicable configs left
                if any(device in d for d in main_dict['active_config']):
                    logger.debug(f'Assigning device definition "{device}" to device "{device}"')
                    connected_devices[device]['aliases'].append(device)
                    unassigned_aliases.remove(device)

    # Now handling actual aliases
    # Warning: Crappy hack time!
    # We will basically run the same loop twice, but on the first loop we will only handle devices without any aliases
    # so they will be prioritized
    # Another hack here, because python doesn't like modifying the list while looping on it, we will loop on a copy
    unassigned_aliases_copy = unassigned_aliases.copy()
    for _ in range(2):
        if _ == 0:
            logger.debug('Checking devices without any aliases assigned')
        else:
            logger.debug('Checking to spread the remaining aliases')
        for alias in unassigned_aliases_copy:
            if alias not in unassigned_aliases:
                # This alias is already assigned, skip
                continue
            for device, properties in main_dict['connected_devices'].items():
                if _ == 0 and properties['aliases'] != []:
                    # On first loop we will only handle devices without any aliases
                    continue
                # Test aliases by order to see if the resolutions are applicable
                logger.debug(f'Checking compatibility of "{alias}" for device "{device}"')

                mismatch = False
                for section in main_dict['active_config']:
                    if alias not in section:
                        # This alias is not related with this config section, skip
                        continue
                    # Check if the real device can supply these defined resolutions
                    needed_x, needed_y = None, None
                    # If there is no resolution defined in config, that means we can use any resolution, nice
                    if 'resolution' in section[alias]:
                        needed_x, needed_y = (int(x) for x in section[alias]['resolution'].split('x'))
                    # Also check for supported frequencies
                    needed_frequency = None
                    if 'frequency' in section[alias]:
                        needed_frequency = section[alias]['frequency']

                    # Now validate if there is a required resolution and/or frequency
                    if needed_x and not any(
                        mode['resolution_width'] == needed_x and mode['resolution_height'] == needed_y for mode in properties['modes']
                    ):
                        logger.debug(f'Config "{section}" is not applicable to device "{device}" due to resolution mismatch')
                        mismatch = True
                        continue
                    if needed_frequency and not any(
                        frequency['frequency'] == needed_frequency for mode in properties['modes'] for frequency in mode['frequencies']
                    ):
                        logger.debug(f'Config "{section}" is not applicable to device "{device}" due to frequency mismatch')
                        mismatch = True
                        continue

                if mismatch:
                    continue

                # This alias is applicable, at least one of the configs can be applied
                logger.debug(f'Assigning alias "{alias}" to device "{device}"')
                connected_devices[device]['aliases'].append(alias)
                unassigned_aliases.remove(alias)
                break

    for device, properties in main_dict['connected_devices'].items():
        logger.info(f'Determined aliases for device "{device}": {", ".join(properties["aliases"])}')

    return main_dict


def sanitize_config(
    main_dict: Dict,
    config_to_convert: Dict,
    logger: logging.Logger,
) -> Dict:

    # First replace the aliases with the real device names
    sanitized_reference_config = replace_aliases_with_real_names(
        main_dict=main_dict,
        config_to_convert=config_to_convert,
        logger=logger,
    )

    # Then add the implicit values if they are not defined
    for device, config in sanitized_reference_config.items():
        if 'rotate' not in config:
            sanitized_reference_config[device]['rotate'] = 'normal'
        if 'primary' not in config:
            sanitized_reference_config[device]['primary'] = False

    if 'hooks' in sanitized_reference_config:
        # We don't care about hooks here, remove them
        del sanitized_reference_config['hooks']

    return sanitized_reference_config


def _compare_with_empty_values(
    sanitized_current_state: Dict,
    sanitized_reference_config: Dict,
) -> bool:
    # What matters is the reference config, if there are more items in current state, it's fine
    for device, config in sanitized_reference_config.items():
        if device not in sanitized_current_state:
            return False
        for key, value in config.items():
            if key not in sanitized_current_state[device]:
                return False
            if sanitized_current_state[device][key] != value:
                return False
    return True


def get_next_config(
    active_config: List,
    reset: bool,
    logger: logging.Logger,
) -> Dict:
    """Get the xrandr output, return the next config in the list"""
    # Check if there is a currently applied config
    # If there is, rotate to the next one
    # If there isn't, apply the first one

    for config in active_config:
        if reset:
            logger.debug('Reset requested, applying the first config')
            return config
        if 'is_current' in config:
            next_config = active_config[(active_config.index(config) + 1) % len(active_config)]
            logger.debug(f'Rotating to the next config: {next_config}')
            return next_config

    logger.debug(f'No active configuration found, applying the first config: {active_config[0]}')

    return active_config[0]


def _print_and_exit(anyobject):
    pprint(anyobject, width=1)
    exit(0)


def get_active_config(
    main_dict: Dict,
    config: Dict,
    connected_count: int,
    logger: logging.Logger,
    dry_run: bool,
) -> Tuple[int, Dict]:
    """Returns the active config for the current screen count"""

    if connected_count not in config['on_screen_count']:
        logger.warning(
            f'No config found for {connected_count} screens! '
            'Applying global failback directive.'
        )
        if 'global_failback' not in config:
            logger.error('Can\'t even find global_failback directive in the config, exiting!')
            exit(1)
        apply_xrandr_command(
            main_dict=main_dict,
            config_to_apply=config['global_failback'],
            logger=logger,
            dry_run=dry_run,
        )

    return config['on_screen_count'][connected_count]


def main():
    enforce_python_version()
    args = get_parser(print_help=True if len(sys.argv) == 1 else False)

    if args.version:
        print(f'loose 🫠 version: {VERSION}')
        exit(0)

    # Let's handle this first, so we can exit early
    if args.command == 'generate':
        # Example config is in the same folder as the script
        current_folder = dirname(abspath(__file__))
        schema_file = path_join(current_folder, 'example_config.yaml')
        with open(schema_file, 'r') as file_stream:
            print(file_stream.read())
        exit(0)

    config = read_config()

    # Ensure our state folder exists
    save_path = path_join(Path(xdg_state_home(), 'loose'))
    Path(save_path).mkdir(parents=True, exist_ok=True)

    save_file = path_join(save_path, 'loose.statefile')
    logger = get_logger(verbose=args.verbose)

    validate_config(config=config, logger=logger)

    # First check if we have a saved state and they match with connected devices
    # We rely on product_id's from EDID, they are supposed to be unique
    randr = subprocess.check_output(['xrandr', '--verbose'])
    connected_products = get_identifiers(randr)

    logger.info(f'Found {len(connected_products)} connected screen{"" if len(connected_products) == 1 else "s"}')

    try:
        previous_dict = load_from_disk(save_file)
        # If loose itself is updated, we will start from scratch
        if 'VERSION' not in previous_dict or previous_dict['VERSION'] != VERSION:
            logger.info('Config version mismatch. Scraping the old config.')
            raise FileNotFoundError
        # Compare loaded xrandr output with the current one
        # If they don't have same device hash, we will start from scratch
        elif sorted(previous_dict['connected_products']) == sorted(connected_products):
            logger.debug('Devices match with previously saved data')
            if 'raw_config' in previous_dict and previous_dict['raw_config'] == config:
                if args.command == 'rotate' and args.ensure:
                    logger.info('Ensure flag is set & no changes detected, exiting peacefully')
                    exit(0)
                logger.debug('Config also match with previously saved data, using it')
                main_dict = previous_dict
            else:
                logger.info('Config changed since last save, scraping the old config.')
                raise FileNotFoundError
        else:
            logger.info('Devices mismatch due to connected/disconnected devices. Scraping the old config.')
            raise FileNotFoundError
    except FileNotFoundError:
        main_dict = None

    if main_dict is None:
        main_dict = parse_xrandr()
        main_dict['raw_config'] = deepcopy(config)

        main_dict['active_config'] = get_active_config(
            main_dict=main_dict,
            config=config,
            connected_count=len(connected_products),
            logger=logger,
            dry_run=args.dry_run,
        )
        main_dict = assign_aliases(main_dict=main_dict, logger=logger)

        main_dict['VERSION'] = VERSION
        main_dict['connected_products'] = connected_products

        # And at last, save the state to disk
        save_to_disk(
            main_dict=main_dict,
            save_path=save_file,
            logger=logger,
        )

    if args.command == 'rotate':
        logger.debug('Got request to rotate.')
        next_config = get_next_config(
            active_config=main_dict['active_config'],
            reset=args.reset,
            logger=logger,
        )
        run_result = apply_xrandr_command(
            main_dict=main_dict,
            config_to_apply=next_config,
            logger=logger,
            dry_run=args.dry_run,
        )
        if run_result:
            # Save the state to disk with new current tag
            save_to_disk(
                main_dict=main_dict,
                save_path=save_file,
                logger=logger,
                current_config=next_config
            )
        else:
            logger.error('Failed to apply the config, exiting!')
            exit(1)
    elif args.command == 'show':
        print('Currently active config:')
        print()
        print('-' * round(get_terminal_size().columns/3))

        for conf in main_dict['active_config']:
            current = False
            if 'is_current' in conf:
                current = True
                del conf['is_current']
            converted_config = replace_aliases_with_real_names(
                main_dict=main_dict,
                config_to_convert=conf,
                logger=logger,
            )
            if current:
                print('👉 ', end='')
            else:
                print('  ', end='')
            print(dump(
                converted_config,
                default_flow_style=False,
                indent=7,
            ))
            print('-' * round(get_terminal_size().columns/3))

        if 'global_failback' in config:
            print('-' * round(get_terminal_size().columns/3))
            print('Global failback directive:')
            print()
            print(dump(
                replace_aliases_with_real_names(
                    main_dict=main_dict,
                    config_to_convert=config['global_failback'],
                    logger=logger,
                ),
                default_flow_style=False,
                indent=7,
            ))
            print('-' * round(get_terminal_size().columns/3))

if __name__ == '__main__':
    main()
