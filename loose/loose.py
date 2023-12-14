#!/usr/bin/env python3

import argparse
import jc
import logging
import pickle
import sys
from collections import defaultdict
from os import get_terminal_size
from os.path import (
    dirname,
    abspath,
    join as path_join,
)
from pathlib import Path
from pprint import pprint
from pykwalify.core import Core
from subprocess import Popen, check_output
from typing import Dict, List, Tuple
from xdg_base_dirs import (
    xdg_state_home,
    xdg_config_home,
)
from yaml import safe_load, dump


CONFIG_FILE = f'{xdg_config_home()}/loose/config.yaml'
PY_MAJOR_VERSION = 3
PY_MINOR_VERSION = 10
# Can't believe I don't have a portable way to do get the real version
# Poetry™ bullshit, has to be synced with pyproject.toml
VERSION = '0.0.5'
CONFIG_VERSION = f'{VERSION}.4'


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
        help='Do not apply the configuration, just print the commands'
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
            'If there is no currently applied configuration,',
            'the first one will be applied.',
            '',
            'Example usage:',
            '  loose rotate',
        ]),
        parents=[common_options],
        add_help=False,
    )
    rotate_parser.add_argument(
        '-r',
        '--reset',
        action='store_true',
        help='Do not check for the next config, apply the first one'
    )
    sub.add_parser(
        'show',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Print the current configuration and exit',
        description='\n'.join([
            'Print the active configuration and exit.',
            'Useful to check validated configuration and aliases,',
            'as well as next configuration(s) to be applied.',
            '',
            'Example usage:',
            '  loose show',
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


def save_to_disk(data, filename):
    # Save the dictionary to a file using pickle
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_from_disk(filename):
    # Load the dictionary from the pickle file
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def parse_xrandr() -> Dict:
    """Parses the output of xrandr command and returns as dictionary"""

    outta = check_output('xrandr', text=True)

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

    # _print_and_exit(config['on_screen_count'])

    if has_loops(config['on_screen_count']):
        logger.error(
            'Config file has loops, please do not refer "below/above/left-of/right-of" '
            'directions bi-directionally between screens (or refer to itself).'
        )
        exit(1)

    current_folder = dirname(abspath(__file__))
    schema_file = path_join(current_folder, 'yamale.yaml')

    # Validate the data against the schema
    schema_file = path_join(current_folder, 'config_schema.yaml')

    core = Core(source_data=config, schema_files=[schema_file])

    # Prevent pykwalify from printing to stdout itself
    logging.getLogger('pykwalify.core').handlers = [logging.NullHandler()]

    try:
        core.validate(raise_exception=True)
        assert_unique_primary(config)
        logger.debug('Validation of configuration successful.')
    except Exception as e:
        logger.error('Validation of configuration failed. Details:')
        pprint(e.args[0], width=get_terminal_size().columns)
        exit(1)


def has_loops(on_screen_config) -> bool:
    """Detects whether there is a loop in the graph, for position references

    We don't allow:
        - Self-reference (e.g. _X: {below: _X})
        - Level-1 reference loops for "any" directives (e.g. _X: {below: _Y}, _Y: {above: _X})

    Credit: ChatGPT4 (No way in hell I can write this myself)
    """

    graph = defaultdict(dict)
    # Build the graph with details about each directional relationship
    for screens in on_screen_config.values():
        for screen_dict in screens:
            for screen_id, properties in screen_dict.items():
                for direction, ref_id in properties.items():
                    if direction in ['above', 'below', 'left-of', 'right-of']:
                        if ref_id == screen_id:  # Rule: No screen can refer to itself
                            return True
                        if ref_id not in graph[screen_id].values():
                            graph[screen_id][direction] = ref_id
                        else:
                            # Rule: If referred screen ref_ids a different direction back to the screen_id
                            return True
                        # Check the reverse direction for a bidirectional link
                        opposite_dir = {
                            'above': 'below',
                            'below': 'above',
                            'right-of': 'left-of',
                            'left-of': 'right-of'
                        }[direction]
                        # Rule: No bidirectional direct references allowed
                        if graph.get(ref_id, {}).get(opposite_dir) == screen_id:
                            return True
    # Perform DFS to detect whether there is a loop in the graph
    def dfs(node_id, visited, rec_stack):
        # If the node_id is in the recursion stack, then we have found a loop
        if node_id in rec_stack:
            return True
        # If the node_id is visited and not in the recursion stack, then no loop is found in this path
        if node_id in visited:
            return False
        visited.add(node_id)
        rec_stack.add(node_id)
        # Perform DFS for adjacent nodes
        for neighbor_id in graph[node_id].values():
            if dfs(neighbor_id, visited, rec_stack):
                return True

        # Remove node_id from the recursion stack before backtracking
        rec_stack.remove(node_id)
        return False

    visited, rec_stack = set(), set()
    nodes = list(graph.keys())  # Create a static list of nodes to prevent RuntimeError during iteration
    for node_id in nodes:
        if dfs(node_id, visited, rec_stack):
            return True  # Loop detected

    return False  # No loops detected in the graph


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
    for device, config in config_to_convert.items():
        if device == 'is_current':
            # This is a hacky special key, ignore it
            continue

        interim_config = {}
        if device.startswith('_'):
            # This is a token, replace it with the actual device name
            real_name = find_real_device_name(device, main_dict['connected_devices'], logger)
        else:
            real_name = device
        interim_config[real_name] = config

        # There might also be positioning directives, we have to replace them too
        for key, value in config.items():
            if key in ['left-of', 'right-of', 'above', 'below']:
                interim_config[real_name][key] = find_real_device_name(value, main_dict['connected_devices'], logger)
        replaced_config.update(interim_config)

    return replaced_config


def apply_xrandr_command(
    main_dict: Dict,
    config_to_apply: Dict,
    logger: logging.Logger,
    dry_run: bool,
) -> bool:
    """Applies the given config to the xrandr output"""

    replaced_config = replace_aliases_with_real_names(
        main_dict=main_dict,
        config_to_convert=config_to_apply,
        logger=logger,
    )

    xrandr_command = ['xrandr']
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

    if dry_run:
        logger.info(
            f'DRY RUN: Would run command: {" ".join(xrandr_command)} '
            f'for config {replaced_config}'
        )
        return True

    logger.debug(f'Running command: {" ".join(xrandr_command)} for config {replaced_config}')

    command = Popen(xrandr_command)
    command.communicate()

    if command.returncode == 0:
        return True

    return False


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

    # # Get all remaining aliases
    unassigned_aliases = set()
    for item in main_dict['active_config']:
        unassigned_aliases.update(item.keys())
    unassigned_aliases = list(unassigned_aliases)

    # First check if we have any device specific aliases
    for device, properties in main_dict['connected_devices'].items():
        if properties['aliases'] == []:
            if any(device in d for d in main_dict['active_config']):
                # Means this is aliased with its own name
                for related_conf in [
                    d for d in main_dict['active_config'] if device in d and d[device] and 'resolution' in d[device]
                ]:
                    # Check if the real device can supply these defined resolutions
                    needed_x, needed_y = (int(x) for x in related_conf[device]['resolution'].split('x'))
                    if not any(
                        mode['resolution_width'] == needed_x and mode['resolution_height'] == needed_y for mode in properties['modes']
                    ):
                        logger.debug(f'Config "{related_conf}" is not applicable to device "{device}" due to resolution mismatch')
                        main_dict['active_config'].remove(related_conf)
                        continue
                # Re-check if any applicable configs left
                if any(device in d for d in main_dict['active_config']):
                    logger.debug(f'Assigning alias "{device}" to device "{device}"')
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
        for alias in unassigned_aliases:
            for device, properties in main_dict['connected_devices'].items():
                if _ == 0 and properties['aliases'] != []:
                    # On first loop we will only handle devices without any aliases
                    continue
                # Test aliases by order to see if the resolutions are applicable
                logger.debug(f'Checking compatibility of "{alias}" for device "{device}"')
                # First check if there is any config defined without any special resolution
                for related_conf in [
                    d for d in main_dict['active_config'] if alias in d and d[alias] and 'resolution' not in d[alias]
                ]:
                    # Yay! We found at least one, just snap the alias to it and don't think about it
                    logger.debug(f'Assigning alias "{alias}" to device "{device}"')
                    connected_devices[device]['aliases'].append(alias)
                    unassigned_aliases_copy.remove(alias)
                    break
                if alias not in unassigned_aliases_copy:
                    # We found a config for this alias, skip on the upper loop too
                    break

                # At here, we couldn't find any config without a resolution, so we have to check the resolutions
                mismatch = False
                related_conf_list = [
                    d for d in main_dict['active_config'] if alias in d and d[alias] and 'resolution' in d[alias]
                ]
                for related_conf in related_conf_list:
                    needed_x, needed_y = (int(x) for x in related_conf[alias]['resolution'].split('x'))
                    if not any(
                        mode['resolution_width'] == needed_x and mode['resolution_height'] == needed_y for mode in properties['modes']
                    ):
                        logger.debug(
                            f'"{device}" can\'t supply resolution "{needed_x}x{needed_y}", '
                            f'so will not be assigned to alias "{alias}".'
                        )
                        mismatch = True

                if mismatch:
                    continue

                # This alias is applicable, at least one of the configs can be applied
                logger.debug(f'Assigning alias "{alias}" to device "{device}"')
                connected_devices[device]['aliases'].append(alias)
                unassigned_aliases_copy.remove(alias)

    for device, properties in main_dict['connected_devices'].items():
        logger.info(f'Determined aliases for device "{device}": {", ".join(properties["aliases"])}')

    return main_dict


def get_current_state(main_dict: Dict) -> Dict:

    current_status = {}
    for device in main_dict['screens'][0]['devices']:
        if 'modes' not in device:
            # invalid, pass
            continue
        for mode in device['modes']:
            if 'frequencies' not in mode:
                # invalid, pass
                continue
            for frequency in mode['frequencies']:
                if frequency['is_current']:
                    current_status[device['device_name']] = {
                        'frequency': round(float(frequency['frequency'])),
                        'resolution': f'{mode["resolution_width"]}x{mode["resolution_height"]}',
                        'primary': device['is_primary'],
                        'rotate': device['rotation'],
                    }

    return current_status


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

    return sanitized_reference_config


def compare_states(
    current_state: Dict,
    main_dict: Dict,
    reference_config: Dict,
    logger: logging.Logger,
) -> bool:

    sanitized_current_state = {}
    for device, details in current_state.items():
        sanitized_current_state[device] = {
            'frequency': details['frequency'],
            'resolution': details['resolution'],
        }
        if 'position' in details:
            sanitized_current_state[device]['position'] = details['position']
        if 'primary' in details:
            sanitized_current_state[device]['primary'] = details['primary']
        if 'rotate' in details:
            sanitized_current_state[device]['rotate'] = details['rotate']

    sanitized_reference_config = sanitize_config(
        main_dict=main_dict,
        config_to_convert=reference_config,
        logger=logger,
    )

    return sanitized_current_state == sanitized_reference_config


def get_next_config(active_config: List, logger: logging.Logger) -> Dict:
    """Get the xrandr output, return the next config in the list"""
    # Check if there is a currently applied config
    # If there is, rotate to the next one
    # If there isn't, apply the first one

    index = 0
    found = False
    for config in active_config:
        if 'is_current' in config:
            found = True
            index = active_config.index(config) + 1
            break

    if not found:
        logger.debug(f'No active configuration found, applying the first config: {active_config[0]}')

    next_config = active_config[index % len(active_config)]

    logger.debug(f'Rotating to the next config: {next_config}')

    return next_config


def _print_and_exit(anyobject):
    pprint(anyobject, width=1)
    exit(0)


def get_active_config(
    main_dict: Dict,
    config: Dict,
    logger: logging.Logger,
    dry_run: bool,
) -> Tuple[int, Dict]:
    """Returns the active config for the current screen count"""

    # Multi-screen support is not implemented yet
    connected_count = [x['is_connected'] for x in main_dict['screens'][0]['devices']].count(True)

    logger.info(f'Found {connected_count} connected screens.')

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

    return connected_count, config['on_screen_count'][connected_count]


def main():
    enforce_python_version()
    args = get_parser(print_help=True if len(sys.argv) == 1 else False)

    action = sys.argv[1]

    if args.version:
        print(f'loose 🫠 version: {VERSION}')
        exit(0)

    config = read_config()

    # Ensure our state folder exists
    save_path = path_join(Path(xdg_state_home(), 'loose'))
    Path(save_path).mkdir(parents=True, exist_ok=True)

    save_file = path_join(save_path, 'loose.statefile')
    logger = get_logger(verbose=args.verbose)

    validate_config(config=config, logger=logger)

    main_dict = parse_xrandr()

    connected_screen_count, main_dict['active_config'] = get_active_config(
        main_dict=main_dict,
        config=config,
        logger=logger,
        dry_run=args.dry_run,
    )

    main_dict = assign_aliases(main_dict=main_dict, logger=logger)

    current_state = get_current_state(main_dict=main_dict)

    # Label the current config inside active_config
    for conf in main_dict['active_config']:
        if compare_states(
            current_state=current_state,
            reference_config=conf,
            main_dict=main_dict,
            logger=logger,
        ):
            conf['is_current'] = True

    main_dict['CONFIG_VERSION'] = CONFIG_VERSION

    # Check if save file exists
    try:
        previous_dict = load_from_disk(save_file)
        # If loose itself is updated, we will start from scratch
        if previous_dict['CONFIG_VERSION'] != CONFIG_VERSION:
            logger.debug('Config version mismatch. Scraping the old config.')
            raise FileNotFoundError
        # Compare loaded xrandr output with the current one
        # If they don't have same device hash, we will start from scratch
        elif previous_dict['connected_devices'].keys() == main_dict['connected_devices'].keys():
            logger.debug('Devices match with previously saved config.')
        else:
            logger.debug('Devices mismatch due to either connected or disconnected devices. Scraping the old config.')
            raise FileNotFoundError
    except FileNotFoundError:
        # Save the xrandr output to disk before continuing
        previous_dict = None
        save_to_disk(main_dict, save_file)

    if main_dict == previous_dict:
        logger.debug('No config/connectivity changes detected since last run.')

    if action == 'rotate':
        logger.debug('Got request to rotate.')
        if args.reset:
            logger.debug('Reset requested, applying the first config')
            next_config = main_dict['active_config'][0]
        else:
            next_config = get_next_config(active_config=main_dict['active_config'], logger=logger)
        apply_xrandr_command(
            main_dict=main_dict,
            config_to_apply=next_config,
            logger=logger,
            dry_run=args.dry_run,
        )
    elif action == 'show':
        print(
            f'Currently validated config for {connected_screen_count} '
            f'screen{"" if connected_screen_count == 1 else "s"}:'
        )
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
