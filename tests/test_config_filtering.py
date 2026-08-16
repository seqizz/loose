import pytest

from loose.backends import XrandrBackend
from loose.loose import (
    clear_impossible_configs,
    get_active_config,
    get_next_config,
)

# --- get_next_config ---


class TestGetNextConfig:
    def test_no_current_returns_first(self, logger):
        configs = [{'_1': {}}, {'_2': {}}]
        result = get_next_config(configs, logger)
        assert result == {'_1': {}}

    def test_current_last_wraps_to_first(self, logger):
        configs = [{'_1': {}}, {'_2': {}, 'is_current': True}]
        result = get_next_config(configs, logger)
        assert result == {'_1': {}}

    def test_current_first_returns_second(self, logger):
        configs = [{'_1': {}, 'is_current': True}, {'_2': {}}]
        result = get_next_config(configs, logger)
        assert result == {'_2': {}}

    def test_current_middle_returns_next(self, logger):
        configs = [{'a': {}}, {'b': {}, 'is_current': True}, {'c': {}}]
        result = get_next_config(configs, logger)
        assert result == {'c': {}}

    def test_single_entry_wraps_to_itself(self, logger):
        configs = [{'_1': {}, 'is_current': True}]
        result = get_next_config(configs, logger)
        assert result == {'_1': {}, 'is_current': True}


# --- clear_impossible_configs ---


class TestClearImpossibleConfigs:
    def test_removes_disconnected_device_config(
        self, make_main_dict, device_edp1, device_dp2_disconnected, logger
    ):
        active_config = [{'DP-2': {'resolution': '1920x1080'}}]
        md = make_main_dict(
            [device_edp1, device_dp2_disconnected], active_config
        )
        result = clear_impossible_configs(md, logger)
        assert len(result['active_config']) == 0

    def test_keeps_valid_config(self, make_main_dict, device_edp1, logger):
        active_config = [{'eDP-1': {'resolution': '1920x1080'}}]
        md = make_main_dict([device_edp1], active_config)
        result = clear_impossible_configs(md, logger)
        assert len(result['active_config']) == 1

    def test_keeps_alias_configs(self, make_main_dict, device_edp1, logger):
        """Aliases (starting with _) should not be filtered by this function."""
        active_config = [{'_1': {'resolution': '1920x1080'}}]
        md = make_main_dict([device_edp1], active_config)
        result = clear_impossible_configs(md, logger)
        assert len(result['active_config']) == 1

    def test_keeps_hooks_key(self, make_main_dict, device_edp1, logger):
        """'hooks' key should not be treated as a device name."""
        active_config = [{'eDP-1': {}, 'hooks': {'pre': ['echo hi']}}]
        md = make_main_dict([device_edp1], active_config)
        result = clear_impossible_configs(md, logger)
        assert len(result['active_config']) == 1

    def test_removes_nonexistent_device_config(
        self, make_main_dict, device_edp1, logger
    ):
        """Config referencing a device name that doesn't exist at all."""
        active_config = [{'FAKE-1': {'resolution': '1920x1080'}}]
        md = make_main_dict([device_edp1], active_config)
        result = clear_impossible_configs(md, logger)
        assert len(result['active_config']) == 0

    def test_mixed_valid_and_invalid(
        self, make_main_dict, device_edp1, device_dp2_disconnected, logger
    ):
        """One valid config and one referencing a disconnected device."""
        active_config = [
            {'eDP-1': {'resolution': '1920x1080'}},
            {'DP-2': {'resolution': '1920x1080'}},
        ]
        md = make_main_dict(
            [device_edp1, device_dp2_disconnected], active_config
        )
        result = clear_impossible_configs(md, logger)
        assert len(result['active_config']) == 1
        assert 'eDP-1' in result['active_config'][0]


# --- get_active_config ---


class TestGetActiveConfig:
    def test_matching_screen_count(self, make_main_dict, device_edp1, logger):
        config = {
            'on_screen_count': {1: [{'_1': {}}]},
            'global_failback': {'_1': {}},
        }
        md = make_main_dict([device_edp1], [])
        result = get_active_config(
            md, config, logger, dry_run=True, backend=XrandrBackend()
        )
        assert result == [{'_1': {}}]

    def test_no_matching_count_triggers_failback(
        self, make_main_dict, device_edp1, logger, mocker
    ):
        """When no matching screen count, apply_global_failback is called.

        The real apply_global_failback calls exit(), so execution never
        reaches the return statement. We simulate that with side_effect.
        """
        config = {
            'on_screen_count': {3: [{'_1': {}}]},
            'global_failback': {'_1': {}},
        }
        md = make_main_dict([device_edp1], [])
        mock_failback = mocker.patch(
            'loose.loose.apply_global_failback', side_effect=SystemExit(1)
        )
        with pytest.raises(SystemExit):
            get_active_config(
                md, config, logger, dry_run=True, backend=XrandrBackend()
            )
        mock_failback.assert_called_once()
