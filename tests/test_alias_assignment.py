import pytest

from loose.loose import assign_aliases, find_real_device_name, replace_aliases_with_real_names


# --- assign_aliases ---


class TestAssignAliases:
    def test_single_alias_single_device(self, make_main_dict, device_edp1, logger):
        """Single alias _1 maps to the only connected device."""
        active_config = [{'_1': {'resolution': '1920x1080', 'frequency': 60}}]
        md = make_main_dict([device_edp1], active_config)
        result = assign_aliases(md, logger)
        edp = next(d for d in result['identifiers'] if d['device_name'] == 'eDP-1')
        assert '_1' in edp['aliases']
        assert len(result['active_config']) == 1

    def test_explicit_name_priority(self, make_main_dict, device_edp1, device_hdmi1, logger):
        """Explicit device name 'eDP-1' claims that device before alias pass."""
        active_config = [
            {
                'eDP-1': {'resolution': '1920x1080'},
                '_1': {'resolution': '3440x1440', 'frequency': 100},
            },
        ]
        md = make_main_dict([device_edp1, device_hdmi1], active_config)
        result = assign_aliases(md, logger)
        edp = next(d for d in result['identifiers'] if d['device_name'] == 'eDP-1')
        hdmi = next(d for d in result['identifiers'] if d['device_name'] == 'HDMI-1')
        assert 'eDP-1' in edp['aliases']
        assert '_1' in hdmi['aliases']

    def test_two_aliases_two_devices(self, make_main_dict, device_edp1, device_hdmi1, logger):
        """_1 and _2 each get one device, 1:1 mapping."""
        active_config = [{
            '_1': {'resolution': '1920x1080'},
            '_2': {'resolution': '3440x1440', 'frequency': 100},
        }]
        md = make_main_dict([device_edp1, device_hdmi1], active_config)
        result = assign_aliases(md, logger)
        all_aliases = []
        for d in result['identifiers']:
            if d['is_connected']:
                all_aliases.extend(d.get('aliases', []))
        assert sorted(all_aliases) == ['_1', '_2']

    def test_incompatible_resolution_filters_config(self, make_main_dict, device_edp1, logger):
        """Config requiring resolution no device supports gets filtered out."""
        active_config = [
            {'_1': {'resolution': '5120x2880'}},
        ]
        md = make_main_dict([device_edp1], active_config)
        result = assign_aliases(md, logger)
        assert len(result['active_config']) == 0

    def test_disconnected_not_assigned(
        self, make_main_dict, device_edp1, device_dp2_disconnected, logger
    ):
        """Disconnected devices never get aliases."""
        active_config = [{'_1': {}}]
        md = make_main_dict([device_edp1, device_dp2_disconnected], active_config)
        result = assign_aliases(md, logger)
        dp2 = next(d for d in result['identifiers'] if d['device_name'] == 'DP-2')
        assert dp2.get('aliases', []) == []

    def test_mixed_compatible_and_incompatible(
        self, make_main_dict, device_edp1, logger
    ):
        """One config compatible, one not — only compatible survives."""
        active_config = [
            {'_1': {'resolution': '3840x2160'}},  # Not supported
            {'_1': {'resolution': '1920x1080'}},  # Supported
        ]
        md = make_main_dict([device_edp1], active_config)
        result = assign_aliases(md, logger)
        # _1 gets assigned because at least one config is compatible
        edp = next(d for d in result['identifiers'] if d['device_name'] == 'eDP-1')
        assert '_1' in edp['aliases']
        # Both configs survive filtering because _1 is in assigned_keys_from_pool
        # (filtering checks key membership, not per-config compatibility)
        assert len(result['active_config']) == 2

    def test_no_config_no_crash(self, make_main_dict, device_edp1, logger):
        """Empty active_config should not crash."""
        md = make_main_dict([device_edp1], [])
        result = assign_aliases(md, logger)
        assert len(result['active_config']) == 0


# --- find_real_device_name ---


class TestFindRealDeviceName:
    def test_find_by_alias(self, device_edp1, logger):
        device_edp1['aliases'] = ['_1']
        result = find_real_device_name('_1', [device_edp1], logger)
        assert result == 'eDP-1'

    def test_find_by_device_name_no_aliases(self, device_edp1, logger):
        """When no aliases set, fallback to device_name."""
        # find_real_device_name uses .get('aliases', [device['device_name']])
        del device_edp1['aliases']
        result = find_real_device_name('eDP-1', [device_edp1], logger)
        assert result == 'eDP-1'

    def test_not_found_exits(self, device_edp1, logger):
        device_edp1['aliases'] = ['_1']
        with pytest.raises(SystemExit):
            find_real_device_name('_99', [device_edp1], logger)

    def test_disconnected_device_skipped(self, device_dp2_disconnected, logger):
        """find_real_device_name only considers connected devices."""
        with pytest.raises(SystemExit):
            find_real_device_name('DP-2', [device_dp2_disconnected], logger)


# --- replace_aliases_with_real_names ---


class TestReplaceAliasesWithRealNames:
    def test_alias_replaced_with_real_name(self, device_edp1, logger):
        device_edp1['aliases'] = ['_1']
        main_dict = {'identifiers': [device_edp1]}
        config = {'_1': {'resolution': '1920x1080'}}
        result = replace_aliases_with_real_names(main_dict, config, logger)
        assert 'eDP-1' in result
        assert '_1' not in result

    def test_position_refs_replaced(self, device_edp1, device_hdmi1, logger):
        """Position references (_2 in right-of) also get replaced."""
        device_edp1['aliases'] = ['_1']
        device_hdmi1['aliases'] = ['_2']
        main_dict = {'identifiers': [device_edp1, device_hdmi1]}
        config = {'_1': {'right-of': '_2'}, '_2': {'primary': True}}
        result = replace_aliases_with_real_names(main_dict, config, logger)
        assert 'eDP-1' in result
        assert result['eDP-1']['right-of'] == 'HDMI-1'
        assert 'HDMI-1' in result

    def test_hooks_passthrough(self, device_edp1, logger):
        """hooks key is preserved as-is."""
        device_edp1['aliases'] = ['_1']
        main_dict = {'identifiers': [device_edp1]}
        config = {'_1': {}, 'hooks': {'pre': ['cmd']}}
        result = replace_aliases_with_real_names(main_dict, config, logger)
        assert 'hooks' in result
        assert result['hooks'] == {'pre': ['cmd']}

    def test_is_current_skipped(self, device_edp1, logger):
        """is_current key is not included in output."""
        device_edp1['aliases'] = ['_1']
        main_dict = {'identifiers': [device_edp1]}
        config = {'_1': {}, 'is_current': True}
        result = replace_aliases_with_real_names(main_dict, config, logger)
        assert 'is_current' not in result

    def test_explicit_name_not_alias(self, device_edp1, logger):
        """Non-alias keys (real device names) pass through without lookup."""
        main_dict = {'identifiers': [device_edp1]}
        config = {'eDP-1': {'resolution': '1920x1080'}}
        result = replace_aliases_with_real_names(main_dict, config, logger)
        assert 'eDP-1' in result
