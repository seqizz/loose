import pytest

from loose.loose import _safe_file_operation, load_from_disk, save_to_disk


# --- _safe_file_operation ---


class TestSafeFileOperation:
    def test_read_yaml(self, tmp_path, logger):
        f = tmp_path / 'test.yaml'
        f.write_text('key: value\n')
        result = _safe_file_operation(str(f), 'read_yaml', logger)
        assert result == {'key': 'value'}

    def test_read_yaml_missing_exits(self, logger):
        with pytest.raises(SystemExit):
            _safe_file_operation('/nonexistent/path.yaml', 'read_yaml', logger)

    def test_read_text(self, tmp_path, logger):
        f = tmp_path / 'test.txt'
        f.write_text('hello world')
        result = _safe_file_operation(str(f), 'read_text', logger)
        assert result == 'hello world'

    def test_write_and_read_pickle_roundtrip(self, tmp_path, logger):
        f = tmp_path / 'test.pkl'
        data = {'test': [1, 2, 3], 'nested': {'a': 'b'}}
        _safe_file_operation(str(f), 'write_pickle', logger, data)
        result = _safe_file_operation(str(f), 'read_pickle', logger)
        assert result == data

    def test_read_pickle_missing_returns_none(self, logger):
        result = _safe_file_operation('/nonexistent.pkl', 'read_pickle', logger)
        assert result is None


# --- save_to_disk + load_from_disk ---


class TestSaveAndLoadFromDisk:
    def test_marks_applied_config_as_current(self, tmp_path, logger):
        save_path = str(tmp_path / 'state.pkl')
        config_a = {'_1': {'resolution': '1920x1080'}, 'is_current': True}
        config_b = {'_1': {'resolution': '3440x1440'}}
        old_dict = {
            'active_config': [config_a, config_b],
            'identifiers': [],
            'VERSION': '0.2.7',
            'raw_config': {},
        }
        save_to_disk(config_b, {}, logger, old_dict, save_path)
        loaded = load_from_disk(save_path, logger)

        current_configs = [c for c in loaded['active_config'] if c.get('is_current')]
        assert len(current_configs) == 1
        assert current_configs[0]['_1']['resolution'] == '3440x1440'

    def test_removes_current_from_previous(self, tmp_path, logger):
        """Previously current config should lose is_current marker."""
        save_path = str(tmp_path / 'state.pkl')
        config_a = {'_1': {'resolution': '1920x1080'}, 'is_current': True}
        config_b = {'_1': {'resolution': '3440x1440'}}
        old_dict = {
            'active_config': [config_a, config_b],
            'identifiers': [],
            'VERSION': '0.2.7',
            'raw_config': {},
        }
        save_to_disk(config_b, {}, logger, old_dict, save_path)
        loaded = load_from_disk(save_path, logger)

        old_config = next(
            c for c in loaded['active_config']
            if c['_1']['resolution'] == '1920x1080'
        )
        assert 'is_current' not in old_config

    def test_preserves_identifiers(self, tmp_path, logger):
        save_path = str(tmp_path / 'state.pkl')
        identifiers = [{'device_name': 'eDP-1', 'is_connected': True}]
        config_a = {'_1': {}}
        old_dict = {
            'active_config': [config_a],
            'identifiers': identifiers,
            'VERSION': '0.2.7',
            'raw_config': {},
        }
        save_to_disk(config_a, {'test': True}, logger, old_dict, save_path)
        loaded = load_from_disk(save_path, logger)

        assert loaded['identifiers'] == identifiers
        assert loaded['raw_config'] == {'test': True}

    def test_load_missing_returns_none(self, logger):
        result = load_from_disk('/nonexistent/path.pkl', logger)
        assert result is None
