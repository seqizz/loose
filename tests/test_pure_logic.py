import pytest

from loose.loose import _replace_none_with_dict, assert_unique_primary, has_loops


# --- has_loops ---


class TestHasLoops:
    def test_empty_config(self):
        assert has_loops({}) == (False, '')

    def test_single_device_no_position_refs(self):
        config = {1: [{'_1': {'resolution': '1920x1080'}}]}
        assert has_loops(config) == (False, '')

    def test_valid_chain_two_devices(self):
        config = {2: [{'_1': {'left-of': '_2'}, '_2': {'resolution': '1920x1080'}}]}
        assert has_loops(config) == (False, '')

    def test_valid_chain_three_devices(self):
        config = {3: [{
            '_1': {'left-of': '_2'},
            '_2': {'left-of': '_3'},
            '_3': {},
        }]}
        assert has_loops(config) == (False, '')

    def test_self_reference(self):
        config = {1: [{'_1': {'left-of': '_1'}}]}
        result = has_loops(config)
        assert result[0] is True
        assert 'self-reference' in result[1]

    def test_mutual_reference(self):
        config = {2: [{'_1': {'left-of': '_2'}, '_2': {'right-of': '_1'}}]}
        result = has_loops(config)
        assert result[0] is True
        assert 'referring to each other' in result[1]

    def test_three_node_cycle(self):
        config = {3: [{
            '_1': {'left-of': '_2'},
            '_2': {'above': '_3'},
            '_3': {'below': '_1'},
        }]}
        result = has_loops(config)
        assert result[0] is True

    def test_non_positional_keys_not_treated_as_edges(self):
        """resolution, frequency etc. should not create graph edges."""
        config = {2: [{'_1': {'resolution': '_2'}, '_2': {}}]}
        assert has_loops(config) == (False, '')

    def test_multiple_sections_one_has_loop(self):
        config = {
            1: [{'_1': {}}],
            2: [{'_1': {'left-of': '_2'}, '_2': {'right-of': '_1'}}],
        }
        result = has_loops(config)
        assert result[0] is True

    def test_multiple_config_items_in_same_section(self):
        """Two config items under same screen count, only second has loop."""
        config = {2: [
            {'_1': {'left-of': '_2'}, '_2': {}},
            {'_1': {'left-of': '_2'}, '_2': {'right-of': '_1'}},
        ]}
        result = has_loops(config)
        assert result[0] is True

    def test_hooks_key_ignored(self):
        """hooks dict should not be parsed for position refs."""
        config = {1: [{'_1': {'left-of': '_2'}, '_2': {}, 'hooks': {'pre': ['cmd']}}]}
        # hooks value is not a dict with position keys, isinstance check filters it
        assert has_loops(config) == (False, '')


# --- _replace_none_with_dict ---


class TestReplaceNoneWithDict:
    def test_simple_none(self):
        d = {'a': None}
        _replace_none_with_dict(d)
        assert d == {'a': {}}

    def test_nested_none(self):
        d = {'a': {'b': None, 'c': 'keep'}}
        _replace_none_with_dict(d)
        assert d == {'a': {'b': {}, 'c': 'keep'}}

    def test_no_nones(self):
        d = {'a': {'b': 'value'}}
        _replace_none_with_dict(d)
        assert d == {'a': {'b': 'value'}}

    def test_deeply_nested(self):
        d = {'a': {'b': {'c': None}}}
        _replace_none_with_dict(d)
        assert d['a']['b']['c'] == {}

    def test_empty_dict(self):
        d = {}
        _replace_none_with_dict(d)
        assert d == {}

    def test_multiple_nones(self):
        d = {'a': None, 'b': None, 'c': 'keep'}
        _replace_none_with_dict(d)
        assert d == {'a': {}, 'b': {}, 'c': 'keep'}


# --- assert_unique_primary ---


class TestAssertUniquePrimary:
    def test_single_primary_ok(self):
        data = {'on_screen_count': {1: [{'_1': {'primary': True}}]}}
        assert_unique_primary(data)  # Should not raise

    def test_no_primary_ok(self):
        data = {'on_screen_count': {1: [{'_1': {}}]}}
        assert_unique_primary(data)  # Should not raise

    def test_two_primaries_in_same_section_raises(self):
        data = {'on_screen_count': {2: [{
            '_1': {'primary': True},
            '_2': {'primary': True},
        }]}}
        with pytest.raises(ValueError, match='Multiple.*primary'):
            assert_unique_primary(data)

    def test_primaries_in_different_list_items_ok(self):
        """Each list item is a separate config option, so each can have its own primary."""
        data = {'on_screen_count': {1: [
            {'_1': {'primary': True}},
            {'_2': {'primary': True}},
        ]}}
        assert_unique_primary(data)  # Should not raise

    def test_hooks_not_confused_as_primary(self):
        """hooks dict doesn't have 'primary' key, should not interfere."""
        data = {'on_screen_count': {1: [{
            '_1': {'primary': True},
            'hooks': {'pre': ['cmd']},
        }]}}
        assert_unique_primary(data)  # Should not raise
