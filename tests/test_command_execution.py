import subprocess

import pytest

from loose.loose import _execute_hooks, _execute_xrandr, run_command


# --- run_command ---


class TestRunCommand:
    def test_success(self, mocker, logger):
        mock_run = mocker.patch('loose.loose.subprocess.run')
        mock_run.return_value.returncode = 0
        assert run_command('echo hello', logger) == 0

    def test_failure(self, mocker, logger):
        mock_run = mocker.patch('loose.loose.subprocess.run')
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = b'error'
        mock_run.return_value.stdout = b''
        assert run_command('false', logger) == 1

    def test_nonzero_returncode_preserved(self, mocker, logger):
        mock_run = mocker.patch('loose.loose.subprocess.run')
        mock_run.return_value.returncode = 42
        mock_run.return_value.stderr = b''
        mock_run.return_value.stdout = b''
        assert run_command('exit 42', logger) == 42

    def test_timeout(self, mocker, logger):
        mocker.patch(
            'loose.loose.subprocess.run',
            side_effect=subprocess.TimeoutExpired('cmd', 30),
        )
        assert run_command('sleep 999', logger) == 1


# --- _execute_hooks ---


class TestExecuteHooks:
    def test_no_hooks_key(self, logger):
        assert _execute_hooks({}, 'pre', logger, dry_run=False) is True

    def test_no_matching_hook_type(self, logger):
        config = {'hooks': {'post': ['cmd']}}
        assert _execute_hooks(config, 'pre', logger, dry_run=False) is True

    def test_pre_hook_failure_returns_false(self, mocker, logger):
        mocker.patch('loose.loose.run_command', return_value=1)
        config = {'hooks': {'pre': ['failing_cmd']}}
        assert _execute_hooks(config, 'pre', logger, dry_run=False) is False

    def test_pre_hook_failure_ignored_with_flag(self, mocker, logger):
        mocker.patch('loose.loose.run_command', return_value=1)
        config = {'hooks': {'pre': ['failing_cmd']}}
        assert (
            _execute_hooks(
                config, 'pre', logger, dry_run=False, ignore_failing_hooks=True
            )
            is True
        )

    def test_post_hook_failure_continues(self, mocker, logger):
        mocker.patch('loose.loose.run_command', return_value=1)
        config = {'hooks': {'post': ['failing_cmd']}}
        assert _execute_hooks(config, 'post', logger, dry_run=False) is True

    def test_dry_run_no_execution(self, mocker, logger):
        mock_run = mocker.patch('loose.loose.run_command')
        config = {'hooks': {'pre': ['echo hi']}}
        _execute_hooks(config, 'pre', logger, dry_run=True)
        mock_run.assert_not_called()

    def test_multiple_hooks_first_fails_pre(self, mocker, logger):
        """First pre-hook fails, second should not run."""
        mock_run = mocker.patch('loose.loose.run_command', return_value=1)
        config = {'hooks': {'pre': ['cmd1', 'cmd2']}}
        assert _execute_hooks(config, 'pre', logger, dry_run=False) is False
        # Only the first hook should have been called
        mock_run.assert_called_once()

    def test_multiple_hooks_all_succeed(self, mocker, logger):
        mock_run = mocker.patch('loose.loose.run_command', return_value=0)
        config = {'hooks': {'pre': ['cmd1', 'cmd2']}}
        assert _execute_hooks(config, 'pre', logger, dry_run=False) is True
        assert mock_run.call_count == 2


# --- _execute_xrandr ---


class TestExecuteXrandr:
    def test_dry_run_no_execution(self, mocker, logger):
        mock_run = mocker.patch('loose.loose.run_command')
        result = _execute_xrandr(['xrandr', '--auto'], {}, logger, dry_run=True)
        assert result is True
        mock_run.assert_not_called()

    def test_success(self, mocker, logger):
        mocker.patch('loose.loose.run_command', return_value=0)
        assert _execute_xrandr(['xrandr', '--auto'], {}, logger, dry_run=False) is True

    def test_failure(self, mocker, logger):
        mocker.patch('loose.loose.run_command', return_value=1)
        assert _execute_xrandr(['xrandr', '--auto'], {}, logger, dry_run=False) is False
