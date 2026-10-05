import io
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from train_restart import atomic_checkpoint, child_arguments, publish_checkpoint, supervise


class RestartTests(unittest.TestCase):
    def args(self, directory, **kwargs):
        return SimpleNamespace(log_dir=Path(directory), resume=None, save_interval=2,
                               keep_checkpoints=2, max_restarts=2, restart_delay=0, **kwargs)

    def test_cli_stripping(self):
        self.assertEqual(child_arguments(['--auto-restart', '--max-restarts=4',
            '--resume', 'old.pt', '--save-interval', '2', '--keep-checkpoints=3',
            '--restart-delay', '0', '--recipe', 'config.json']), ['--recipe', 'config.json'])

    def test_atomic_failure_preserves_checkpoint(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'model_00001.pt'
            atomic_checkpoint(path, {'step':1})
            before=path.read_bytes()
            def fail(payload,handle):
                handle.write(b'partial')
                raise OSError('disk full')
            with patch('torch.save',side_effect=fail), self.assertRaises(OSError):
                atomic_checkpoint(path,{'step':2})
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_retention_only_numbered_checkpoints(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('model_00001.pt', 'model_00002.pt', 'model_00003.pt',
                         'model_best.pt', 'model_00004.pt.tmp', 'other.pt'):
                (root / name).write_text('fixture')
            status = root / 'status.json'
            publish_checkpoint(root / 'model_00003.pt', 2, status)
            self.assertFalse((root / 'model_00001.pt').exists())
            self.assertTrue((root / 'model_00002.pt').exists())
            self.assertTrue((root / 'model_best.pt').exists())
            self.assertTrue((root / 'model_00004.pt.tmp').exists())
            self.assertEqual(Path(json.loads(status.read_text())['checkpoint']), root.resolve() / 'model_00003.pt')

    def test_cuda_restart_uses_committed_checkpoint(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model_00001.pt'
            calls = []
            def launch(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    checkpoint.write_text('committed fixture')
                    publish_checkpoint(checkpoint, 2, kwargs['env']['DEEPSEEK_TRAIN_CHECKPOINT_STATUS'])
                    return Mock(stdout=io.StringIO('RuntimeError: CUDA error: the launch timed out\n'),
                                wait=Mock(return_value=1))
                return Mock(stdout=io.StringIO('finished\n'), wait=Mock(return_value=0))
            with patch('train_restart.subprocess.Popen', side_effect=launch):
                supervise(self.args(folder), ['--auto-restart','--config','tiny.json'])
            self.assertEqual(len(calls), 2)
            self.assertNotIn('--config',calls[1])
            self.assertEqual(calls[1][-2:], ['--resume', str(checkpoint.resolve())])
            self.assertNotIn('--auto-restart', calls[1])

    def test_non_cuda_failure_and_retry_limit(self):
        with TemporaryDirectory() as folder:
            for error, expected in [('ValueError: invalid dataset', 1), ('CUDA error: timeout', 3)]:
                with self.subTest(error=error):
                    def launch(*args, **kwargs):
                        return Mock(stdout=io.StringIO(error), wait=Mock(return_value=1))
                    with patch('train_restart.subprocess.Popen', side_effect=launch) as call:
                        with self.assertRaises(SystemExit):
                            supervise(self.args(folder), ['--auto-restart'])
                        self.assertEqual(call.call_count, expected)

    def test_interrupt_stops_child_without_restart(self):
        with TemporaryDirectory() as folder:
            process = Mock(stdout=io.StringIO(''), wait=Mock(side_effect=[KeyboardInterrupt, 0]))
            with patch('train_restart.subprocess.Popen', return_value=process) as call:
                with self.assertRaises(KeyboardInterrupt):
                    supervise(self.args(folder), ['--auto-restart'])
            process.terminate.assert_called_once()
            self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
