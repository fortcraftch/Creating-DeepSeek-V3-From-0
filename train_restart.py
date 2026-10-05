"""Process isolation for CUDA recovery; never reuse a poisoned CUDA context."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from collections import deque


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_checkpoint(path, keep, status=None):
    """Called only after an atomic checkpoint commit. Prune this run's names."""
    path = Path(path).resolve()
    if status:
        atomic_json(status, {'checkpoint': str(path)})
    if keep:
        checkpoints = sorted(
            (p for p in path.parent.iterdir()
             if re.fullmatch(r'model_\d+\.pt', p.name) and p.is_file() and not p.is_symlink()),
            key=lambda p: int(p.stem.split('_')[1]), reverse=True)
        # Always protect the just-committed checkpoint, even after an older resume.
        others = [p for p in checkpoints if p != path]
        for old in others[max(0, keep - 1):]:
            old.unlink()


def child_arguments(argv):
    flags = {'--auto-restart': 0, '--max-restarts': 1, '--restart-delay': 1,
             '--resume': 1, '--save-interval': 1, '--keep-checkpoints': 1}
    result = []
    i = 0
    while i < len(argv):
        value = argv[i]
        key = value.split('=', 1)[0]
        if key in flags:
            i += 1 + (flags[key] if '=' not in value else 0)
        else:
            result.append(value)
            i += 1
    return result


def cuda_failure(output):
    text = output.lower()
    return any(marker in text for marker in (
        'cuda error:', 'cuda_error_launch_timeout', 'cudaerrorlaunchtimeout',
        'torch.acceleratorerror: cuda', 'torch.cuda.outofmemoryerror'))


def supervise(args, argv):
    base = [sys.executable, '-u', str(Path(__file__).with_name('train_deepseek_v3.py')), *child_arguments(argv),
            '--save-interval', str(args.save_interval),
            '--keep-checkpoints', str(args.keep_checkpoints)]
    resume = args.resume
    args.log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='deepseek-train-') as temporary:
        status = Path(temporary) / 'checkpoint.json'
        env = dict(os.environ, DEEPSEEK_TRAIN_CHECKPOINT_STATUS=str(status))
        for attempt in range(args.max_restarts + 1):
            command = list(base)
            if resume:
                for i in range(len(command)-1, -1, -1):
                    if command[i] == '--config': del command[i:i+2]
                    elif command[i].startswith('--config='): del command[i]
                command += ['--resume', str(resume)]
            print(f'[supervisor] attempt={attempt + 1}, resume={resume}', flush=True)
            tail = deque(maxlen=100)
            with (args.log_dir / 'restart.log').open('a', encoding='utf-8') as log:
                log.write(f'\nAttempt {attempt + 1}; resume={resume}\n')
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                        tail.append(line)
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    process.stdout.close()
            if code == 0:
                return
            if not cuda_failure(''.join(tail)) or attempt == args.max_restarts:
                raise SystemExit(code)
            if status.exists():
                resume = Path(json.loads(status.read_text(encoding='utf-8'))['checkpoint'])
                if not resume.is_file():
                    raise RuntimeError(f'Committed checkpoint is missing: {resume}')
            print(f'[supervisor] CUDA failure; restarting in {args.restart_delay}s '
                  f'from {resume or "the initial state (no checkpoint yet)"}', flush=True)
            time.sleep(args.restart_delay)


def atomic_checkpoint(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    try:
        with temporary.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
