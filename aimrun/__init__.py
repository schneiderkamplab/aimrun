from accelerate.state import PartialState
import aim
from functools import wraps
import sys
from threading import Thread

from .commands.sync import do_sync
from .utils import graceful_exit

_strict = True
def set_strict(strict):
    global _strict
    _strict = strict

_repo = None
def set_default_repo(repo):
    global _repo
    _repo = repo

_runs = []
_threads = []
def get_runs():
    return _runs

def on_main_process(function):
    @wraps(function)
    def execute_on_main_process(*args, **kwargs):
        PartialState().on_main_process(function)(*args, **kwargs)
    return execute_on_main_process

@on_main_process
def _track(*args, **kwargs):
    for run in _runs:
        run.track(*args, **kwargs)

@on_main_process
def _close():
    for run in _runs:
        run.close()
    graceful_exit()
    for thread in _threads:
        thread.join()

@on_main_process
def _init(repo=_repo, description=None, args=None, sync_repo=None, sync_args={}, **kwargs):
    global _runs
    if args is None:
        if _strict:
            raise ValueError("args is None - please provide a dictionary of hyperparameters to track!")
        else:
            print("WARNING: args is None - expecting a dictionary of hyperparameters to track.", file=sys.stderr)
    if repo is None:
        if _strict:
            raise ValueError("repo is None - please provide a repository to track the experiment!")
        else:
            print("WARNING: repo is None - defaulting to local repository to track the experiment.", file=sys.stderr)
    run = aim.Run(repo=repo, **kwargs)
    if args is not None:
        run['args'] = args
    if description is not None:
        run['description'] = description
    _runs.append(run)
    if sync_repo is not None:
        thread = Thread(target=do_sync, args=(repo, sync_repo, [run.hash]), kwargs=sync_args)
        thread.start()
        _threads.append(thread)
    return run

# aimrun interface

def init(repo=_repo, args=None, **kwargs):
    _init(repo=repo, args=args, **kwargs)

def track(*args, **kwargs):
    _track(*args, **kwargs)

def close():
    _close()

# wandb interface

class wandb:
    @staticmethod
    def init(project=None, name=None, config=None, **kwargs):
        _init(experiment=project, args=config, description=name, **kwargs)
    @staticmethod
    def log(*args, **kwargs):
        _track(*args, **kwargs)
    @staticmethod
    def finish():
        _close()
    @staticmethod
    def set_default_repo(repo):
        set_default_repo(repo)

# aim interface

class Run:
    def __init__(self, *args, **kwargs):
        _init(*args, **kwargs)
    def track(self, *args, **kwargs):
        _track(*args, **kwargs)
    def close(self):
        _close()
