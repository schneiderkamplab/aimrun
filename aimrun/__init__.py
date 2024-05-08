from accelerate.state import PartialState
import aim
from functools import wraps
import sys

_strict = True
def set_strict(strict):
    global _strict
    _strict = strict

_repo = None
def set_default_repo(repo):
    global _repo
    _repo = repo

_runs = []
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

@on_main_process
def _init(repo=_repo, name=None, args=None, **kwargs):
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
    if name is not None:
        run['name'] = name
    _runs.append(run)
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
    def init(project=None, config=None, **kwargs):
        _init(experiment=project, args=config, **kwargs)
    @staticmethod
    def log(*args, **kwargs):
        _track(*args, **kwargs)
    @staticmethod
    def finish():
        _close()

# aim interface

class Run:
    def __init__(self, *args, **kwargs):
        _init(*args, **kwargs)
    def track(self, *args, **kwargs):
        _track(*args, **kwargs)
    def close(self):
        _close()
