from accelerate.state import PartialState
import aim
from functools import wraps
import sys
from threading import Thread

from .commands.sync import do_sync
from .utils import clean_args, get_repo, get_runs, get_strict, get_threads, graceful_exit, set_repo

def on_main_process(function):
    @wraps(function)
    def execute_on_main_process(*args, **kwargs):
        PartialState().on_main_process(function)(*args, **kwargs)
    return execute_on_main_process

@on_main_process
def _track(*args, **kwargs):
    for run in get_runs():
        run.track(*args, **kwargs)

@on_main_process
def _close():
    for run in get_runs():
        run.close()
    graceful_exit()
    for thread in get_threads():
        thread.join()

@on_main_process
def _init(repo=get_repo(), description=None, args=None, sync_repo=None, sync_args={}, **kwargs):
    if args is None:
        if get_strict():
            raise ValueError("args is None - please provide a dictionary of hyperparameters to track!")
        else:
            print("WARNING: args is None - expecting a dictionary of hyperparameters to track.", file=sys.stderr)
    if repo is None:
        if get_strict():
            raise ValueError("repo is None - please provide a repository to track the experiment!")
        else:
            print("WARNING: repo is None - defaulting to local repository to track the experiment.", file=sys.stderr)
    run = aim.Run(repo=repo, **kwargs)
    if args is not None:
        run['args'] = clean_args(args)
    if description is not None:
        run['description'] = description
    get_runs().append(run)
    if sync_repo is not None:
        thread = Thread(target=do_sync, args=(repo, sync_repo, [run.hash]), kwargs=sync_args)
        thread.start()
        get_threads().append(thread)
    return run

# aimrun interface

def init(repo=get_repo(), args=None, **kwargs):
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
        set_repo(repo)

# aim interface

class Run:
    def __init__(self, *args, **kwargs):
        _init(*args, **kwargs)
    def track(self, *args, **kwargs):
        _track(*args, **kwargs)
    def close(self):
        _close()
