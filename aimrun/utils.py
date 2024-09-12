import click
from enum import Enum
import signal
import time

# global state
_strict = True
def get_strict():
    return _strict
def set_strict(strict):
    global _strict
    _strict = strict

_repo = None
def get_repo():
    return _repo
def set_repo(repo):
    global _repo
    _repo = repo

_runs = []
def get_runs():
    return _runs

_threads = []
def get_threads():
    return _threads

# logging
base = time.time()
ERROR = 0
INFO = 1
PROGRESS = 2
DETAIL = 3
DEBUG = 4
_verbosity = PROGRESS
def log(verbosity, message, *args, **kwargs):
    global _verbosity
    if verbosity <= _verbosity:
        click.echo(f"[{time.time()-base:.2f}] {message}", *args, **kwargs)
def get_verbosity():
    return _verbosity
def set_verbosity(verbosity):
    global _verbosity
    _verbosity = verbosity

# data fetching
_retries = 1
_sleep = 0
def fetch(name, f, args=[], kwargs={}):
    retries = _retries
    while retries:
        try:
            return f(*args, **kwargs)
        except Exception as e:
            retries -= 1
            time.sleep(_sleep)
    raise RuntimeError(f"failed to fetch {name} after {_retries} retries")
def set_fetch(retries, sleep):
    global _retries, _sleep
    _retries, _sleep = retries, sleep

# chunking
def chunker(seq, size):
    return (seq[idx:idx+size] for idx in range(0,len(seq),size))

# signal handling
exit_flag = False
def graceful_exit():
    global exit_flag
    exit_flag = True
def signal_handler(sig, frame):
    graceful_exit()
    log(ERROR, "Ctrl-C pressed: exiting gracefully")
def install_signal_handler():
    signal.signal(signal.SIGINT, signal_handler)
def should_exit():
    return exit_flag

#sanitizing
def clean_args(obj):
    if isinstance(obj, dict):
        return {k: clean_args(v) for k, v in obj.items()}
    if isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
        return [clean_args(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool) or obj is None:
        return obj
    if get_strict():
        raise ValueError(f"Unexpected type {type(obj)} for {repr(obj)}")
    return repr(obj)
