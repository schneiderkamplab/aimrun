# aimrun
A simple interface for integrating aim into MLOps frameworks. 

## Installation 
```bash
pip install aimrun
```

## Features
1. Multiple runs. Simply initialize multiple aimruns using ```aimrun.init()```, and track to multiple repositories at once. 
2. No need for a main-process wrapper. You do not need to make sure that only the main-process calls aimrun functions - we take care of that for you.
3. Sync project-specific (local) repositories to larger (remote) repositories. See ```python -m aimrun sync --help``` for guidance.

## Usage (Recommended)
1. Initialize one or more aimruns using ```aimrun.init()```
2. Use  ```aimrun.track()``` to track values. Parse a dictionary.
3. Use  ```aimrun.close()``` to finalize the experiments.


### Example usage
```python
import aimrun

# initalize 
aimrun.init(repo='aim://172.3.66.145:53800', experiment='my_experiment', description='description of run' args={"arg": 1}) # args=vars(args) if you use argsparse

# track 
aimrun.track({"value_0": A, "value_1": B})
# or 
aimrun.track(A, name="value_0")

# close 
aimrun.close() 
```

### Synchronizing on-going runs
```python
aimrun.init(repo=".", sync_repo='aim://172.3.66.145:53800', sync_args={"repeat": 60}, experiment='my_experiment', description='description of run' args={"arg": 1})
```
This starts a thread that incrementally synchronizes the current on-going run to a remote repo while using the current directory as the local repository.

To profit from mass updates (faster synchronization), consider installing an improved aim version:
```bash
pip install git+https://github.com/schneiderkamplab/aim
```

## Drop-in replacement Wandb (Experimental)
We experimentally offer aimrun as a drop-in replacement for wandb, making a seamless integration in your framework even easier.

1. Replace``import wandb`` with ``from aimrun import wandb``
2. Set default repository before init (e.g. right after import)  ```wandb.set_default_repo('aim://172.3.66.145:53800')```
3. Supported functions ```.init(), .log(),  .finish()```
