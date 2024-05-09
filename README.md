# aimrun
A simple interface for integrating aim into MLOps frameworks. 

## Installation 
```bash
pip install aimrun
```

## Features
1. Multiple runs. Simply initialize multiple aimruns using ``aimrun.init()``, and track on multiple repositories at once. 
2. No need of main-process wrapper. You do not need to make sure, that only the main-process calls aimrun functions, we take care of that for you.

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

## Drop-in replacement Wandb (Experimental)
We experimentally offer aimrun as a drop-in replacement for wandb, making a seamless integration in your framework.

1. Replace``import wandb`` with ``from aimrun import wandb``
2. Set default repository før init (fx lige efter import)  ```wandb.set_default_repo('aim://172.3.66.145:53800')```
3. Supported functions ```.init(), .log(),  .finit()```
