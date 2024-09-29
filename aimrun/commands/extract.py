from aim import Repo, Run
import click
import os
import pandas as pd
from tqdm import tqdm

from ..utils import (
    DETAIL,
    INFO,
    install_signal_handler,
    get_verbosity,
    log,
    set_fetch,
    set_verbosity,
)

@click.group()
def _extract():
    pass
@_extract.command()
@click.option("--repo-path", default=".", help="Path to the repository (default: current directory)")
@click.option("--run", default=None, multiple=True, help="Specific run hash(es) to synchronize (default: None)")
@click.option("--metric", default=None, multiple=True, help="Specific run hash(es) to synchronize (empty string for none) (default: all)")
@click.option("--terminal-logs/--no-terminal-logs", default=True, help="Fetch terminal logs (default: True)")
@click.option("--output-path", default=".", help="Path to save the logs to (default: current directory)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def extract(repo_path, run, metric, terminal_logs, output_path, retries, sleep, verbosity):
    install_signal_handler()
    do_extract(repo_path, run, metric, terminal_logs, output_path, retries, sleep, verbosity)

def do_extract(
        repo_path=".",
        run=None,
        metric=None,
        terminal_logs=False,
        output_path=".",
        retries=10,
        sleep=1,
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    set_fetch(retries, sleep)
    log(DETAIL, f"opening repository at {repo_path}")
    repo = Repo(path=repo_path)
    log(DETAIL, f"fetching runs from repository")
    runs = [r for ru in run for r in ru.split()] if run else [run.hash for run in repo.iter_runs()]
    for run_hash in tqdm(runs):
        log(INFO, f"Fetching run {run_hash}")
        run = Run(run_hash=run_hash, repo=repo_path, read_only=True)
        if terminal_logs:
            log(INFO, f"Fetching terminal logs {run_hash}")
            logs = run.get_terminal_logs()
            if logs is None:
                log(INFO, f"Terminal logs not found for {run_hash}")
                continue
            logs = logs.values.tolist()
            logs = [x.data for x in logs]
            logs = '\n'.join(logs)
            file_name = os.path.join(output_path, f'{run_hash}.terminal_logs.txt')
            with open(file_name, 'w') as f:
                f.write(logs)
            log(INFO, f"Terminal logs saved to {file_name}")
        metrics = [m for me in metric for m in me.split()] if metric else ["terminal_logs"]+[seq.name for seq in run.metrics()]
        for seq in run.metrics():
            if all(metric.lower() not in seq.name.lower() for metric in metrics):
                continue
            log(INFO, f"Fetching metric {seq.name}")
            data = [(step, val, epoch, _time) for step, (val, epoch, _time) in seq.data.items()]
            df = pd.DataFrame(data, columns=["step", "val", "epoch", "timestamp"])
            file_name = os.path.join(output_path, f'{run_hash}.{seq.name.replace("/","__")}.csv')
            df.to_csv(file_name, index=False)
            log(INFO, f"Metric {seq.name} saved to {file_name}")
