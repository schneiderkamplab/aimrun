from aim import Run
import click
import os

from ..utils import (
    ERROR,
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
@click.argument("run_hash", default=[], type=str, nargs=-1)
@click.option("--repo-path", default=".", help="Path to the repository (default: current directory)")
@click.option("--output-path", default=".", help="Path to save the logs to (default: current directory)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def extract(run_hash, repo_path, output_path, retries, sleep, verbosity):
    install_signal_handler()
    do_extract(run_hash, repo_path, output_path, retries, sleep, verbosity)

def do_extract(
        run_hash=[],
        repo_path=".",
        output_path=".",
        retries=10,
        sleep=1,
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    set_fetch(retries, sleep)
    for run_h in run_hash:
        log(INFO, f"Fetching run {run_h}")
        run = Run(run_hash=r["hash"], repo=repo_path, read_only=True)
        logs = run.get_terminal_logs().values.tolist()
        logs = [x.data for x in logs]
        logs = '\n'.join(logs)
        file_name = os.path.join(output_path, f'{run_h}.log')
        with open(file_name, 'w') as f:
            f.write(logs)
        log(INFO, f"Logs saved to {file_name}")