from aim import Repo, Run
import click
import os

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
@click.option("--output-path", default=".", help="Path to save the logs to (default: current directory)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def extract(repo_path, run, output_path, retries, sleep, verbosity):
    install_signal_handler()
    do_extract(repo_path, run, output_path, retries, sleep, verbosity)

def do_extract(
        repo_path=".",
        run=None,
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
    for run in runs:
        log(INFO, f"Fetching run {run}")
        run = Run(run_hash=r["hash"], repo=repo_path, read_only=True)
        logs = run.get_terminal_logs().values.tolist()
        logs = [x.data for x in logs]
        logs = '\n'.join(logs)
        file_name = os.path.join(output_path, f'{run}.log')
        with open(file_name, 'w') as f:
            f.write(logs)
        log(INFO, f"Logs saved to {file_name}")