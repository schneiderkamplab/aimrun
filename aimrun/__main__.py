import click

from .commands.extract import _extract
from .commands.plot import _plot
from .commands.sync import _sync
from .commands.agent import _agent

cli = click.CommandCollection(sources=[
    _extract,
    _plot,
    _sync,
    _agent
])

if __name__ == "__main__":
    cli()
