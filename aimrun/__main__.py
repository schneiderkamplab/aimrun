import click

from .commands.plot import _plot
from .commands.sync import _sync

cli = click.CommandCollection(sources=[
    _plot,
    _sync,
])

if __name__ == "__main__":
    cli()
