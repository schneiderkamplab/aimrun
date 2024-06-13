import click

from .commands.sync import _sync

cli = click.CommandCollection(sources=[
    _sync,
])

if __name__ == "__main__":
    cli()
