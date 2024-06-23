from aim import Run
import click
from matplotlib import pyplot as plt
import os
import yaml

from ..utils import install_signal_handler, log, ERROR, _verbosity

def plot_multiple_lines(data, colors, legend_labels, ylim, plot_name):
    """
    Create a line plot with multiple lines, a legend, and custom colors.
    
    Parameters:
        data (list of lists): Each sublist contains data points for one line.
        colors (list of str): List of colors for each line.
        legend_labels (list of str): Labels for the legend.
        
    Returns:
        None
    """
    # Ensure equal aspect ratio for a square plot
    plt.figure(figsize=(8, 8))
    # Adjust layout for minimal padding
    plt.tight_layout(pad=0.2)
    # Plot each line
    for i, line_data in enumerate(data):
        plt.plot(line_data, color=colors[i], label=legend_labels[i])
    # Add legend
    plt.legend(loc='upper left')
    # Set axis range
    plt.ylim(ylim)
    # Show plot
    plt.savefig(plot_name, bbox_inches='tight')

@click.group()
def _plot():
    pass
@_plot.command()
@click.argument("figures", type=click.Path(exists=True), nargs=-1)
@click.option("--output-path", default=".", help="Path to save the plots (default: current directory)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--verbosity", default=_verbosity, help="Verbosity of the output (default: {_verbosity})")
def plot(figures, output_path, retries, sleep, verbosity):
    install_signal_handler()
    do_plot(figures, output_path, retries, sleep, verbosity)

def do_plot(
        figures,
        output_path=".",
        retries=10,
        sleep=1,
        verbosity=_verbosity,
    ):
    global _verbosity, _retries, _sleep
    _verbosity, _retries, _sleep = verbosity, retries, sleep
    for fs in figures:
        fs = yaml.safe_load(open(fs))
        std_repo = fs.pop("repo", None)
        std_color_defs = fs.pop("colors", None)
        std_ylim = fs.pop("ylim", None)
        std_metric = fs.pop("metric", None)
        for fname, fdef in fs.items():
            repo = fdef.get("repo", std_repo)
            if repo is None:
                log(ERROR, "no repository specified - skipping")
                continue
            color_defs = fdef.get("colors", std_color_defs)
            if color_defs is None:
                log(ERROR, "no colors specified - skipping")
                continue
            ylim = fdef.get("ylim", std_ylim)
            if ylim is None:
                log(ERROR, "no ylim specified - skipping")
                continue
            metric = fdef.get("metric", std_metric)
            if metric is None:
                log(ERROR, "no metric specified - skipping")
                continue
            runs = fdef.get("runs", [])
            if not runs:
                log(ERROR, "no runs specified - skipping")
                continue
            done = False
            data = []
            colors = []
            labels = []
            for r in runs:
                run = Run(run_hash=r["hash"], repo=repo)
                for seq in run.metrics():
                    if seq.name == metric:
                        data.append([val for _, (val, _, _) in seq.data.items()])
                        break
                else:
                    log(ERROR, f"metric {metric} not found for {r['hash']} - skipping")
                    break
                colors.append([(x if isinstance(x, float) else x/255) for x in color_defs.get(r["color"], r["color"])])
                labels.append(r["label"])
            else:
                done = True
            if not done:
                continue
            plot_multiple_lines(
                data=data,
                colors=colors,
                legend_labels=labels,
                ylim=ylim,
                plot_name=os.path.join(output_path, fname+".png"),
            )