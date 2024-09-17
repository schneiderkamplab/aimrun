from aim import Run
import click
from matplotlib import pyplot as plt
import os
from scipy.signal import savgol_filter
import yaml

from ..utils import (
    ERROR,
    INFO,
    install_signal_handler,
    get_verbosity,
    log,
    set_fetch,
    set_verbosity,
)

def plot_multiple_lines(data, colors, legend_labels, xlim, ylim, plot_name):
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
    # Set axis x range
    if xlim is not None:
        plt.xlim(xlim)
    # Set axis y range
    if ylim is not None:
        plt.ylim(ylim)
    # Show plot
    plt.savefig(plot_name, bbox_inches='tight')

def smoothening(vector, smooth):
    if smooth is None:
        return vector
    alg, window = smooth[:2]
    if alg == "mean":
        return [sum(vector[i:i+window])/window for i in range(len(vector)-window+1)]
    elif alg == "median":
        return [sorted(vector[i:i+window])[window//2] for i in range(len(vector)-window+1)]
    elif alg == "exponential":
        alpha = 2/(window+1)
        result = [vector[0]]
        for i in range(1, len(vector)):
            result.append(alpha*vector[i] + (1-alpha)*result[-1])
        return result
    elif alg == "savitzky-golay":
        return savgol_filter(vector, window, smooth[2])
    else:
        raise ValueError(f"unknown smoothing algorithm {alg}")

@click.group()
def _plot():
    pass
@_plot.command()
@click.argument("figures", type=click.Path(exists=True), nargs=-1)
@click.option("--output-path", default=".", help="Path to save the plots (default: current directory)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def plot(figures, output_path, retries, sleep, verbosity):
    install_signal_handler()
    do_plot(figures, output_path, retries, sleep, verbosity)

def do_plot(
        figures,
        output_path=".",
        retries=10,
        sleep=1,
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    set_fetch(retries, sleep)
    for fs in figures:
        log(INFO, f"Loading {fs}")
        fs = yaml.safe_load(open(fs))
        std_repo = fs.pop("repo", None)
        std_color_defs = fs.pop("colors", None)
        std_xlim = fs.pop("xlim", None)
        std_ylim = fs.pop("ylim", None)
        std_metric = fs.pop("metric", None)
        std_smooth = fs.pop("smooth", None)
        for fname, fdef in fs.items():
            log(INFO, '-'*40)
            log(INFO, f"Processing figure {fname}")
            repo = fdef.get("repo", std_repo)
            if repo is None:
                log(ERROR, "no repository specified - skipping")
                continue
            color_defs = fdef.get("colors", std_color_defs)
            if color_defs is None:
                log(ERROR, "no colors specified - skipping")
                continue
            xlim = fdef.get("xlim", std_xlim)
            ylim = fdef.get("ylim", std_ylim)
            metric = fdef.get("metric", std_metric)
            if metric is None:
                log(ERROR, "no metric specified - skipping")
                continue
            runs = fdef.get("runs", [])
            if not runs:
                log(ERROR, "no runs specified - skipping")
                continue
            smooth = fdef.get("smooth", std_smooth)
            done = False
            data = []
            colors = []
            labels = []
            for r in runs:
                log(INFO, f"Fetching run {r['hash']}")
                run = Run(run_hash=r["hash"], repo=repo, read_only=True)
                scale = r.get("scale", 1.0)
                for seq in run.metrics():
                    if seq.name == metric:
                        data.append(smoothening([val/scale for _, (val, _, _) in seq.data.items()], smooth))
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
            plot_name = os.path.join(output_path, f"{fname}.png")
            log(INFO, f"Plotting to {plot_name}")
            plot_multiple_lines(
                data=data,
                colors=colors,
                legend_labels=labels,
                xlim=xlim,
                ylim=ylim,
                plot_name=plot_name,
            )
        log(INFO, '='*40)
