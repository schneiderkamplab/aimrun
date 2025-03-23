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

def plot_multiple_lines(
        data,
        colors,
        legend_labels,
        xlim,
        ylim,
        plot_name,
        xsize,
        ysize,
        pad,
    ):
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
    plt.figure(figsize=(8 if xsize is None else xsize, 8 if ysize is None else ysize))
    # Adjust layout for minimal padding
    plt.tight_layout(pad=0.2 if pad is None else pad)
    # Plot each line
    for i, line_data in enumerate(data):
        plt.plot(*line_data, color=colors[i], label=legend_labels[i])
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

def ensure_int(x):
    if x is None or isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, list):
        return [ensure_int(y) for y in x]
    if x[-1].lower() == "g":
        return int(float(x[:-1])*10**9)
    if x[-1].lower() == "m":
        return int(float(x[:-1])*10**6)
    if x[-1].lower() == "k":
        return int(float(x[:-1])*10**3)
    return int(x)

@click.group()
def _plot():
    pass
@_plot.command()
@click.argument("figures", type=click.Path(exists=True), nargs=-1)
@click.option("--output-path", default=".", help="Path to save the plots (default: current directory)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
@click.option("--format", default="png", help="Format of the output plots (default: png)")
@click.option("--dump", is_flag=True, help="Dump the plot data to CSV files (default: False)")
def plot(figures, output_path, retries, sleep, verbosity, format, dump):
    install_signal_handler()
    do_plot(figures, output_path, retries, sleep, verbosity, format, dump)

def do_plot(
        figures,
        output_path=".",
        retries=10,
        sleep=1,
        verbosity=get_verbosity(),
        format="png",
        dump=False,
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
        std_xsize = fs.pop("xsize", None)
        std_ysize = fs.pop("ysize", None)
        std_pad = fs.pop("pad", None)
        std_xtimeoffset = fs.pop("xtimeoffset", None)
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
            xlim = ensure_int(fdef.get("xlim", std_xlim))
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
            xsize = fdef.get("xsize", std_xsize)
            ysize = fdef.get("ysize", std_ysize)
            pad = fdef.get("pad", std_pad)
            xtimeoffset = fdef.get("xtimeoffset", std_xtimeoffset)
            done = False
            data = []
            colors = []
            labels = []
            for r in runs:
                rs = [r] if isinstance(r, dict) else r
                proto_indices = []
                proto_raw_data = []
                color = None
                label = None
                _xtimeoffset = xtimeoffset
                for r in rs:
                    log(INFO, f"Fetching run {r['hash']}")
                    run = Run(run_hash=r["hash"], repo=repo, read_only=True)
                    scale = r.get("scale", 1.0)
                    if r.get("color") is not None:
                        if color is None:
                            color = r["color"]
                        else:
                            log(INFO, "WARNING: multiple colors specified - using first specified")
                    if r.get("label") is not None:
                        if label is None:
                            label = r["label"]
                        else:
                            log(INFO, "WARNING: multiple labels specified - using first specified")
                    for seq in run.metrics():
                        if seq.name == metric:
                            raw_data = [val/scale for _, (val, _, _) in seq.data.items()]
                            raw_data = raw_data[r.get("min", 0):r.get("max", len(raw_data))]
                            raw_data = smoothening(raw_data, smooth)
                            offset = r.get("offset", 0)
                            if _xtimeoffset is not None:
                                indices = [t for _, (_, _, t) in seq.data.items()]
                                indices = [(t-indices[0]+_xtimeoffset) for t in indices]
                                indices = indices[r.get("min", 0):r.get("max", len(indices))]
                                _xtimeoffset = 2*indices[-1]-indices[-2]
                            else:
                                indices = list(range(1+offset, len(raw_data)+1+offset))
                            proto_indices.extend(indices)
                            proto_raw_data.extend(raw_data)
                            break
                    else:
                        log(ERROR, f"metric {metric} not found for {r['hash']} - skipping")
                        continue
                data.append((proto_indices, proto_raw_data))
                colors.append([(x if isinstance(x, float) else x/255) for x in color_defs.get(color, color)])
                labels.append(label)
            else:
                done = True
            if not done:
                continue
            if dump:
                log(INFO, f"Dumping data to {output_path}")
                for i, (indices, raw_data) in enumerate(data):
                    dump_name = os.path.join(output_path, f"{fname}-{labels[i]}.csv")
                    with open(dump_name, "w") as f:
                        for j, val in zip(indices, raw_data):
                            f.write(f"{j},{val}\n")
            plot_name = os.path.join(output_path, f"{fname}.{format}")
            log(INFO, f"Plotting to {plot_name}")
            plot_multiple_lines(
                data=data,
                colors=colors,
                legend_labels=labels,
                xlim=xlim,
                ylim=ylim,
                plot_name=plot_name,
                xsize=xsize,
                ysize=ysize,
                pad=pad,
            )
        log(INFO, '='*40)
