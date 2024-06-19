from aim import Repo
import click
import datetime
import signal
import time
from tqdm import tqdm

base = time.time()

ERROR = 0
PROGRESS = 1
INFO = 2
DETAIL = 3
DEBUG = 4
verbosity = PROGRESS
def log(level, message, *args, **kwargs):
    global verbosity
    if level <= verbosity:
        click.echo(f"[{time.time()-base:.2f}] {message}", *args, **kwargs)

def fetch_items(view, retries, sleep):
    _retries = retries
    while _retries:
        try:
            return list(view.items())
        except Exception as e:
            _retries -= 1
            time.sleep(sleep)
    raise RuntimeError(f"failed to fetch items after {retries} retries")

def chunker(seq, size):
    return (seq[idx:idx+size] for idx in range(0,len(seq),size))

def sync_run(src_repo, run_hash, dest_repo, mass_update, retries, sleep, full_copy):
    def copy_trees():
        log(DETAIL, "copy run meta tree")
        source_meta_tree = src_repo.request_tree(
            'meta', run_hash, read_only=True, from_union=False, no_cache=True
        ).subtree('meta')
        dest_meta_tree = dest_repo.request_tree(
            'meta', run_hash, read_only=False, from_union=False, no_cache=True
        ).subtree('meta')
        dest_meta_run_tree = dest_meta_tree.subtree('chunks').subtree(run_hash)
        dest_traces = None if full_copy else dest_meta_run_tree.get('traces', None)
        dest_meta_tree[...] = source_meta_tree[...]
        dest_index = dest_repo._get_index_tree('meta', timeout=10).view(())
        dest_meta_run_tree.finalize(index=dest_index)

        log(DETAIL, "copy run series tree")
        source_series_run_tree = src_repo.request_tree(
            'seqs', run_hash, read_only=True, no_cache=True
        ).subtree('seqs')
        dest_series_run_tree = dest_repo.request_tree(
            'seqs', run_hash, read_only=False, no_cache=True
        ).subtree('seqs')

        log(DETAIL, "copy v2 sequences")
        source_v2_tree = source_series_run_tree.subtree(('v2', 'chunks', run_hash))
        dest_v2_tree = dest_series_run_tree.subtree(('v2', 'chunks', run_hash))
        for ctx_id in tqdm(list(source_v2_tree.keys()), desc="contexts", disable=verbosity < DETAIL):
            for metric_name in (metric_bar := tqdm(list(source_v2_tree.subtree(ctx_id).keys()), desc="metrics", disable=verbosity < DETAIL)):
                metric_bar.set_postfix_str(f"{ctx_id}/{metric_name}")
                log(DEBUG, f"obtain val view for {ctx_id}/{metric_name}")
                source_val_view = source_v2_tree.subtree((ctx_id, metric_name)).array('val')
                log(DEBUG, f"obtain step view for {ctx_id}/{metric_name}")
                source_step_view = source_v2_tree.subtree((ctx_id, metric_name)).array('step', dtype='int64')
                log(DEBUG, f"obtain epoch view for {ctx_id}/{metric_name}")
                source_epoch_view = source_v2_tree.subtree((ctx_id, metric_name)).array('epoch', dtype='int64')
                log(DEBUG, f"obtain time view for {ctx_id}/{metric_name}")
                source_time_view = source_v2_tree.subtree((ctx_id, metric_name)).array('time', dtype='int64')

                log(DEBUG, f"allocate val view for {ctx_id}/{metric_name}")
                dest_val_view = dest_v2_tree.subtree((ctx_id, metric_name)).array('val').allocate()
                log(DEBUG, f"allocate step view for {ctx_id}/{metric_name}")
                dest_step_view = dest_v2_tree.subtree((ctx_id, metric_name)).array('step', dtype='int64').allocate()
                log(DEBUG, f"allocate epoch view for {ctx_id}/{metric_name}")
                dest_epoch_view = dest_v2_tree.subtree((ctx_id, metric_name)).array('epoch', dtype='int64').allocate()
                log(DEBUG, f"allocate time view for {ctx_id}/{metric_name}")
                dest_time_view = dest_v2_tree.subtree((ctx_id, metric_name)).array('time', dtype='int64').allocate()

                last_step = -1
                if dest_traces is not None:
                    _context = dest_traces.get(ctx_id, None)
                    if _context is not None:
                        _metric = _context.get(metric_name, None)
                        if _metric is not None:
                            last_step = _metric.get('last_step', -1)
                new_keys = {k for k, v in fetch_items(source_step_view, retries, sleep) if v > last_step}
                log(DEBUG, f"last step for {metric_name} is {last_step} and there are {len(new_keys)} new keys")

                if mass_update:
                    for chunk in chunker([x for x in fetch_items(source_val_view, retries, sleep) if x[0] in new_keys], size=mass_update):
                        dest_val_view.update(chunk)
                    for chunk in chunker([x for x in fetch_items(source_step_view, retries, sleep) if x[0] in new_keys], size=mass_update):
                        dest_step_view.update(chunk)
                    for chunk in chunker([x for x in fetch_items(source_epoch_view, retries, sleep) if x[0] in new_keys], size=mass_update):
                        dest_epoch_view.update(chunk)
                    for chunk in chunker([x for x in fetch_items(source_time_view, retries, sleep) if x[0] in new_keys], size=mass_update):
                        dest_time_view.update(chunk)
                    continue
                for key, val in tqdm(list(source_val_view.items()), "keys", disable=verbosity < 3):
                    dest_val_view[key] = val
                    dest_step_view[key] = source_step_view[key]
                    dest_epoch_view[key] = source_epoch_view[key]
                    dest_time_view[key] = source_time_view[key]
        log(DETAIL, "finished syncing v2 sequences")

        log(DETAIL, "copy v1 sequences")
        source_v1_tree = source_series_run_tree.subtree(('chunks', run_hash))
        dest_v1_tree = dest_series_run_tree.subtree(('chunks', run_hash))
        for ctx_id in tqdm(list(source_v1_tree.keys()), desc="contexts", disable=verbosity < DETAIL):
            for metric_name in (metric_bar := tqdm(list(source_v1_tree.subtree(ctx_id).keys()), desc="metrics", disable=verbosity < DETAIL)):
                metric_bar.set_postfix_str(f"{ctx_id}/{metric_name}")
                log(DEBUG, f"obtain val view for {ctx_id}/{metric_name}")
                source_val_view = source_v1_tree.subtree((ctx_id, metric_name)).array('val')
                log(DEBUG, f"obtain epoch view for {ctx_id}/{metric_name}")
                source_epoch_view = source_v1_tree.subtree((ctx_id, metric_name)).array('epoch', dtype='int64')
                log(DEBUG, f"obtain time view for {ctx_id}/{metric_name}")
                source_time_view = source_v1_tree.subtree((ctx_id, metric_name)).array('time', dtype='int64')

                log(DEBUG, f"allocate val view for {ctx_id}/{metric_name}")
                dest_val_view = dest_v1_tree.subtree((ctx_id, metric_name)).array('val').allocate()
                log(DEBUG, f"allocate epoch view for {ctx_id}/{metric_name}")
                dest_epoch_view = dest_v1_tree.subtree((ctx_id, metric_name)).array('epoch', dtype='int64').allocate()
                log(DEBUG, f"allocate time view for {ctx_id}/{metric_name}")
                dest_time_view = dest_v1_tree.subtree((ctx_id, metric_name)).array('time', dtype='int64').allocate()

                last_step = -1
                if dest_traces is not None:
                    _context = dest_traces.get(ctx_id, None)
                    if _context is not None:
                        _metric = _context.get(metric_name, None)
                        if _metric is not None:
                            last_step = _metric.get('last_step', -1)
                log(DEBUG, f"last step for {metric_name} is {last_step}")

                if mass_update:
                    for chunk in chunker([x for x in fetch_items(source_val_view, retries, sleep) if x[0] > last_step], size=mass_update):
                        dest_val_view.update(chunk)
                    for chunk in chunker([x for x in fetch_items(source_epoch_view, retries, sleep) if x[0] > last_step], size=mass_update):
                        dest_epoch_view.update(chunk)
                    for chunk in chunker([x for x in fetch_items(source_time_view, retries, sleep) if x[0] > last_step], size=mass_update):
                        dest_time_view.update(chunk)
                    continue
                for key, val in tqdm(list(source_val_view.items()), desc="keys", disable=verbosity < 3):
                    dest_val_view[key] = val
                    dest_epoch_view[key] = source_epoch_view[key]
                    dest_time_view[key] = source_time_view[key]
        log(DETAIL, "finished syncing v1 sequences")
        del dest_v1_tree, dest_v2_tree, dest_series_run_tree, dest_meta_tree, dest_index, dest_meta_run_tree

    def copy_structured_props():
        log(DETAIL, "copy run structured properties")
        source_structured_run = src_repo.request_props(run_hash, read_only=True) #structured_db.find_run(run_hash)
        created_at = datetime.datetime.fromtimestamp(source_structured_run.creation_time, tz=datetime.timezone.utc)
        dest_structured_run = dest_repo.request_props(run_hash,
                                                        read_only=False,
                                                        created_at=created_at)
        dest_structured_run.name = source_structured_run.name
        dest_structured_run.experiment = source_structured_run.experiment
        dest_structured_run.description = source_structured_run.description
        dest_structured_run.archived = source_structured_run.archived
        for source_tag in source_structured_run.tags:
            dest_structured_run.add_tag(source_tag)

    if dest_repo.is_remote_repo:
        copy_trees()
        log(DETAIL, "finished copying run trees")
        copy_structured_props()
        log(DETAIL, "finished copying run structured properties")
    else:
        with dest_repo.structured_db:
            copy_structured_props()
            log(DETAIL, "finished copying run structured properties")
            copy_trees()
            log(DETAIL, "finished copying run trees")

def fetch_run(repo, run_hash, retries, sleep):
    _retries = retries
    while _retries:
        try:
            return repo.get_run(run_hash)
        except Exception as e:
            _retries -= 1
            time.sleep(sleep)
    raise RuntimeError(f"failed to fetch run {run_hash} after {retries} retries")

exit_flag = False
def signal_handler(sig, frame):
    global exit_flag
    exit_flag = True
    log(ERROR, "Ctrl-C pressed: exiting gracefully")

@click.group()
def _sync():
    pass
@_sync.command()
@click.argument("src_repo_path", type=str)
@click.argument("dst_repo_path", type=str)
@click.option("--run", default=None, multiple=True, help="Specific run hash(es) to synchronize (default: None)")
@click.option("--offset", default=0, help="Offset for the duration in seconds (default: 0)")
@click.option("--eps", default=0.00001, help="Error margin for the duration (default: 1e5)")
@click.option("--retries", default=10, help="Number of retries to fetch run (default: 10)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--repeat", default=0, help="Sleep time in seconds between repetitions (0 to deactivate) (default: 0)")
@click.option("--force", is_flag=True, help="Force synchronization of all runs (default: False)")
@click.option("--first", default=0, help="First run to synchronize (default: 0)")
@click.option("--last", default=-1, help="Last run to synchronize (default: -1)")
@click.option("--mass-update", default=0, help="Mass update chunk size (0 to deactivate) (default: 0)")
@click.option("--raise-errors", is_flag=True, help="Raise errors during synchronization (default: False)")
@click.option("--verbosity-level", default=verbosity, help="Verbosity of the output (default: {verbosity})")
@click.option("--full-copy", is_flag=True, help="Full copy of the runs (default: False)")
def sync(src_repo_path, dst_repo_path, run, offset, eps, retries, sleep, repeat, force, first, last, mass_update, raise_errors, verbosity_level, full_copy):
    global verbosity
    verbosity = verbosity_level
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        src_repo = None
        dst_repo = None
        try:
            log(DETAIL, f"opening source repository at {src_repo_path}")
            src_repo = Repo(path=src_repo_path)
            log(DETAIL, f"opening destination repository at {dst_repo_path}")
            dst_repo = Repo(path=dst_repo_path)
            log(DETAIL, f"fetching runs from source repository")
            runs = run if run else [run.hash for run in src_repo.iter_runs()]
            successes = []
            failures = []
            skips = []
            _first = first
            _last = last
            while _first < 0:
                _first += len(runs)
            while _last < 0:
                _last += len(runs)
            for idx, run_hash in tqdm(enumerate(runs), total=len(runs), disable=verbosity < PROGRESS):
                if idx < _first or idx > _last:
                    continue
                if exit_flag:
                    break
                try:
                    log(DETAIL, f"fetching run for {run_hash} from destination repository")
                    dst_run = fetch_run(dst_repo, run_hash, retries=retries, sleep=sleep)
                    if force:
                        log(INFO, f"syncing {run_hash}: force synchronization")
                    elif dst_run is None:
                        log(INFO, f"syncing {run_hash}: run hash not found in destination repository")
                    else:
                        log(DETAIL, f"fetching run for {run_hash} from source repository")
                        src_run = fetch_run(src_repo, run_hash, retries=retries, sleep=sleep)
                        diff = abs(src_run.duration + offset - dst_run.duration)
                        if src_run.active == dst_run.active and diff < eps:
                            log(INFO, f"skipping {run_hash}: run hash exists with {diff} difference in duration")
                            skips.append(run_hash)
                            continue
                        log(INFO, f"syncing {run_hash}: run hash exists with {diff} difference in duration")
                    sync_run(src_repo, run_hash, dst_repo, mass_update=mass_update, retries=retries, sleep=sleep, full_copy=full_copy)
                    log(INFO, f"sucesss: successfully synchronized {run_hash}")
                    successes.append(run_hash)
                except Exception as e:
                    log(ERROR, f"failure: failed to synchronize {run_hash} - {e}")
                    failures.append((run_hash, e))
                    if raise_errors:
                        raise e
            if len(skips) > 0:
                log(PROGRESS, f"summary: skipped {len(skips)} runs - {skips}")
            if len(successes) > 0:
                log(PROGRESS, f"summary: successfully synchronized {len(successes)} runs - {successes}")
            if len(failures) > 0:
                log(PROGRESS, f"summary: failed to synchronize {len(failures)} runs - {failures}")
        except Exception as e:
            log(ERROR, f"failure: failed to synchronize runs - {e}")
            if raise_errors:
                raise e
        finally:
            if src_repo is not None:
                src_repo.close()
            if dst_repo is not None:
                dst_repo.close()
        if repeat <= 0 or exit_flag:
            return
        wait_time = repeat
        log(INFO, f"waiting {wait_time}s before next repetition: ", nl=False)
        while wait_time > 0:
            time.sleep(min(wait_time, 1.0))
            log(DETAIL, ".", nl=False)
            wait_time -= 1.0
            if exit_flag:
                log(DETAIL, " interrupted")
                return
        log(INFO, " done")
