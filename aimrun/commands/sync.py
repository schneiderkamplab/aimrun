from aim import Repo
import click
import datetime
import time
from tqdm import tqdm

from ..utils import (
    ERROR,
    PROGRESS,
    INFO,
    DETAIL,
    DEBUG,
    chunker,
    fetch,
    get_verbosity,
    install_signal_handler,
    log,
    set_fetch,
    set_verbosity,
    should_exit,
)

def fetch_items(view):
    return fetch("items", lambda v: list(v.items()), args=[view])

def fetch_run(repo, run_hash):
    return fetch("run", lambda r, h: r.get_run(h), args=[repo, run_hash])

def fetch_traces(run_tree):
    return fetch("traces", lambda x: x.get('traces', None), args=[run_tree])

def sync_run(src_repo, run_hash, dest_repo, mass_update, retries, sleep, full_copy):
    def copy_trees():
        nonlocal mass_update
        num_chunks = num_items = 0
        log(DETAIL, "copy run meta tree")
        source_meta_tree = src_repo.request_tree(
            'meta', run_hash, read_only=True, from_union=False, no_cache=True
        ).subtree('meta')
        dest_meta_tree = dest_repo.request_tree(
            'meta', run_hash, read_only=False, from_union=False, no_cache=True
        ).subtree('meta')
        dest_meta_run_tree = dest_meta_tree.subtree('chunks').subtree(run_hash)
        dest_traces = None if full_copy else fetch_traces(dest_meta_run_tree)
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
        for ctx_id in source_v2_tree.keys():
            for metric_name in source_v2_tree.subtree(ctx_id).keys():
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
                new_keys = {k for k, v in fetch_items(source_step_view) if v > last_step}
                log(DETAIL, f"last step for {metric_name} is {last_step} and there are {len(new_keys)} new keys")

                if mass_update < 0:
                    try:
                        dest_val_view.update([])
                        mass_update = -mass_update
                        log(DETAIL, f"detected mass update-compatible server - using chunk size of {mass_update}")
                    except Exception as e:
                        print(e)
                        mass_update = 0
                        log(DETAIL, f"unable to detect mass update-compatible server - deactivating mass update")
                if mass_update:
                    for chunk in chunker([x for x in fetch_items(source_val_view) if x[0] in new_keys], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} value items")
                        dest_val_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    for chunk in chunker([x for x in fetch_items(source_step_view) if x[0] in new_keys], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} step items")
                        dest_step_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    for chunk in chunker([x for x in fetch_items(source_epoch_view) if x[0] in new_keys], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} epoch items")
                        dest_epoch_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    for chunk in chunker([x for x in fetch_items(source_time_view) if x[0] in new_keys], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} time items")
                        dest_time_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    continue
                for key, val in (x for x in fetch_items(source_val_view) if x[0] in new_keys):
                    log(DEBUG, f"updating single value, step, epoch, and time")
                    dest_val_view[key] = val
                    dest_step_view[key] = source_step_view[key]
                    dest_epoch_view[key] = source_epoch_view[key]
                    dest_time_view[key] = source_time_view[key]
                    num_chunks += 4
                    num_items += 4
        log(DETAIL, "finished syncing v2 sequences")

        log(DETAIL, "copy v1 sequences")
        source_v1_tree = source_series_run_tree.subtree(('chunks', run_hash))
        dest_v1_tree = dest_series_run_tree.subtree(('chunks', run_hash))
        for ctx_id in source_v1_tree.keys():
            for metric_name in source_v1_tree.subtree(ctx_id).keys():
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
                log(DETAIL, f"last step for {metric_name} is {last_step} and there are {len([x for x in fetch_items(source_val_view) if x[0] > last_step]) if get_verbosity() >= DETAIL else None} new keys")

                if mass_update:
                    for chunk in chunker([x for x in fetch_items(source_val_view) if x[0] > last_step], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} value items")
                        dest_val_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    for chunk in chunker([x for x in fetch_items(source_epoch_view) if x[0] > last_step], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} epoch items")
                        dest_epoch_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    for chunk in chunker([x for x in fetch_items(source_time_view) if x[0] > last_step], size=mass_update):
                        log(DEBUG, f"updating {len(chunk)} time items")
                        dest_time_view.update(chunk)
                        num_chunks += 1
                        num_items += len(chunk)
                    continue
                for key, val in (x for x in fetch_items(source_val_view) if x[0] > last_step):
                    log(DEBUG, f"updating single value, epoch, and time")
                    dest_val_view[key] = val
                    dest_epoch_view[key] = source_epoch_view[key]
                    dest_time_view[key] = source_time_view[key]
                    num_chunks += 3
                    num_items += 3
        log(DETAIL, "finished syncing v1 sequences")
        del dest_v1_tree, dest_v2_tree, dest_series_run_tree, dest_meta_tree, dest_index, dest_meta_run_tree
        return num_chunks, num_items

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
        num_chunks, num_items = copy_trees()
        log(DETAIL, "finished copying run trees")
        copy_structured_props()
        log(DETAIL, "finished copying run structured properties")
    else:
        with dest_repo.structured_db:
            copy_structured_props()
            log(DETAIL, "finished copying run structured properties")
            num_chunks, num_items = copy_trees()
            log(DETAIL, "finished copying run trees")
    return num_chunks, num_items

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
@click.option("--mass-update", default=-128, help="Mass update chunk size (0 to deactivate, negative to detect) (default: -128)")
@click.option("--raise-errors", is_flag=True, help="Raise errors during synchronization (default: False)")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
@click.option("--full-copy", is_flag=True, help="Full copy of the runs (default: False)")
def sync(src_repo_path, dst_repo_path, run, offset, eps, retries, sleep, repeat, force, first, last, mass_update, raise_errors, verbosity, full_copy):
    install_signal_handler()
    do_sync(src_repo_path, dst_repo_path, run, offset, eps, retries, sleep, repeat, force, first, last, mass_update, raise_errors, verbosity, full_copy)

def do_sync(
        src_repo_path,
        dst_repo_path,
        run,
        offset=0,
        eps=0.00001,
        retries=10,
        sleep=1,
        repeat=0,
        force=False,
        first=0,
        last=-1,
        mass_update=-128,
        raise_errors=False,
        verbosity=get_verbosity(),
        full_copy=False,
    ):
    set_verbosity(verbosity)
    set_fetch(retries, sleep)
    while True:
        src_repo = None
        dst_repo = None
        try:
            log(DETAIL, f"opening source repository at {src_repo_path}")
            src_repo = Repo(path=src_repo_path)
            log(DETAIL, f"opening destination repository at {dst_repo_path}")
            dst_repo = Repo(path=dst_repo_path)
            log(DETAIL, f"fetching runs from source repository")
            runs = [r for ru in run for r in ru.split()] if run else [run.hash for run in src_repo.iter_runs()]
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
                if should_exit():
                    break
                try:
                    log(DETAIL, f"fetching run for {run_hash} from destination repository")
                    dst_run = fetch_run(dst_repo, run_hash)
                    if force:
                        log(INFO, f"syncing {run_hash}: force synchronization")
                    elif dst_run is None:
                        log(INFO, f"syncing {run_hash}: run hash not found in destination repository")
                    else:
                        log(DETAIL, f"fetching run for {run_hash} from source repository")
                        src_run = fetch_run(src_repo, run_hash)
                        diff = abs(src_run.duration + offset - dst_run.duration)
                        if src_run.active == dst_run.active and diff < eps:
                            log(INFO, f"skipping {run_hash}: run hash exists with {diff} difference in duration")
                            skips.append(run_hash)
                            continue
                        log(INFO, f"syncing {run_hash}: run hash exists with {diff} difference in duration")
                    num_chunks, num_items = sync_run(src_repo, run_hash, dst_repo, mass_update=mass_update, retries=retries, sleep=sleep, full_copy=full_copy)
                    log(INFO, f"sucesss: successfully synchronized {run_hash} ({num_chunks} chunks and {num_items} items copied)")
                    successes.append(run_hash)
                except Exception as e:
                    log(ERROR, f"failure: failed to synchronize {run_hash} - {e}")
                    failures.append((run_hash, e))
                    if raise_errors:
                        raise e
            if len(skips) > 0:
                log(PROGRESS, f"summary: skipped {len(skips)} runs - {' '.join(skips)}")
            if len(successes) > 0:
                log(PROGRESS, f"summary: successfully synchronized {len(successes)} runs - {' '.join(successes)}")
            if len(failures) > 0:
                log(PROGRESS, f"summary: failed to synchronize {len(failures)} runs - {' '.join(failures)}")
        except Exception as e:
            log(ERROR, f"failure: failed to synchronize runs - {e}")
            if raise_errors:
                raise e
        finally:
            if src_repo is not None:
                src_repo.close()
            if dst_repo is not None:
                dst_repo.close()
        if repeat <= 0 or should_exit():
            return
        wait_time = repeat
        log(INFO, f"waiting {wait_time}s before next repetition: ", nl=False)
        while wait_time > 0:
            time.sleep(min(wait_time, 1.0))
            log(DETAIL, ".", nl=False)
            wait_time -= 1.0
            if should_exit():
                log(DETAIL, " interrupted")
                return
        log(INFO, " done")
