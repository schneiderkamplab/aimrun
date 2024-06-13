from aim import Repo
import click
import signal
import time
from tqdm import tqdm

def sync_run(src_repo, run_hash, dest_repo):
    def copy_trees():
        # copy run meta tree
        source_meta_tree = src_repo.request_tree(
            'meta', run_hash, read_only=True, from_union=False, no_cache=True
        ).subtree('meta')
        dest_meta_tree = dest_repo.request_tree(
            'meta', run_hash, read_only=False, from_union=False, no_cache=True
        ).subtree('meta')
        dest_meta_run_tree = dest_meta_tree.subtree('chunks').subtree(run_hash)
        dest_meta_tree[...] = source_meta_tree[...]
        dest_index = dest_repo._get_index_tree('meta', timeout=10).view(())
        dest_meta_run_tree.finalize(index=dest_index)

        # copy run series tree
        source_series_run_tree = src_repo.request_tree(
            'seqs', run_hash, read_only=True, no_cache=True
        ).subtree('seqs')
        dest_series_run_tree = dest_repo.request_tree(
            'seqs', run_hash, read_only=False, no_cache=True
        ).subtree('seqs')

        # copy v2 sequences
        source_v2_tree = source_series_run_tree.subtree(('v2', 'chunks', run_hash))
        dest_v2_tree = dest_series_run_tree.subtree(('v2', 'chunks', run_hash))
        for ctx_id in source_v2_tree.keys():
            for metric_name in source_v2_tree.subtree(ctx_id).keys():
                source_val_view = source_v2_tree.\
                    subtree((ctx_id, metric_name)).array('val')
                source_step_view = source_v2_tree.\
                    subtree((ctx_id, metric_name)).array('step', dtype='int64')
                source_epoch_view = source_v2_tree.\
                    subtree((ctx_id, metric_name)).array('epoch', dtype='int64')
                source_time_view = source_v2_tree.\
                    subtree((ctx_id, metric_name)).array('time', dtype='int64')

                dest_val_view = dest_v2_tree.\
                    subtree((ctx_id, metric_name)).array('val').allocate()
                dest_step_view = dest_v2_tree.\
                    subtree((ctx_id, metric_name)).array('step', dtype='int64').allocate()
                dest_epoch_view = dest_v2_tree.\
                    subtree((ctx_id, metric_name)).array('epoch', dtype='int64').allocate()
                dest_time_view = dest_v2_tree.\
                    subtree((ctx_id, metric_name)).array('time', dtype='int64').allocate()

                for key, val in source_val_view.items():
                    dest_val_view[key] = val
                    dest_step_view[key] = source_step_view[key]
                    dest_epoch_view[key] = source_epoch_view[key]
                    dest_time_view[key] = source_time_view[key]

        # copy v1 sequences
        source_v1_tree = source_series_run_tree.subtree(('chunks', run_hash))
        dest_v1_tree = dest_series_run_tree.subtree(('chunks', run_hash))
        for ctx_id in source_v1_tree.keys():
            for metric_name in source_v1_tree.\
                    subtree(ctx_id).keys():
                source_val_view = source_v1_tree.\
                    subtree((ctx_id, metric_name)).array('val')
                source_epoch_view = source_v1_tree.\
                    subtree((ctx_id, metric_name)).array('epoch', dtype='int64')
                source_time_view = source_v1_tree.\
                    subtree((ctx_id, metric_name)).array('time', dtype='int64')

                dest_val_view = dest_v1_tree.\
                    subtree((ctx_id, metric_name)).array('val').allocate()
                dest_epoch_view = dest_v1_tree.\
                    subtree((ctx_id, metric_name)).array('epoch', dtype='int64').allocate()
                dest_time_view = dest_v1_tree.\
                    subtree((ctx_id, metric_name)).array('time', dtype='int64').allocate()

                for key, val in source_val_view.items():
                    dest_val_view[key] = val
                    dest_epoch_view[key] = source_epoch_view[key]
                    dest_time_view[key] = source_time_view[key]

    def copy_structured_props():
        source_structured_run = src_repo.structured_db.find_run(run_hash)
        dest_structured_run = dest_repo.request_props(run_hash,
                                                        read_only=False,
                                                        created_at=source_structured_run.created_at)
        dest_structured_run.name = source_structured_run.name
        dest_structured_run.experiment = source_structured_run.experiment
        dest_structured_run.description = source_structured_run.description
        dest_structured_run.archived = source_structured_run.archived
        for source_tag in source_structured_run.tags:
            dest_structured_run.add_tag(source_tag)

    if dest_repo.is_remote_repo:
        # create remote run
        try:
            copy_trees()
            copy_structured_props()
        except Exception as e:
            raise e
    else:
        with dest_repo.structured_db:  # rollback destination db entity if subsequent actions fail.
            # copy run structured data
            copy_structured_props()
            copy_trees()

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
    click.echo("Ctrl-C pressed: exiting gracefully")

@click.group()
def _sync():
    pass
@_sync.command()
@click.argument("src_repo_path", type=str)
@click.argument("dst_repo_path", type=str)
@click.option("--run", default=None, help="Specific run hash to synchronize (default: None)")
@click.option("--offset", default=0, help="Offset for the duration in seconds (default: 0)")
@click.option("--eps", default=1.0, help="Error margin for the duration in seconds (default: 1.0)")
@click.option("--retries", default=3, help="Number of retries to fetch run (default: 3)")
@click.option("--sleep", default=1.0, help="Sleep time in seconds between retries (default: 1.0)")
@click.option("--repeat", default=60.0, help="Sleep time in seconds between repetitions(default: 60.0)")
def sync(src_repo_path, dst_repo_path, run, offset, eps, retries, sleep, repeat):
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        try:
            src_repo = Repo(path=src_repo_path)
            dst_repo = Repo(path=dst_repo_path)
            runs = [run.hash for run in src_repo.iter_runs()] if run is None else [run]
            successes = []
            failures = []
            skips = []
            for run_hash in tqdm(runs):
                if exit_flag:
                    break
                try:
                    click.echo(f"fetching run for {run_hash} from destination repository")
                    dst_run = fetch_run(dst_repo, run_hash, retries=retries, sleep=sleep)
                    if dst_run is None:
                        click.echo(f"syncing {run_hash}: run hash not found in destination repository")
                    else:
                        click.echo(f"fetching run for {run_hash} from source repository")
                        src_run = fetch_run(src_repo, run_hash, retries=retries, sleep=sleep)
                        diff = abs(src_run.duration + offset - dst_run.duration)
                        if src_run.active == dst_run.active and diff < eps:
                            click.echo(f"skipping {run_hash}: run hash exists with {diff}s difference in duration")
                            skips.append(run_hash)
                            continue
                        click.echo(f"syncing {run_hash}: run hash exists with {diff}s difference in duration")
                    sync_run(src_repo, run_hash, dst_repo)
                    click.echo(f"sucesss: successfully synchronized {run_hash}")
                    successes.append(run_hash)
                except Exception as e:
                    click.echo(f"failure: failed to synchronize {run_hash} - {e}")
                    failures.append((run_hash, e))
            if len(skips) > 0:
                click.echo(f"summary: skipped {len(skips)} runs - {skips}")
            if len(successes) > 0:
                click.echo(f"summary: successfully synchronized {len(successes)} runs - {successes}")
            if len(failures) > 0:
                click.echo(f"summary: failed to synchronize {len(failures)} runs - {failures}")
        except Exception as e:
            click.echo(f"failure: failed to synchronize runs - {e}")
        finally:
            src_repo.close()
            dst_repo.close()
        if repeat <= 0 or exit_flag:
            return
        wait_time = repeat
        click.echo(f"waiting {wait_time}s before next repetition: ", nl=False)
        while wait_time > 0:
            time.sleep(min(wait_time, 1.0))
            click.echo(".", nl=False)
            wait_time -= 1.0
            if exit_flag:
                click.echo(" interrupted")
                return
        click.echo(" done")
