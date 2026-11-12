"""Mirror a SimReady Hugging Face dataset at a pinned revision into a local directory; resumable.

Usage: python tools/simready_mirror.py <repo> <revision> <destination> [threads] [--with-duplicates]

The repo has more files than repo_info lists (100k), so the file list comes from per-folder tree
listings, cached in <destination>/.mirror/files.json. Files already present with the listed size are
skipped without network access; rate limits (429) back off and retry. Each asset's source_assets/ keeps
byte-identical copies of its Textures/ PNGs (only the .mdl files there are referenced by the USD); these are
not mirrored unless --with-duplicates is given. The editor's SimReady browser and `run.py asset simready
<asset>` download single assets into the same layout (shared/simready.py).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.simready import duplicates, fetch, retried  # noqa: E402


def file_list(repo, revision, state):
    """Listed folder by folder (a single recursive listing of 100k+ files is one fragile long request);
    each listed folder is appended to listing.jsonl so an interrupted listing resumes."""
    cached = state / 'files.json'
    if cached.is_file():
        return json.loads(cached.read_text())
    api = HfApi()
    top = retried(lambda: list(api.list_repo_tree(repo, repo_type='dataset', revision=revision)))
    files = [{'path': item.path, 'size': item.size} for item in top if hasattr(item, 'size')]
    folders = [item.path for item in top if not hasattr(item, 'size')]
    partial = state / 'listing.jsonl'
    listed = {}
    if partial.is_file():
        for line in partial.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                listed[record['folder']] = record['files']
    lock = threading.Lock()

    def list_folder(folder):
        entries = retried(lambda: [{'path': item.path, 'size': item.size} for item in api.list_repo_tree(
            repo, path_in_repo=folder, repo_type='dataset', revision=revision, recursive=True) if hasattr(item, 'size')])
        with lock, partial.open('a') as stream:
            stream.write(json.dumps({'folder': folder, 'files': entries}) + '\n')
            listed[folder] = entries
            if len(listed) % 100 == 0:
                print(f'  listed {len(listed)}/{len(folders)} folders', flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(list_folder, [folder for folder in folders if folder not in listed]))
    files += [entry for folder in folders for entry in listed[folder]]
    cached.write_text(json.dumps(files))
    return files


def main(repo, revision, destination, threads=16, with_duplicates=False):
    destination = Path(destination)
    state = destination / '.mirror'
    state.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = file_list(repo, revision, state)
    skipped = set() if with_duplicates else duplicates(files)
    listed_bytes = sum(item['size'] for item in files)
    files = [item for item in files if item['path'] not in skipped]
    total_bytes = sum(item['size'] for item in files)
    print(f'{repo}@{revision[:8]}: {len(files)} files, {total_bytes / 1e9:.1f} GB to mirror; {len(skipped)} duplicate '
          f'source_assets PNGs ({(listed_bytes - total_bytes) / 1e9:.1f} GB) left out '
          f'(listed in {time.time() - started:.0f}s)', flush=True)
    counts, failures, done_bytes, lock = {'downloaded': 0, 'skipped': 0}, [], [0], threading.Lock()

    def run(item):
        try:
            outcome = fetch(repo, revision, destination, item)
        except Exception as error:  # Recorded and reported; a rerun retries only what is missing.
            with lock:
                failures.append({'path': item['path'], 'error': f'{type(error).__name__}: {error}'[:300]})
            return
        with lock:
            counts[outcome] += 1
            done_bytes[0] += item['size']

    reporter_stop = threading.Event()

    def report():
        while not reporter_stop.wait(60):
            elapsed = time.time() - started
            with lock:
                finished = counts['downloaded'] + counts['skipped'] + len(failures)
                print(f'  {finished}/{len(files)} files, {done_bytes[0] / 1e9:.1f}/{total_bytes / 1e9:.1f} GB, '
                      f'{len(failures)} failed, {elapsed / 60:.0f} min', flush=True)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(run, files))
    reporter_stop.set()
    summary = {'repo': repo, 'revision': revision, 'url': f'https://huggingface.co/datasets/{repo}',
               'license': 'CC-BY-4.0', 'files': len(files), 'bytes': total_bytes,
               'duplicates_left_out': len(skipped), **counts,
               'failed': len(failures), 'finished': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'minutes': round((time.time() - started) / 60, 1)}
    (state / 'report.json').write_text(json.dumps({**summary, 'failures': failures}, indent=2))
    if not failures:
        (destination / 'SOURCE.json').write_text(json.dumps(
            {key: summary[key] for key in ('repo', 'revision', 'url', 'license', 'files', 'bytes', 'duplicates_left_out', 'finished')}, indent=2))
    print(json.dumps(summary), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    arguments = [value for value in sys.argv[1:] if value != '--with-duplicates']
    sys.exit(main(*arguments[:3], *(int(value) for value in arguments[3:4]),
                  with_duplicates='--with-duplicates' in sys.argv))
