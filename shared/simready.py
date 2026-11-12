"""SimReady assets from the pinned Hugging Face datasets: the local mirror first, downloaded when missing.

Assets keep the dataset layout under assets/simready/<publisher>/<asset>/, as tools/simready_mirror.py
mirrors whole datasets; ``prepare`` completes one asset there and wraps it into assets/<asset>/
(shared.asset_formats.simready) for usd_object. File lists come from the mirror's .mirror/files.json,
else from tree listings cached under .cache/simready/<publisher>/. Network calls give up after a few
attempts and further ones fail fast for a minute, so an offline machine still lists its local assets.
"""

from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / 'assets' / 'simready'
CACHE = ROOT / '.cache' / 'simready'
LICENSE = 'CC-BY-4.0'
USD = ('.usd', '.usda', '.usdc')
# Pinned dataset revisions; `category` for publishers whose asset names carry no category prefix.
SOURCES = {
    'nvidia': {'repo': 'nvidia/simready-assets', 'revision': 'a72e6a76d53501cbf5a719910a674eb2c1f6f161'},
    'deeprobotics': {'repo': 'deeproboticslab/simready-assets',
                     'revision': '38f97b2ab0a583c9537fc9e1d878b38ee6e63e09', 'category': 'robot'},
    'robotiq': {'repo': 'Robotiq-Official/simready-assets',
                'revision': '7bb7b202d9e5f9749f3cee91d1c7f585ba70a5db', 'category': 'gripper'},
}


class Offline(OSError):
    """Hugging Face could not be reached (no network, a timeout, or HF_HUB_OFFLINE)."""


def retried(call, attempts=8):
    """Retry rate limits, server and network errors with backoff; other client errors raise at once."""
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(attempts):
        try:
            return call()
        except (HfHubHTTPError, OSError, httpx.HTTPError) as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            if attempt == attempts - 1 or (status is not None and 400 <= status < 500 and status != 429):
                raise
            retry_after = response.headers.get('Retry-After') if response is not None else None
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(300, 5 * 2 ** attempt))


_unreachable = [0.0]  # Until when network calls fail fast after a connection failure.


def _online(call, attempts=2):
    import httpx
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
    if time.time() < _unreachable[0]:
        raise Offline('Hugging Face 暂时无法连接（一分钟内不再重试）')
    try:
        return retried(call, attempts)
    except (httpx.TransportError, ConnectionError, LocalEntryNotFoundError, OfflineModeIsEnabled) as error:
        _unreachable[0] = time.time() + 60
        raise Offline(f'无法连接 Hugging Face：{type(error).__name__}: {str(error)[:200]}') from error


def fetch(repo, revision, destination, item, attempts=8):
    """Download one listed file into `destination` (the dataset layout) unless present with its size."""
    from huggingface_hub import hf_hub_download
    target = Path(destination) / item['path']
    if target.is_file() and target.stat().st_size == item['size']:
        return 'skipped'
    retried(lambda: hf_hub_download(repo, item['path'], repo_type='dataset', revision=revision,
                                    local_dir=destination), attempts)
    return 'downloaded'


def duplicates(files):
    """source_assets PNGs whose Textures counterpart is listed with the same size (the USD loads only
    the .mdl files there)."""
    sizes = {item['path']: item['size'] for item in files}
    return {path for path, size in sizes.items() if '/source_assets/' in path and path.endswith('.png')
            and sizes.get(path.replace('/source_assets/', '/Textures/')) == size}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


_lock = threading.RLock()
_mirrored = {}


def _mirror_files(publisher):
    """{asset: listed files} from a mirror's .mirror/files.json (a complete listing of the pinned revision)."""
    with _lock:
        if publisher not in _mirrored:
            path = MIRROR / publisher / '.mirror' / 'files.json'
            grouped = {}
            if path.is_file():
                for item in json.loads(path.read_text()):
                    name, separator, _ = item['path'].partition('/')
                    if separator:
                        grouped.setdefault(name, []).append(item)
            _mirrored[publisher] = grouped
        return _mirrored[publisher]


def _source(publisher):
    if publisher not in SOURCES:
        raise KeyError(f'未知的 SimReady 来源 {publisher!r}；可选 {", ".join(SOURCES)}')
    return SOURCES[publisher]['repo'], SOURCES[publisher]['revision']


def names(publisher, online=True):
    """Asset folders of a publisher: the mirror's file list and local folders, plus the online
    top-level listing (cached) when there is no complete mirror listing."""
    found = set(_mirror_files(publisher))
    if (MIRROR / publisher).is_dir():
        found.update(path.parent.name for path in (MIRROR / publisher).glob('*/com.nvidia.simready.packaging.json'))
    if not _mirror_files(publisher):
        cached = CACHE / publisher / 'assets.json'
        if cached.is_file():
            found.update(json.loads(cached.read_text()))
        elif online:
            from huggingface_hub import HfApi
            repo, revision = _source(publisher)
            listed = _online(lambda: [item.path for item in HfApi().list_repo_tree(
                repo, repo_type='dataset', revision=revision) if not hasattr(item, 'size')])
            _write(cached, listed)
            found.update(listed)
    return sorted(found)


def files(publisher, name, online=True):
    """Listed files of one asset (paths from the dataset root), or None offline without a listing."""
    listed = _mirror_files(publisher).get(name)
    if listed is not None:
        return listed
    cached = CACHE / publisher / 'files' / f'{name}.json'
    if cached.is_file():
        return json.loads(cached.read_text())
    if not online:
        return None
    from huggingface_hub import HfApi
    repo, revision = _source(publisher)
    listed = _online(lambda: [{'path': item.path, 'size': item.size} for item in HfApi().list_repo_tree(
        repo, path_in_repo=name, repo_type='dataset', revision=revision, recursive=True) if hasattr(item, 'size')])
    if not listed:
        raise KeyError(f'{publisher}/{name} 不在 {repo}@{revision[:8]}')
    _write(cached, listed)
    return listed


def root_usd(name, listed):
    """The asset's USD: the single USD file directly in simready_usd/ (as root_usds.json lists it)."""
    roots = sorted(item['path'] for item in listed if item['path'].count('/') == 2
                   and item['path'].startswith(f'{name}/simready_usd/') and item['path'].endswith(USD))
    return roots[0] if roots else None


def thumbnail_source(name, listed):
    """Path of the official preview: the white-background render of the root USD (as on the dataset
    card), else its plain render, else any preview the asset has."""
    paths = {item['path'] for item in listed}
    root = root_usd(name, listed)
    folder = f'{name}/simready_usd/.thumbs/'
    preferred = [f'{folder}256x256/{root.rsplit("/", 1)[1]}{suffix}.png' for suffix in ('_light', '')] if root else []
    preferred += sorted(path for path in paths if path.startswith(folder + '256x256/') and path.endswith('_light.png'))
    preferred += sorted(path for path in paths if path.startswith(folder) and path.endswith('_thumbnail.png'))
    preferred += sorted(path for path in paths if path.startswith(folder + '256x256/') and path.endswith('.png')
                        and not path.endswith('_transparent.png'))
    return next((path for path in preferred if path in paths), None)


def category(publisher, name):
    return SOURCES[publisher].get('category') or name.split('_', 1)[0]


def local_root(publisher, name, listed=None):
    """The asset's root USD inside the mirror, if present (files it loads may still be missing)."""
    folder = MIRROR / publisher / name / 'simready_usd'
    if listed:
        root = root_usd(name, listed)
        return MIRROR / publisher / root if root and (MIRROR / publisher / root).is_file() else None
    found = sorted(path for path in folder.glob('*') if path.suffix in USD) if folder.is_dir() else []
    return found[0] if found else None


def missing(publisher, name, listed):
    """Listed files (duplicates left out) absent from the mirror or of another size."""
    skipped = duplicates(listed)
    absent = []
    for item in listed:
        target = MIRROR / publisher / item['path']
        if item['path'] not in skipped and not (target.is_file() and target.stat().st_size == item['size']):
            absent.append(item)
    return absent


def catalog(online=True):
    """Every known asset as {publisher, name, category, local, bytes}, and {publisher: problem} for
    publishers that could not be listed online. `bytes` (duplicates left out) is None until listed."""
    entries, problems = [], {}
    for publisher in SOURCES:
        try:
            found = names(publisher, online)
        except Exception as error:  # noqa: BLE001 - offline: keep the local assets
            problems[publisher] = str(error)
            found = names(publisher, online=False)
        for name in found:
            listed = files(publisher, name, online=False)
            skipped = duplicates(listed) if listed else ()
            entries.append({'publisher': publisher, 'name': name, 'category': category(publisher, name),
                            'local': local_root(publisher, name, listed) is not None,
                            'bytes': sum(item['size'] for item in listed if item['path'] not in skipped) if listed else None})
    return entries, problems


def thumbnail(publisher, name, target, size=128, online=True):
    """Write the official preview, scaled to fit `size` px, to `target`; False when the asset has none
    or it is not available offline."""
    listed = files(publisher, name, online)
    source = thumbnail_source(name, listed) if listed else None
    if source is None:
        return False
    path = MIRROR / publisher / source
    if not path.is_file():
        path = CACHE / publisher / source
        if not path.is_file():
            if not online:
                return False
            from huggingface_hub import hf_hub_download
            repo, revision = _source(publisher)
            _online(lambda: hf_hub_download(repo, source, repo_type='dataset', revision=revision,
                                            local_dir=CACHE / publisher))
    from PIL import Image
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'{target.name}.{threading.get_ident()}.tmp')
    jpeg = target.suffix.lower() in ('.jpg', '.jpeg')
    with Image.open(path) as image:
        image.thumbnail((size, size), Image.LANCZOS)
        if jpeg:
            image = image.convert('RGB')
        image.save(temporary, format='JPEG' if jpeg else 'PNG', **({'quality': 88} if jpeg else {}))
    os.replace(temporary, target)
    return True


def preview(publisher, name, size=320, online=True):
    """Cached JPEG of the official preview at `size` px (for browsing), or None when unavailable."""
    target = CACHE / 'previews' / str(size) / publisher / f'{name}.jpg'
    if target.is_file() or thumbnail(publisher, name, target, size, online):
        return target
    return None


def download(publisher, name, progress=None, threads=8):
    """Complete one asset in the mirror (duplicates left out); returns its root USD.
    `progress(done_bytes, total_bytes)` follows the files still to fetch."""
    listed = files(publisher, name)
    root = root_usd(name, listed)
    if root is None:
        raise ValueError(f'{publisher}/{name}: simready_usd/ 下没有 USD 文件')
    absent = missing(publisher, name, listed)
    if absent:
        repo, revision = _source(publisher)
        destination = MIRROR / publisher
        total, done, lock = sum(item['size'] for item in absent), [0], threading.Lock()

        def run(item, attempts=4):
            fetch(repo, revision, destination, item, attempts)
            with lock:
                done[0] += item['size']
                if progress:
                    progress(done[0], total)

        first, rest = absent[0], absent[1:]  # One call settles whether the network is reachable.
        _online(lambda: run(first, 1))
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(run, rest))
    return MIRROR / publisher / root


def wrap(publisher, name, output=None):
    """assets/<asset>/ (or `output`) wrapping the mirrored asset; an existing wrapper of the same source
    is reused."""
    from shared.asset_formats import simready, workspace_path
    root = local_root(publisher, name, files(publisher, name, online=False))
    if root is None:
        raise FileNotFoundError(f'{publisher}/{name} 不在本地镜像中，先下载')
    output = Path(output) if output else ROOT / 'assets' / name
    record_path = output / 'ASSET.json'
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if record.get('source') == workspace_path(root):
            return record
        raise FileExistsError(f'{workspace_path(output)} 已是另一个资产（{record.get("source")}）；'
                              f'用 run.py asset simready {publisher}/{name} <其他目录> 包装')
    return simready(root, output)


def prepare(publisher, name, progress=None):
    """Download what is missing, then wrap: the ASSET.json record for usd_object."""
    try:
        download(publisher, name, progress)
    except Offline:
        if local_root(publisher, name, files(publisher, name, online=False)) is None:
            raise
        # Offline with the asset already here: wrapping reports any file still missing.
    return wrap(publisher, name)


def resolve(text, online=True):
    """(publisher, asset) for "publisher/asset" or a bare asset name (local mirror first)."""
    publisher, separator, name = text.strip().strip('/').rpartition('/')
    candidates = [publisher] if separator else list(SOURCES)
    for check_online in ((False, True) if online else (False,)):
        for source in candidates:
            try:
                if name in names(source, check_online):
                    return source, name
            except Offline:
                continue
    known = [f'{source}/{asset}' for source in candidates for asset in names(source, online=False)]
    close = difflib.get_close_matches(text, known, n=5) or difflib.get_close_matches(name, [item.split('/', 1)[1] for item in known], n=5)
    raise KeyError(f'找不到 SimReady 资产 {text!r}' + (f'；相近：{", ".join(close)}' if close else ''))
