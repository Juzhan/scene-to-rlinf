"""Pack, fetch and verify the shared asset bundle (``assets/``), which is distributed outside git.

``assets.lock.json`` in the repository names the archive, its download URL and SHA-256, and the
asset directories it holds. Standard library only, so it runs before any environment exists:

    python3 tools/asset_bundle.py fetch                 # download assets.lock.json "url", verify, extract
    python3 tools/asset_bundle.py fetch --file A.tar.gz # same, from an archive already on disk
    python3 tools/asset_bundle.py verify                # every listed asset directory is present
    python3 tools/asset_bundle.py pack [NAME ...]       # maintainers: archive the listed assets, update the lock

Extraction never overwrites: asset directories that already exist are kept unless --force.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
LOCK = ROOT / 'assets.lock.json'
# Local registrations (library.json holds machine-specific paths) and loose archives stay out.
EXCLUDED = ('library.json', '*.zip', '__pycache__', '.DS_Store')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def direct_url(url):
    """A Google Drive share link becomes its direct download; confirm=t skips the large-file scan page."""
    match = re.search(r'drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:.*&)?id=)([\w-]+)', url)
    return f'https://drive.usercontent.google.com/download?id={match[1]}&export=download&confirm=t' if match else url


def excluded(path):
    return any(path.match(pattern) for pattern in EXCLUDED)


def pack(options):
    previous = json.loads(LOCK.read_text()) if LOCK.is_file() else {}
    directories = sorted(options.directories or previous.get('directories', ()))
    missing = [name for name in directories if not (ASSETS / name).is_dir()]
    if not directories or missing:
        raise SystemExit(f'Name the asset directories to pack (missing: {", ".join(missing) or "none given"})')
    version = options.version or time.strftime('%Y.%m.%d')
    output = (options.output or ROOT / 'outputs/asset_bundle').resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'rlinf-assets-{version}.tar.gz'

    def keep(info):
        return None if excluded(Path(info.name)) else info

    with tarfile.open(archive, 'w:gz') as bundle:
        for name in directories:
            bundle.add(ASSETS / name, arcname=f'assets/{name}', filter=keep)
    lock = {'version': version, 'archive': archive.name,
            'url': options.url if options.url is not None else previous.get('url', ''),
            'sha256': sha256(archive), 'size': archive.stat().st_size, 'directories': directories}
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + '\n')
    print(f'{archive} ({lock["size"] / 2**20:.0f} MiB, {len(directories)} assets)\n'
          f'Upload it, then set "url" in {LOCK.name} (or pass --url when packing).')


def fetch(options):
    lock = json.loads(LOCK.read_text())
    if options.file:
        archive = Path(options.file).resolve()
    else:
        url = options.url or lock.get('url')
        if not url:
            raise SystemExit(f'{LOCK.name} has no "url"; pass --url or --file')
        archive = ROOT / '.cache' / lock['archive']
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file() or sha256(archive) != lock['sha256']:
            url = direct_url(url)
            print(f'Downloading {url}', flush=True)
            partial = archive.with_suffix('.part')
            with urllib.request.urlopen(url) as response:
                if response.headers.get_content_type() == 'text/html':
                    raise SystemExit(f'{url} returned a web page, not the archive (is the link shared publicly?); '
                                     'download it in a browser and pass --file')
                with open(partial, 'wb') as file:
                    shutil.copyfileobj(response, file, 1 << 20)
            partial.replace(archive)
    digest = sha256(archive)
    if digest != lock['sha256']:
        raise SystemExit(f'{archive}: SHA-256 {digest} does not match {LOCK.name} ({lock["sha256"]})')
    ASSETS.mkdir(exist_ok=True)
    present = {name for name in lock['directories'] if (ASSETS / name).exists()}
    replace = present if options.force else set()
    with tempfile.TemporaryDirectory(dir=ROOT) as staging, tarfile.open(archive, 'r:gz') as bundle:
        bundle.extractall(staging, filter='data')
        for name in lock['directories']:
            if name in present and name not in replace:
                print(f'kept existing assets/{name}')
                continue
            if name in replace:
                shutil.rmtree(ASSETS / name)
            shutil.move(Path(staging) / 'assets' / name, ASSETS / name)
            print(f'installed assets/{name}')
    return verify(options)


def verify(options):
    lock = json.loads(LOCK.read_text())
    missing = [name for name in lock['directories'] if not (ASSETS / name).is_dir()]
    if missing:
        raise SystemExit(f'Missing assets: {", ".join(missing)}; run python3 tools/asset_bundle.py fetch')
    print(f'assets {lock["version"]}: {len(lock["directories"])} directories present')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    packing = commands.add_parser('pack', help='archive assets/ and rewrite assets.lock.json')
    packing.add_argument('directories', nargs='*', help='asset directories to include (default: those in the lock)')
    packing.add_argument('--version', help='bundle version (default: today, YYYY.MM.DD)')
    packing.add_argument('--output', type=Path, help='archive directory (default outputs/asset_bundle)')
    packing.add_argument('--url', help='download URL to record in the lock')
    fetching = commands.add_parser('fetch', help='download, verify and extract the bundle')
    fetching.add_argument('--url', help='override the lock URL')
    fetching.add_argument('--file', help='use a local archive instead of downloading')
    fetching.add_argument('--force', action='store_true', help='replace asset directories that already exist')
    commands.add_parser('verify', help='check every asset directory in the lock is present')
    options = parser.parse_args(argv)
    return {'pack': pack, 'fetch': fetch, 'verify': verify}[options.command](options)


if __name__ == '__main__':
    sys.exit(main())
