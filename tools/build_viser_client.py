"""Build the editor's viser client: viser's own client source (shipped in the installed viser package) with
tools/viser_client/rlinf.patch applied, built into tools/viser_client/viewer.html (the editor page).

Users run the built page and need no Node. Building (only after changing the patch) needs Node >= 24 on the
PATH or under ~/.local/opt/node-v*/bin; the source is copied to .cache/viser_client and its dependencies
come from viser's package-lock, installed there once.

  python tools/build_viser_client.py           # fresh patched copy, npm ci if the lock changed, build
  python tools/build_viser_client.py --edit    # only the patched copy, to edit in .cache/viser_client
  python tools/build_viser_client.py --diff    # write rlinf.patch from that edited copy
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tools' / 'viser_client'
PATCH, OUTPUT, RECORD = HERE / 'rlinf.patch', HERE / 'viewer.html', HERE / 'BUILD.json'
WORK = ROOT / '.cache' / 'viser_client' / 'viser'  # mirrors the package: client/ beside _assets/
CLIENT = WORK / 'client'
NODE_MAJOR = 24


def upstream():
    """The installed viser package directory and its version."""
    import viser
    return Path(viser.__file__).resolve().parent, viser.__version__


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node_path():
    """A PATH with Node >= NODE_MAJOR first."""
    candidates = [Path(p).parent for p in [shutil.which('node')] if p]
    candidates += sorted(Path.home().glob('.local/opt/node-v*/bin'), reverse=True)
    for folder in candidates:
        version = subprocess.run([str(folder / 'node'), '--version'], capture_output=True, text=True).stdout.strip()
        if version and int(version.lstrip('v').split('.')[0]) >= NODE_MAJOR:
            return os.pathsep.join([str(folder), os.environ.get('PATH', '')]), version
    raise SystemExit(f'Node >= {NODE_MAJOR} not found (PATH or ~/.local/opt/node-v*/bin): https://nodejs.org/en/download')


def prepare():
    """A fresh copy of viser's client source with the patch applied (its node_modules kept)."""
    package, version = upstream()
    CLIENT.mkdir(parents=True, exist_ok=True)
    for entry in CLIENT.iterdir():
        if entry.name != 'node_modules':
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    shutil.copytree(package / 'client', CLIENT, ignore=shutil.ignore_patterns('build', 'node_modules'), dirs_exist_ok=True)
    shutil.copytree(package / '_assets', WORK / '_assets', dirs_exist_ok=True)  # the default HDRI it imports
    subprocess.run(['patch', '-p1', '--forward', '--no-backup-if-mismatch', '--quiet', '-d', str(CLIENT), '-i', str(PATCH)],
                   check=True)
    return version


def build():
    version = prepare()
    path, node = node_path()
    env = {**os.environ, 'PATH': path}
    stamp = CLIENT / 'node_modules' / '.rlinf-lock'
    lock = digest(CLIENT / 'package-lock.json')
    if not stamp.is_file() or stamp.read_text() != lock:
        subprocess.run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=CLIENT, env=env, check=True)
        stamp.write_text(lock)
    subprocess.run(['npm', 'run', 'build'], cwd=CLIENT, env=env, check=True)
    shutil.copyfile(CLIENT / 'build' / 'index.html', OUTPUT)
    RECORD.write_text(json.dumps({'viser': version, 'patch_sha256': digest(PATCH), 'node': node}, indent=2) + '\n')
    print(f'{OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KiB) for viser {version}')


def diff():
    """rlinf.patch from the edited copy: paths a/src, b/src, no timestamps."""
    package, _ = upstream()
    done = subprocess.run(['diff', '-ruN', str(package / 'client' / 'src'), str(CLIENT / 'src')], capture_output=True, text=True)
    if done.returncode > 1:
        raise SystemExit(done.stderr)
    text = done.stdout.replace(str(package / 'client') + '/', 'a/').replace(str(CLIENT) + '/', 'b/')
    PATCH.write_text(re.sub(r'^((?:---|\+\+\+) \S+)\t.*$', r'\1', text, flags=re.M))
    print(f'{PATCH.relative_to(ROOT)}: {text.count(chr(10) + "+++ ")} files')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--edit', action='store_true', help='only prepare the patched copy')
    mode.add_argument('--diff', action='store_true', help='write rlinf.patch from the patched copy')
    options = parser.parse_args(argv)
    if options.edit:
        prepare()
        print(f'edit {CLIENT.relative_to(ROOT)}/src, then --diff and build')
    elif options.diff:
        diff()
    else:
        build()


if __name__ == '__main__':
    sys.exit(main())
