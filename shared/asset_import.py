"""Assets a user brings in: files in common formats become an asset directory the scene can use.

The editor uploads the files (one file, a folder or a zip) to ``assets/imported/<name>/source/`` and calls
:func:`import_asset`; the simulator always gets USD, the original files stay beside it:

    robot  URDF / MJCF   -> run.py asset convert (Isaac importer for USD, MuJoCo compiler for MJCF);
                            a ROS .urdf.xacro is expanded first (expand_xacro), its packages found by package.xml
    any    USD / USDZ    -> used as it is (a scene: wrapped as a static environment, colliders added); links
                            without mass (PhysX virtual joints) get a negligible one in a wrapper layer
    any    mesh files    -> USD written here: trimesh reads OBJ, STL, PLY, glTF/GLB, OFF, DAE, 3MF;
                            Blender (when installed) reads FBX and whatever trimesh cannot

A mesh becomes one rigid body with convex-hull colliders per part and a mass estimated from those hulls
(an object), or static triangle colliders (a scene). glTF is Y-up by definition and OBJ/FBX exports usually are: they are turned Z-up. Units cannot be
known from a mesh file; the bounds are reported and the entity's scale fixes millimetre/centimetre models.

    run.py asset import <file or directory> [--kind object|robot|scene] [--name NAME] [--project PROJECT]
                        [--main FILE] [--xacro "dof:=7 add_gripper:=true"]
"""
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A user's imports belong to their project: projects/<name>/assets/<asset>/, listed on that project's shelf by
# projects/<name>/assets/library.json. The workspace's own assets/imported/ and assets/library.json are the
# maintainers' additions, offered to every project.
IMPORTED = ROOT / 'assets' / 'imported'
LIBRARY = ROOT / 'assets' / 'library.json'


def project_assets(project):
    """(directory of imports, shelf library) of ``project`` (its directory, or a name under projects/)."""
    directory = Path(project) if Path(project).is_dir() else ROOT / 'projects' / str(project)
    if not (directory / 'scene.yaml').is_file():
        raise ValueError(f'没有项目 {project}（找不到 scene.yaml）')
    return directory / 'assets', directory / 'assets' / 'library.json'
FLT_MAX = 3.4028234663852886e+38  # USD's default drive maxForce: no limit
KINDS = ('object', 'robot', 'scene')
USD_SUFFIXES = {'.usd', '.usda', '.usdc', '.usdz'}
TRIMESH_SUFFIXES = {'.obj', '.stl', '.ply', '.glb', '.gltf', '.off', '.dae', '.3mf'}
BLENDER_SUFFIXES = {'.fbx', '.blend', '.abc'}
MESH_SUFFIXES = TRIMESH_SUFFIXES | BLENDER_SUFFIXES
Y_UP_SUFFIXES = {'.glb', '.gltf', '.obj'}  # Blender turns FBX Z-up itself
DENSITY = 500.0  # kg/m^3 over the convex hulls of an imported mesh object, when no mass is given (wood, plastic)
SUFFIXES = USD_SUFFIXES | MESH_SUFFIXES | {'.urdf', '.xacro', '.xml', '.mjcf', '.zip'}


def identifier(text):
    name = re.sub(r'\W+', '_', Path(str(text)).stem if '.' in str(text) else str(text)).strip('_') or 'asset'
    return 'asset_' + name if name[0].isdigit() else name


def new_directory(name, root=None):
    """A fresh ``<root>/<name>`` (``name_2`` ... when taken): a project's assets/, else the workspace's imports."""
    root = Path(root or IMPORTED)
    base, index = identifier(name), 2
    candidate = base
    while (root / candidate).exists():
        candidate, index = f'{base}_{index}', index + 1
    (root / candidate / 'source').mkdir(parents=True)
    return root / candidate


def safe_parts(relative):
    """The folders and name of an uploaded path, kept inside the upload: no '..', no root, '\\' read as '/'."""
    return [part for part in Path(str(relative).replace('\\', '/')).parts if part not in ('', '.', '..', '/')]


def store(directory, relative, stream, size):
    """Write one uploaded file below ``directory/source`` (its relative path kept, nothing outside)."""
    parts = safe_parts(relative)
    if not parts:
        raise ValueError('空文件名')
    target = (directory / 'source').joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as out:
        remaining = size
        while remaining > 0:
            chunk = stream.read(min(remaining, 1 << 20))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)
    if target.suffix.lower() == '.zip':
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                inner = safe_parts(member.filename)
                if member.is_dir() or not inner or inner[0] == '__MACOSX':
                    continue
                path = target.parent.joinpath(*inner)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, path.open('wb') as out:
                    shutil.copyfileobj(source, out)
        target.unlink()
    return target


def is_mjcf(path):
    try:
        with open(path, 'rb') as stream:
            return b'<mujoco' in stream.read(4096)
    except OSError:
        return False


def missing_files(description):
    """How many files a URDF / MJCF names (meshes, textures) that are not beside it (a xacro names them through
    its arguments and packages: 0)."""
    if str(description).lower().endswith('.xacro'):
        return 0
    try:
        text = Path(description).read_text(errors='replace')
    except OSError:
        return 0
    names = set(re.findall(r'(?:filename|file)="([^"]+)"', text))
    # MJCF looks for meshes and textures under <compiler meshdir / texturedir / assetdir>
    folders = [Path(description).parent / folder for folder in
               re.findall(r'<compiler\b[^>]*?\b(?:meshdir|texturedir|assetdir)="([^"]*)"', text)]
    folders.insert(0, Path(description).parent)
    return sum(not any((folder / re.sub(r'^package://[^/]+/', '', name)).is_file() for folder in folders)
               for name in names if '://' not in re.sub(r'^package://[^/]+/', '', name))


def find_in_upload(name, base, root):
    """The uploaded file a URDF mesh reference means: ``package://<pkg>/<path>`` through the package.xml naming
    it, else ``<path>`` below the URDF's folder or its parents within the upload, else the one file whose path ends
    the same way (a description copied out of its ROS workspace, its packages flattened into one folder)."""
    from .asset_formats import find_package
    package, relative = None, name
    if name.startswith('package://'):
        package, _, relative = name.removeprefix('package://').partition('/')
        try:
            found = find_package(package, base)
            if (found / relative).is_file():
                return found / relative
        except FileNotFoundError:
            pass
    elif name.startswith('file://'):
        relative = name.removeprefix('file://')
        if Path(relative).is_file():
            return Path(relative)
    relative = Path(relative.lstrip('/'))
    for folder in [base, *base.parents]:
        for candidate in (folder / relative, *([folder / package / relative] if package else [])):
            if candidate.is_file():
                return candidate
        if folder == root:
            break
    tail = [part.lower() for part in relative.parts]
    best, size = [], 0
    for file in root.rglob('*'):
        parts = [part.lower() for part in file.relative_to(root).parts]
        common = next((n for n in range(min(len(parts), len(tail)), 0, -1) if parts[-n:] == tail[-n:]), 0)
        if common > size and file.is_file():
            best, size = [file], common
        elif common == size and common and file.is_file():
            best.append(file)
    return best[0] if len(best) == 1 else None


def _resolves(name, urdf):
    """Whether the converters find ``name`` themselves (a path beside the URDF, a package with its package.xml)."""
    from .asset_formats import resolve_mesh
    try:
        return resolve_mesh(name, urdf, {}).is_file()
    except FileNotFoundError:
        return False


def prepare_urdf(source, root, target, base=None, files=None):
    """A URDF whose mesh references all point at uploaded files, written to ``target`` when the source needs one;
    returns what changed ({} when the source converts as it is). References are found by find_in_upload from
    ``base`` (the URDF's folder; for an expanded xacro, the xacro's folder in the upload); a collision mesh missing
    from the upload takes the link's visual geometry (the importer then uses its convex hull). A missing visual, or
    a missing collision without a visual, stops the import: the robot would lack parts (raises ImportFailed).
    ``files`` (a set) collects the uploaded files the URDF uses."""
    from xml.etree import ElementTree
    source, root = Path(source).resolve(), Path(root).resolve()
    base = Path(base).resolve() if base else source.parent
    tree = ElementTree.parse(source)
    found, missing, from_visual, lost = {}, [], [], []
    for link in tree.getroot().iter('link'):
        shown = []  # the link's visuals that are there, in case a collision mesh is not
        for element in [*link.findall('visual'), *link.findall('collision')]:
            mesh = element.find('geometry/mesh')
            path = mesh is not None and find_in_upload(mesh.get('filename', ''), base, root)
            if mesh is None or path:
                if path and files is not None:
                    files.add(path)
                if path:
                    if not _resolves(mesh.get('filename'), source):
                        found[mesh.get('filename')] = path
                    mesh.set('filename', Path(os.path.relpath(path, Path(target).parent)).as_posix())
                if element.tag == 'visual':
                    shown.append(element)
                continue
            missing.append(mesh.get('filename', ''))
            if element.tag == 'collision' and shown:
                for part in ('origin', 'geometry'):
                    if element.find(part) is not None:
                        element.remove(element.find(part))
                    if shown[0].find(part) is not None:
                        element.append(ElementTree.fromstring(ElementTree.tostring(shown[0].find(part))))
                from_visual.append(link.get('name'))
            else:
                lost.append(mesh.get('filename', ''))
    if lost:
        names = sorted(set(lost))
        raise ImportFailed(f'URDF 用到的 {len(names)} 个网格文件不在导入的文件里（如 {names[0]}）：'
                           '请导入 URDF 所在的整个文件夹（导入 → 选择文件夹，或打包成 zip），网格要一起带上',
                           '找不到的网格：\n' + '\n'.join(names))
    balanced = balance_inertia(tree.getroot())
    if not (found or missing or balanced) and source.parent == base:
        return {}
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding='utf-8', xml_declaration=True)
    return {'urdf': str(target), 'meshes_found': len(found), 'missing_meshes': sorted(set(missing)),
            'collision_from_visual': sorted(set(from_visual)), 'inertia_balanced': balanced}


def balance_inertia(robot):
    """Links whose inertia no body can have (principal moments breaking the triangle inequality, as CAD exports
    sometimes write) get the mean of their moments on every axis, in place; returns their names. The MuJoCo compiler
    does the same itself (balanceinertia); the Isaac importer takes the values as they are and Newton diverges."""
    import numpy as np
    changed = []
    for link in robot.iter('link'):
        inertia = link.find('inertial/inertia')
        if inertia is None:
            continue
        value = {key: float(inertia.get(key, 0)) for key in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')}
        moments = np.sort(np.linalg.eigvalsh(np.array([[value['ixx'], value['ixy'], value['ixz']],
                                                       [value['ixy'], value['iyy'], value['iyz']],
                                                       [value['ixz'], value['iyz'], value['izz']]])))
        if moments[0] + moments[1] >= moments[2] * (1 - 1e-6):
            continue
        mean = f'{moments.mean():.9g}'
        inertia.attrib.update(ixx=mean, iyy=mean, izz=mean, ixy='0', ixz='0', iyz='0')
        changed.append(link.get('name'))
    return changed


def mimic_joints(urdf):
    """{follower: {leader, ratio, offset}} from a URDF's ``<mimic>`` tags (follower = multiplier * leader + offset),
    the form of usd_object ``couplings``. A follower of a follower is not supported there and is left out."""
    from xml.etree import ElementTree
    found = {}
    for joint in ElementTree.parse(urdf).getroot().iter('joint'):
        mimic = joint.find('mimic')
        if mimic is not None and mimic.get('joint'):
            found[joint.get('name')] = {'leader': mimic.get('joint'), 'ratio': float(mimic.get('multiplier', 1)),
                                        'offset': float(mimic.get('offset', 0))}
    return {name: value for name, value in found.items() if value['leader'] not in found}


def expand_xacro(path, root, target, arguments=''):
    """Expand a ROS ``.urdf.xacro`` from the upload into the URDF ``target`` with xacro (in the Isaac interpreter,
    which has it), ``arguments`` as on the command line (``dof:=7 add_gripper:=true``; the file's defaults
    otherwise). ``$(find <pkg>)`` is the uploaded folder whose package.xml names <pkg>: the xacro files are
    expanded from a scratch copy that says so, without ROS installed. Mesh references stay ``package://`` and are
    found by prepare_urdf. Returns ({argument: value} given, packages found); raises ImportFailed with xacro's
    message."""
    import tempfile
    from xml.etree import ElementTree
    path, root = Path(path).resolve(), Path(root).resolve()
    given = dict(item.split(':=', 1) for item in str(arguments or '').split() if ':=' in item)
    if len(given) != len(str(arguments or '').split()):
        raise ImportFailed(f'xacro 参数写成 名字:=值，用空格分开（收到 {arguments!r}）')
    packages = {}
    for manifest in root.rglob('package.xml'):
        with contextlib.suppress(ElementTree.ParseError):
            packages.setdefault(ElementTree.parse(manifest).findtext('name'), manifest.parent)
    from shared.assembly.runtimes import declared_python
    python = str(declared_python('isaac', ROOT))  # RLINF_PYTHON, then .venv (Isaac Sim)
    with tempfile.TemporaryDirectory(prefix='xacro_') as scratch:
        scratch = Path(scratch)
        located = {name: scratch / folder.relative_to(root) for name, folder in packages.items()}

        def find(match):
            return str(located.get(match.group(1), scratch / '.missing_package' / match.group(1)))
        for file in root.rglob('*'):  # the text files xacro reads; meshes stay in the upload
            if file.is_file() and file.suffix.lower() in ('.xacro', '.urdf', '.yaml', '.yml', '.xml', '.json'):
                copy = scratch / file.relative_to(root)
                copy.parent.mkdir(parents=True, exist_ok=True)
                text = file.read_text(errors='replace')
                copy.write_text(re.sub(r'\$\(find\s+([\w-]+)\)', find, text) if file.suffix.lower() == '.xacro' else text)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        command = [python, '-c', 'import sys, xacro; sys.argv[0] = "xacro"; xacro.main()',
                   str(scratch / path.relative_to(root)), '-o', str(target), *(f'{k}:={v}' for k, v in given.items())]
        result = subprocess.run(command, cwd=scratch, capture_output=True, text=True)
    if result.returncode or not Path(target).is_file():
        output = (result.stderr or result.stdout).strip()
        absent = sorted(set(re.findall(r'\.missing_package/([\w-]+)', output)))
        reason = (f'找不到 ROS 包 {"、".join(absent)}（导入的文件里没有写着这个名字的 package.xml）：请导入包含它的整个包文件夹'
                  if absent else (output.splitlines() or ['xacro 没有输出'])[-1].replace(str(scratch) + '/', ''))
        raise ImportFailed(f'xacro 展开失败：{reason}', output.replace(str(scratch) + '/', ''))
    return given, {name: str(folder.relative_to(root)) for name, folder in packages.items()}


def check_links(urdf, chosen):
    """A robot has links: a xacro that only defines macros (lite6.urdf.xacro, used by xarm_device.urdf.xacro) expands
    to an empty <robot/>, on which the Isaac importer aborts (out of memory). Raises ImportFailed saying so."""
    from xml.etree import ElementTree
    if any(True for _ in ElementTree.parse(urdf).getroot().iter('link')):
        return
    if str(chosen).lower().endswith('.xacro'):
        raise ImportFailed(f'{Path(chosen).name} 只定义了宏，展开后没有连杆：请选用这些宏的顶层文件（如 ROS 包 urdf/ 下的 '
                           '*_device.urdf.xacro），用 xacro 参数选型号')
    raise ImportFailed(f'{Path(chosen).name} 里没有连杆（<link>），不是可导入的机器人')


def is_urdf(path):
    return str(path).lower().endswith(('.urdf', '.urdf.xacro', '.xacro'))


# A folder chosen in the page stays on the user's machine: the page sends these small text files first (what a
# description and its packages are written in), the editor works out which of the other files it uses
# (needed_files) and only those are sent. A ROS repository or a model zoo can be chosen whole.
TEXT_SUFFIXES = ('.urdf', '.xacro', '.xml', '.mjcf', '.yaml', '.yml', '.mtl', '.gltf')
TEXT_LIMIT = 2 << 20  # bytes: a larger "text" file is data, sent only when used


def needed_files(root, main, xacro=''):
    """The files below ``root`` that ``main`` (a path inside it) uses, as paths relative to ``root``: a URDF's or an
    expanded xacro's meshes, an MJCF's includes, meshes and textures, an OBJ's materials and their textures, a glTF's
    buffers and images; a USD with everything in its folder (its layers and textures may be anywhere there). The
    other files below ``root`` only need to exist (an empty stand-in is enough). Raises ImportFailed like the import
    would (a URDF missing a mesh, a xacro naming an absent package), before anything large is sent."""
    import tempfile
    from xml.etree import ElementTree
    root = Path(root).resolve()
    main = (root / main).resolve()
    if not main.is_file() or not main.is_relative_to(root):
        raise ValueError(f'没有文件 {main}')
    used, name = {main}, main.name.lower()
    if name.endswith(('.urdf', '.xacro')):
        with tempfile.TemporaryDirectory(prefix='needs_') as scratch:
            urdf = main
            if name.endswith('.xacro'):
                urdf = Path(scratch) / 'expanded.urdf'
                expand_xacro(main, root, urdf, xacro)
            check_links(urdf, main)
            prepare_urdf(urdf, root, Path(scratch) / 'prepared.urdf', main.parent, files=used)
    elif name.endswith(('.xml', '.mjcf')) and is_mjcf(main):
        from .asset_formats import mjcf_files
        used |= {path for path in mjcf_files(main)[0] if path.is_relative_to(root)}
    elif name.endswith('.obj'):
        for line in main.read_text(errors='replace').splitlines():
            if line.startswith('mtllib '):
                material = (main.parent / line.split(maxsplit=1)[1].strip()).resolve()
                if material.is_file():
                    used.add(material)
                    for entry in material.read_text(errors='replace').splitlines():
                        words = entry.split()
                        if words and (words[0].startswith('map_') or words[0] in ('bump', 'disp', 'decal', 'norm')):
                            texture = (material.parent / words[-1]).resolve()
                            if texture.is_file():
                                used.add(texture)
    elif name.endswith('.gltf'):
        document = json.loads(main.read_text(errors='replace'))
        for item in [*document.get('buffers', []), *document.get('images', [])]:
            uri = item.get('uri', '')
            if uri and not uri.startswith('data:') and (main.parent / uri).is_file():
                used.add((main.parent / uri).resolve())
    elif main.suffix.lower() in USD_SUFFIXES:
        used |= {path for path in main.parent.rglob('*') if path.is_file()}
    return sorted(path.relative_to(root).as_posix() for path in used if path.is_relative_to(root))


def main_files(source):
    """What the import is made from: a robot description, else USD, else every mesh (several make one asset)."""
    files = sorted(p for p in Path(source).rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.parts))
    for test in (lambda p: p.suffix.lower() == '.urdf', lambda p: p.name.lower().endswith('.urdf.xacro'),
                 lambda p: p.suffix.lower() in ('.xml', '.mjcf') and is_mjcf(p)):
        found = [p for p in files if test(p)]
        if found:
            # The main description: one whose meshes are all there, not a wrapper including another one (MuJoCo
            # Menagerie's scene.xml adds a floor and lights to ur5e.xml), the shallowest, then the plainest name
            # (r1pro.urdf beside r1pro_original.urdf)
            included = {(p.parent / name).resolve() for p in found
                        for name in re.findall(r'<include\s+file="([^"]+)"', p.read_text(errors='replace'))}
            return 'description', [min(found, key=lambda p: (missing_files(p), p.resolve() not in included, len(p.parts),
                                                             len(p.stem), -p.stat().st_size))]
    found = [p for p in files if p.suffix.lower() in USD_SUFFIXES]
    if found:  # the top layer: the shallowest, then the largest (sublayers and payloads sit deeper or are smaller)
        return 'usd', [max(found, key=lambda p: (-len(p.parts), p.stat().st_size))]
    found = [p for p in files if p.suffix.lower() in MESH_SUFFIXES]
    if found:
        return 'mesh', found
    raise ValueError('没有可导入的文件：支持 URDF、MJCF、USD/USDZ，以及 OBJ、STL、PLY、glTF/GLB、FBX、DAE、OFF、3MF（可打包为 zip）')


class ImportFailed(RuntimeError):
    """An import that did not finish: the message says why in a line, ``detail`` is the tool's own output."""
    def __init__(self, message, detail=''):
        super().__init__(message)
        self.detail = detail


def import_asset(directory, kind='object', name=None, progress=print, main=None, xacro='', started=None):
    """Turn ``directory/source`` into an asset: ``directory/ASSET.json`` with ``formats.usd`` (workspace path).
    ``main`` (a path inside source/) picks the description when a folder holds several variants; ``xacro`` gives
    a ``.urdf.xacro`` its arguments (expand_xacro). ``started`` gets the conversion process (its own session: the
    editor's cancel ends it with the Isaac importer it starts), then None once it has ended. A fixed base
    is the entity's ``fix_root_link``, not part of the asset: the Isaac MJCF importer's own fixed base adds a
    second articulation root (``worldBody``) that Isaac Lab cannot fix."""
    from .asset_formats import workspace_path
    directory = Path(directory).resolve()
    if kind not in KINDS:
        raise ValueError(f'未知的类型 {kind}')
    name = identifier(name or directory.name)
    form, files = main_files(directory / 'source')
    if main:
        chosen = (directory / 'source' / main).resolve()
        if not chosen.is_file() or not chosen.is_relative_to(directory / 'source'):
            raise ValueError(f'没有文件 {main}')
        form = 'description' if chosen.suffix.lower() in ('.urdf', '.xml', '.mjcf', '.xacro') else \
            'usd' if chosen.suffix.lower() in USD_SUFFIXES else 'mesh'
        files = [chosen]
    record = {'source': [workspace_path(p) for p in files], 'kind': kind}
    if form == 'description':
        if kind == 'scene':
            raise ValueError('URDF / MJCF 按机器人或物体导入')
        description, base = files[0], files[0].parent
        if description.name.lower().endswith('.xacro'):
            progress(f'展开 {description.name}（xacro）…')
            expanded = directory / 'urdf' / description.name[:-len('.xacro')]
            arguments, packages = expand_xacro(description, directory / 'source', expanded, xacro)
            record['xacro'] = {'arguments': arguments, 'packages': packages, 'urdf': workspace_path(expanded)}
            description = expanded
        if description.suffix.lower() == '.urdf':
            check_links(description, files[0])
            progress('检查 URDF 用到的网格…')
            prepared = prepare_urdf(description, directory / 'source', directory / 'urdf' / description.name, base)
            if prepared:
                description = Path(prepared.pop('urdf'))
                record['urdf_prepared'] = {**prepared, 'urdf': workspace_path(description)}
        progress(f'转换 {files[0].name}：Isaac 导入器写 USD，MuJoCo 写 MJCF，再核对两者（首次一两分钟）…')
        log = directory / 'convert.log'
        command = [sys.executable, str(ROOT / 'run.py'), 'asset', 'convert', str(description), str(directory), '--name', name]
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            if started:
                started(process)
            code = process.wait()
            if started:
                started(None)  # nothing left to cancel
            if code < 0:  # ended by a signal: cancelled
                raise ImportFailed('已取消', log.read_text(errors='replace')[-6000:])
        # Written USD is enough to use the robot; an imperfect USD/MJCF alignment is recorded, not fatal.
        converted = json.loads((directory / 'ASSET.json').read_text()) if (directory / 'ASSET.json').is_file() else {}
        if 'usd' not in converted.get('formats', {}):
            lines = [line for line in log.read_text(errors='replace').splitlines() if line.strip()]
            # The message: the exception's line and what follows it (a parser adds where it stopped)
            last = max((i for i, line in enumerate(lines) if re.match(r'[\w.]*(Error|Exception)\w*: ', line)), default=len(lines) - 1)
            reason = ' '.join(line.strip() for line in lines[last:])[:300] if lines else '没有输出'
            raise ImportFailed(f'转换失败：{reason}', '\n'.join(lines[-80:]))
        alignment = converted.get('alignment') or {}
        couplings = mimic_joints(description)
        if couplings:  # the Isaac importer drops URDF <mimic>; the entity's couplings put the linkage back
            record['couplings'] = couplings
        record.update(tool='run.py asset convert', usd=converted['formats']['usd'], joint_order=converted.get('joint_order'),
                      aligned=alignment.get('aligned'), deactivated=drop_unresolved(ROOT / converted['formats']['usd']))
        wrapped = wrap_for_isaac(ROOT / record['usd'], directory / 'usd' / f'{name}_isaac.usda')
        if wrapped:
            record.update(usd=workspace_path(directory / 'usd' / f'{name}_isaac.usda'), **wrapped)
    elif form == 'usd':
        if kind == 'scene':
            from .asset_formats import environment
            progress(f'包装场景 {files[0].name}（补碰撞体）…')
            wrapped = environment(files[0], directory, name, add_colliders=True, expand_instancers=True)
            record.update(tool='shared.asset_formats.environment', usd=wrapped['formats']['usd'])
        else:
            wrapped = wrap_for_isaac(files[0], directory / 'usd' / f'{name}.usda')
            record.update(tool='shared.asset_import.wrap_for_isaac' if wrapped else None,
                          usd=workspace_path(directory / 'usd' / f'{name}.usda') if wrapped else workspace_path(files[0]),
                          **wrapped)
    else:
        progress(f'把 {len(files)} 个网格写成 USD…')
        target = directory / 'usd' / f'{name}.usda'
        report = mesh_to_usd(files, target, static=kind == 'scene')
        record.update(tool='shared.asset_import.mesh_to_usd', usd=workspace_path(target), **report)
    if kind == 'robot':
        # A URDF gives no drive gains: the stiffness the Isaac importer writes is its own (625, undamped on the
        # fingers, which a light link cannot integrate), so every joint gets the default PD as an undriven one.
        record['undriven_joints'] = list(joint_stiffness(ROOT / record['usd'])) if is_urdf(files[0]) \
            else undriven_joints(ROOT / record['usd'])
    (directory / 'IMPORT.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def library_entry(directory, record, label=None, fixed=False):
    """The asset shelf entry of an import (a usd_object preset): parameters, category and a note on what the
    import found (units, size, estimated mass, joints given default gains)."""
    kind = record['kind']
    parameters = {'usd_path': record['usd'], **({'fix_root_link': True} if fixed and kind != 'scene' else {})}
    if record.get('couplings'):
        parameters['couplings'] = record['couplings']
    if record.get('undriven_joints'):  # the USD drives none of them: default PD gains, tuned in 高级
        parameters['actuators'] = DEFAULT_ACTUATORS
    hint = {'mm': '尺寸像是毫米', 'cm': '尺寸像是厘米'}.get(record.get('units_hint'))
    prepared = record.get('urdf_prepared') or {}
    note = f'导入自 {Path(record["source"][0]).name}' \
        + (f'（{hint}：在变换里把缩放设为 {0.001 if hint.endswith("毫米") else 0.01}）' if hint else '') \
        + (f'；外形 {" × ".join(f"{v:.2f}" for v in record["size"])} m' if record.get('size') else '') \
        + (f'；质量按外形估算 {record["mass"]} kg，可在“高级”里改' if record.get('mass') else '') \
        + (f'；URDF 里 {len(prepared["missing_meshes"])} 个网格文件不在上传的文件中'
           + (f'，{"、".join(prepared["collision_from_visual"])} 的碰撞改用外观网格' if prepared['collision_from_visual'] else '')
           if prepared.get('missing_meshes') else '') \
        + (f'；{"、".join(prepared["inertia_balanced"])} 的惯量在物理上不可能（主惯量不满足三角不等式），'
           '已按 MuJoCo 的做法取三个主惯量的平均值' if prepared.get('inertia_balanced') else '') \
        + ('；USD 与 MJCF 的对齐检查没有全部通过，差异记在 ASSET.json 的 alignment' if record.get('aligned') is False else '') \
        + (f'；{"URDF 不含驱动增益，" if is_urdf(record["source"][0]) else ""}'
           f'{len(record["undriven_joints"])} 个关节{"" if is_urdf(record["source"][0]) else "没有驱动增益，"}已设默认 PD（刚度 100、阻尼 10），'
           '请按机器人在“高级”的 actuators 里调整' if record.get('undriven_joints') else '')
    return {'id': Path(directory).name, 'label': label or Path(directory).name, 'factory': 'usd_object',
            'parameters': parameters, 'category': 'terrain' if kind == 'scene' else kind, 'note': note}


def register(entry, library=None):
    """Put ``entry`` on the asset shelf (``library``, else the workspace's assets/library.json), replacing one
    with its id."""
    library = Path(library or LIBRARY)
    try:
        entries = json.loads(library.read_text()).get('assets', [])
    except (OSError, ValueError):
        entries = []
    entries = [item for item in entries if item['id'] != entry['id']] + [entry]
    library.parent.mkdir(parents=True, exist_ok=True)
    temporary = library.with_suffix('.tmp')
    temporary.write_text(json.dumps({'assets': entries}, ensure_ascii=False, indent=1) + '\n')
    os.replace(temporary, library)


# A robot whose USD authors no drive stiffness gets these PD gains (usd_object actuators), so its joints hold
# the policy's targets (or the default pose) instead of hanging limp; tune them per robot in its parameters.
DEFAULT_ACTUATORS = {'.*': {'stiffness': 100.0, 'damping': 10.0}}


def joint_stiffness(usd):
    """{movable joint of ``usd``: its drive stiffness (0 without a drive)}."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(usd))
    stiffness = {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        kind = 'angular' if prim.IsA(UsdPhysics.RevoluteJoint) else 'linear' if prim.IsA(UsdPhysics.PrismaticJoint) else None
        if kind is None or not prim.IsActive():
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, kind)
        stiffness[prim.GetName()] = (drive.GetStiffnessAttr().Get() if drive else None) or 0.0
    return stiffness


def undriven_joints(usd):
    """Movable joints of ``usd`` whose drive authors no stiffness."""
    return [name for name, value in joint_stiffness(usd).items() if not value]


def driven(usd):
    """Whether ``usd`` is a robot by its joints: some drive holds a position (stiffness > 0). A door or a drawer
    has none, at most a damping (LIBERO's microwave)."""
    return any(joint_stiffness(usd).values())


def drop_unresolved(path):
    """Remove references of ``path`` that point at nothing (the Isaac URDF importer references visuals of
    frame-only links that it never writes); Newton refuses a stage with composition errors. Only layers of
    this import are edited; returns the prims whose references were dropped."""
    from pxr import Sdf, Usd
    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path))
    broken = sorted({str(error.rootSite.path) for error in stage.GetCompositionErrors()
                     if str(error.errorType).endswith('UnresolvedPrimPath')})
    changed = set()
    for prim_path in broken:
        for spec in stage.GetPrimAtPath(prim_path).GetPrimStack():
            layer = spec.layer
            if not Path(layer.realPath or '').resolve().is_relative_to(path.parent.parent):
                continue
            items = list(spec.referenceList.GetAddedOrExplicitItems())
            kept = []
            for reference in items:
                target = Sdf.Layer.FindOrOpenRelativeToLayer(layer, reference.assetPath) if reference.assetPath else layer
                if target is not None and target.GetPrimAtPath(reference.primPath or target.defaultPrim and '/' + target.defaultPrim):
                    kept.append(reference)
            if len(kept) != len(items):
                spec.referenceList.ClearEditsAndMakeExplicit()
                spec.referenceList.explicitItems = kept
                changed.add(layer)
    for layer in changed:
        layer.Save()
    return broken


TINY_MASS, TINY_INERTIA = 1e-3, 1e-7  # kg, kg m^2: a virtual link that carries no weight


def wrap_for_isaac(source, target):
    """What an imported USD needs before Isaac Lab can place and simulate it, written to ``target``: a layer
    referencing ``source`` unchanged with overrides, only when something is needed. Returns what changed
    ({} when the source is used as it is):

    - ``massless_bodies``: links without mass or colliders get a negligible mass. Robots made for PhysX
      may have them (OmniGibson's virtual base joints x / y / z / roll / pitch / yaw); PhysX tolerates
      them, MuJoCo-based engines (Newton) do not;
    - ``stray_roots``: an articulation root on a prim that is no rigid body, beside one that is, is removed.
      The Isaac MJCF importer adds an empty ``worldBody`` root for a model welded to the world; Isaac Lab
      takes it as the root and cannot fix it;
    - ``world_joints``: joints holding the root body to the world are deactivated, so whether the base is
      fixed is the entity's ``fix_root_link`` alone (the importer welds an MJCF body without a free joint;
      Isaac Lab's own fixed joint beside it cannot be merged in Newton);
    - ``rerooted``: the root body takes the asset frame. Isaac Lab writes the scene pose to the root body,
      so a root turned or moved within the asset (MuJoCo Menagerie's UR5e base is yawed 180 degrees) would
      start turned or moved from where the editor shows it (``asset_formats._reroot``, as ``run.py asset
      simready`` does)."""
    from pxr import Sdf, Usd, UsdPhysics
    from .asset_formats import _reroot, usd_root_offset
    stage = Usd.Stage.Open(str(source))
    top = stage.GetDefaultPrim()
    if not top:
        return {}
    light = []
    for prim in Usd.PrimRange(top):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        # colliders may sit in instances (the Isaac importers make collision meshes instanceable)
        solid = any(child.HasAPI(UsdPhysics.CollisionAPI)
                    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if child != prim)
        if (mass is None or mass <= 0) and not solid:
            light.append(prim.GetPath())
    roots = [prim for prim in Usd.PrimRange(top) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    stray = [prim.GetPath() for prim in roots if not prim.HasAPI(UsdPhysics.RigidBodyAPI)] \
        if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in roots) else []
    bodies = {prim.GetPath() for prim in roots if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    welded = []
    for prim in Usd.PrimRange(top):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            ends = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
            if [] in ends and any(path in bodies for targets in ends for path in targets):
                welded.append(prim.GetPath())
    # A drive allowed no force: the Isaac MJCF importer writes a passive joint's damping as a drive with
    # maxForce 0. PhysX then applies none of it; Newton's MuJoCo solver rejects the empty force range.
    powerless = [(prim.GetPath(), name) for prim in Usd.PrimRange(top) if prim.IsA(UsdPhysics.Joint)
                 for name in ('angular', 'linear', 'rotX', 'rotY', 'rotZ', 'transX', 'transY', 'transZ')
                 if prim.HasAPI(UsdPhysics.DriveAPI, name)
                 and (UsdPhysics.DriveAPI(prim, name).GetMaxForceAttr().Get() or 0.0) <= 0.0
                 and UsdPhysics.DriveAPI(prim, name).GetMaxForceAttr().HasAuthoredValue()]
    # MJCF sites (reference points and regions, drawn in MuJoCo only when their rgba says so) become plain
    # visible shapes under a ``sites`` scope in the Isaac MJCF importer: LIBERO's red heating-region box.
    from pxr import UsdGeom
    sites = [prim.GetPath() for prim in Usd.PrimRange(top, Usd.TraverseInstanceProxies())
             if prim.GetName() == 'sites' and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible]
    turned = usd_root_offset(source) is not None
    if not light and not turned and not stray and not welded and not powerless and not sites:
        return {}
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(target))
    layer.defaultPrim = top.GetName()
    spec = Sdf.CreatePrimInLayer(layer, top.GetPath())
    spec.specifier, spec.typeName = Sdf.SpecifierDef, top.GetTypeName() or 'Xform'
    spec.referenceList.Prepend(Sdf.Reference(os.path.relpath(Path(source).resolve(), target.parent), top.GetPath()))
    over = Usd.Stage.Open(layer)
    for path in light:
        mass = UsdPhysics.MassAPI.Apply(over.OverridePrim(path))
        mass.CreateMassAttr(TINY_MASS)
        mass.CreateDiagonalInertiaAttr((TINY_INERTIA,) * 3)
    for path in stray:
        over.OverridePrim(path).RemoveAPI(UsdPhysics.ArticulationRootAPI)
    report = {'massless_bodies': [str(path) for path in light]} if light else {}
    if stray:
        report['stray_roots'] = [str(path) for path in stray]
    for path in welded:
        over.OverridePrim(path).SetActive(False)
    if welded:
        report['world_joints'] = [str(path) for path in welded]
    for path, name in powerless:  # unlimited, as USD's default
        UsdPhysics.DriveAPI.Apply(over.OverridePrim(path), name).CreateMaxForceAttr(FLT_MAX)
    if powerless:
        report['unlimited_drives'] = [f'{path}:{name}' for path, name in powerless]
    for path in sites:
        UsdGeom.Imageable(over.OverridePrim(path)).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    if sites:
        report['hidden_sites'] = [str(path) for path in sites]
    if turned:  # the root as usd_root_offset finds it: the articulation root, else the single rigid body
        wrapper = over.GetPrimAtPath(top.GetPath())
        prims = list(Usd.PrimRange(wrapper, Usd.TraverseInstanceProxies()))
        roots = [prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)] or \
            [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        report.update(_reroot(over, wrapper, roots[0]))
    layer.Save()
    return report


# Meshes -> USD --------------------------------------------------------------

def mesh_to_usd(files, target, static=False, mass=None):
    """One USD for every mesh in ``files``: parts keep their names, colours and base-colour textures; an
    object gets one rigid body, a scene static triangle colliders.

    An object collides through closed convex hulls of its parts (invisible ``guide`` meshes under
    ``collisions/``), and its mass properties (centre of mass, principal inertia and axes) are authored from
    them: downloaded meshes are often open, and an engine deriving volume and inertia from an open render mesh
    gets them wrong (a PLY bunny turned Newton's whole world to NaN)."""
    import numpy as np
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/' + Tf.MakeValidIdentifier(target.stem))
    stage.SetDefaultPrim(root.GetPrim())
    taken, textures, triangles, low, high, hulls = set(), 0, 0, None, None, []

    def unique(text):
        base = Tf.MakeValidIdentifier(str(text) or 'part')
        name, index = base, 2
        while name in taken:
            name, index = f'{base}_{index}', index + 1
        taken.add(name)
        return name

    def material(prim_name, color, image=None):
        nonlocal textures
        look = UsdShade.Material.Define(stage, root.GetPath().AppendPath(f'Looks/{prim_name}'))
        shader = UsdShade.Shader.Define(stage, look.GetPath().AppendChild('Surface'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.6)
        diffuse = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
        diffuse.Set(Gf.Vec3f(*color))
        if image is not None:
            file = target.parent / 'textures' / f'{prim_name}.png'
            file.parent.mkdir(exist_ok=True)
            image.convert('RGBA' if image.mode in ('RGBA', 'LA', 'P') else 'RGB').save(file)
            reader = UsdShade.Shader.Define(stage, look.GetPath().AppendChild('st'))
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            texture = UsdShade.Shader.Define(stage, look.GetPath().AppendChild('diffuse'))
            texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(f'./textures/{file.name}')
            texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.CreateOutput('result', Sdf.ValueTypeNames.Float2))
            texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
            texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
            diffuse.ConnectToSource(texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
            textures += 1
        look.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface', Sdf.ValueTypeNames.Token))
        return look

    def add(part, vertices, faces, color, uv=None, image=None, colors=None):
        nonlocal triangles, low, high
        mesh = UsdGeom.Mesh.Define(stage, root.GetPath().AppendChild(part))
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr([Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))])
        if uv is not None and image is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                [Gf.Vec2f(float(a), float(b)) for a, b in uv])
        if colors is not None:
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([Gf.Vec3f(*map(float, c)) for c in colors])
        else:
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(part, color, image))
        triangles += len(faces)
        if static:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        else:
            hull = _hull(vertices, faces)
            if hull is not None:
                hulls.append(hull)
                collider = UsdGeom.Mesh.Define(stage, root.GetPath().AppendPath(f'collisions/{part}'))
                collider.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in hull.vertices])
                collider.CreateFaceVertexCountsAttr([3] * len(hull.faces))
                collider.CreateFaceVertexIndicesAttr([int(i) for i in hull.faces.reshape(-1)])
                collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
                UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr('convexHull')
        low = vertices.min(0) if low is None else np.minimum(low, vertices.min(0))
        high = vertices.max(0) if high is None else np.maximum(high, vertices.max(0))

    for file in files:
        suffix = file.suffix.lower()
        try:
            if suffix in BLENDER_SUFFIXES:
                raise ValueError('Blender')
            parts = _trimesh_parts(file, suffix in Y_UP_SUFFIXES)
        except Exception as error:  # noqa: BLE001 - Blender reads more formats
            if not shutil.which('blender'):
                raise RuntimeError(f'{file.name}: {"需要安装 Blender 才能读取" if str(error) == "Blender" else error}') from None
            parts = _blender_parts(file, target.parent)
        for part_name, vertices, faces, color, uv, image, colors in parts:
            add(unique(part_name), np.asarray(vertices, float), np.asarray(faces, int), color, uv, image, colors)
    if not triangles:
        raise ValueError('文件里没有三角网格')
    if not static:
        if not hulls:
            raise ValueError('物体的网格都是平面，没有体积，无法生成碰撞体；请按场景导入')
        # Mass from the parts' hulls at DENSITY when not given: an estimate the entity's mass parameter overrides.
        volume = sum(hull.volume for hull in hulls)
        mass = round(max(float(mass) if mass is not None else volume * DENSITY, .01), 4)
        centre, moments, axes = _hull_inertia(hulls, mass)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        properties = UsdPhysics.MassAPI.Apply(root.GetPrim())
        properties.CreateMassAttr(mass)
        properties.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, centre)))
        properties.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, moments)))
        properties.CreatePrincipalAxesAttr(axes)
    stage.GetRootLayer().Save()
    size, limit = (high - low).tolist(), 500 if static else 20
    return {'parts': len(taken), 'triangles': int(triangles), 'textures': textures, 'mass': None if static else mass,
            'bounds': [[round(float(v), 4) for v in low], [round(float(v), 4) for v in high]],
            'size': [round(float(v), 4) for v in size],
            # An object over 20 m or a scene over 500 m was probably modelled in centimetres (Sponza is 30 m)
            'units_hint': 'mm' if max(size) > limit * 10 else 'cm' if max(size) > limit else None}


def _hull(vertices, faces):
    """The closed convex hull of a part (trimesh), or None for a flat part."""
    import trimesh
    try:
        hull = trimesh.Trimesh(vertices, faces, process=False).convex_hull
    except Exception:  # noqa: BLE001 - qhull rejects flat or degenerate input
        return None
    return hull if hull.volume > 1e-12 else None


def _hull_inertia(hulls, mass):
    """Centre of mass, principal moments and principal axes (Gf.Quatf) of ``hulls`` sharing ``mass`` at one
    density, in the asset frame."""
    import numpy as np
    from pxr import Gf
    density = mass / sum(hull.volume for hull in hulls)
    masses = np.array([hull.volume * density for hull in hulls])
    centres = np.array([hull.center_mass for hull in hulls])
    centre = masses @ centres / mass
    inertia = np.zeros((3, 3))
    for hull, part_mass, part_centre in zip(hulls, masses, centres):
        offset = part_centre - centre  # parallel axis theorem about the common centre
        inertia += hull.moment_inertia * density + part_mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))
    moments, vectors = np.linalg.eigh(inertia)
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    # Gf matrices act on row vectors: rows = the principal axes (columns of ``vectors``).
    rotation = Gf.Matrix4d(Gf.Matrix3d(*vectors.T.reshape(-1).tolist()), Gf.Vec3d(0, 0, 0)).ExtractRotation()
    return centre, np.maximum(moments, 1e-9), Gf.Quatf(rotation.GetQuat())


def _colour(material):
    """Base colour (0-1 RGB) and image of a trimesh material."""
    factor = getattr(material, 'baseColorFactor', None)
    if factor is None:
        factor = getattr(material, 'diffuse', None)
    color = (.7, .7, .7) if factor is None else tuple(float(v) / (255 if max(factor[:3]) > 1 else 1) for v in factor[:3])
    image = getattr(material, 'baseColorTexture', None) or getattr(material, 'image', None)
    return color, image


def _trimesh_parts(file, y_up):
    import numpy as np
    import trimesh
    scene = trimesh.load(str(file), force='scene', process=False)
    turn = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float) if y_up else np.eye(4)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        vertices = trimesh.transform_points(mesh.vertices, turn @ transform)
        color, uv, image, colors = (.7, .7, .7), None, None, None
        visual = mesh.visual
        if visual.kind == 'texture':
            color, image = _colour(visual.material)
            uv = visual.uv if image is not None and visual.uv is not None else None
            image = image if uv is not None else None
        elif visual.kind == 'vertex':
            colors = np.asarray(visual.vertex_colors)[:, :3] / 255.0
            color = tuple(colors.mean(0))
        elif visual.kind == 'face':
            color = tuple(np.asarray(visual.face_colors)[:, :3].mean(0) / 255.0)
        else:
            material = getattr(visual, 'material', None)
            if material is not None:
                color, _ = _colour(material)
        parts.append((node if geometry.startswith('geometry_') else geometry, vertices, np.asarray(mesh.faces), color, uv, image, colors))
    return parts


BLENDER_EXPORT = '''
import bpy, sys
source, target = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
suffix = source.lower().rsplit(".", 1)[-1]
if suffix == "blend":
    bpy.ops.wm.open_mainfile(filepath=source)
else:
    {"fbx": bpy.ops.import_scene.fbx, "abc": bpy.ops.wm.alembic_import,
     "obj": bpy.ops.wm.obj_import, "stl": bpy.ops.wm.stl_import, "ply": bpy.ops.wm.ply_import,
     "glb": bpy.ops.import_scene.gltf, "gltf": bpy.ops.import_scene.gltf}[suffix](filepath=source)
for obj in bpy.context.scene.objects:
    obj.select_set(obj.type == "MESH")
bpy.ops.wm.obj_export(filepath=target, export_selected_objects=True, forward_axis="Y", up_axis="Z",
                      export_materials=True, path_mode="COPY")
'''


def _blender_parts(file, scratch):
    """Formats trimesh cannot read: Blender imports them (Z-up) and writes OBJ, which trimesh then reads."""
    out = Path(scratch) / 'blender' / (file.stem + '.obj')
    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([shutil.which('blender'), '-b', '--factory-startup', '--python-expr', BLENDER_EXPORT, '--',
                           str(file), str(out)], capture_output=True, text=True, timeout=900)
    if done.returncode or not out.is_file():
        raise RuntimeError(f'Blender 读取 {file.name} 失败：{(done.stderr or done.stdout)[-400:]}')
    return _trimesh_parts(out, False)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='run.py asset import', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='A file, a directory of files or a zip')
    parser.add_argument('--kind', choices=KINDS, default='object')
    parser.add_argument('--name')
    parser.add_argument('--main', help='Directory source: the file to import when it holds several (a path inside it)')
    parser.add_argument('--xacro', default='', help='Arguments of a .urdf.xacro, e.g. "dof:=7 add_gripper:=true"')
    parser.add_argument('--project', help='The project it belongs to: kept in projects/<project>/assets/ and on '
                                          'its shelf; without one it is a workspace asset for every project (maintainers)')
    options = parser.parse_args(argv)
    root, library = project_assets(options.project) if options.project else (IMPORTED, LIBRARY)
    directory = new_directory(options.name or options.source.stem, root)
    sources = [options.source] if options.source.is_file() else [p for p in options.source.rglob('*') if p.is_file()]
    for path in sources:
        relative = path.name if options.source.is_file() else path.relative_to(options.source)
        with path.open('rb') as stream:
            store(directory, relative, stream, path.stat().st_size)
    record = import_asset(directory, options.kind, options.name, main=options.main, xacro=options.xacro)
    register(library_entry(directory, record), library)  # on the editor's asset shelf, as an import there would be
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f'资产架：{directory.name}（{library.relative_to(ROOT)}）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
