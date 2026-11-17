"""Browser scene editor for unified scene projects (viser).

Same contract as the Blender add-on: every change goes through
``shared.assembly.editing.apply_edits`` against the loaded revision, so the
terminal, Blender and the browser edit one scene.yaml. Native meshes are
exported by the project's simulator environment in the background; until then entities show
schematic proxies (or the previous mesh). Layout parents and groups never create
physical constraints; only "physical" attachments write connections/mounts.

  python run.py edit <project>                # runs in the project's backend environment
  .venv-il3/bin/python tools/web_editor.py --scene projects/x/scene.yaml
"""
import argparse
import atexit
import contextlib
import copy
import functools
import gzip
import hashlib
import warnings
import html
import io
import json
import os
from pathlib import Path
import posixpath
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import environments, simready  # noqa: E402
from shared.assembly import frames as F  # noqa: E402
from shared.assembly.editing import SceneDocument, apply_edits, editing_copy  # noqa: E402
from shared.assembly.mesh_preview import triangles  # noqa: E402
from shared.assembly.visuals import (ExportWorker, SceneChanged, cache_path, ensure_visuals, exporter_available,  # noqa: E402
                                    exporter_python, geometry_backend, geometry_keys, load_visuals,
                                     previous_visuals)

CATEGORY_LABELS = {'robot': '机器人', 'object': '物体 / 设备', 'terrain': '地面 / 地形',
                   'sensor': '传感器', 'light': '照明', 'group': '组合', 'abstract': '抽象体', 'environment': '场景环境'}
CONTROL_TYPES = {'joint_position': '关节位置 (rad / m)', 'joint_velocity': '关节速度 (rad/s / m/s)',
                 'joint_effort': '关节力矩 / 力 (N·m / N)'}
ABSTRACT_SHAPES = {'point': '点', 'direction': '方向', 'rectangle': '矩形', 'box': '长方体'}
ABSTRACT_ICONS = {'point': 'point', 'direction': 'arrow-narrow-right', 'rectangle': 'square', 'box': 'box-model'}
ABSTRACT_NOTES = {'point': '位置；可选作观察 position / orientation。',
                  'direction': '方向读取本地 +X 轴；观察 position / orientation / direction。',
                  'rectangle': '平面矩形（全长）；可作为其他实体的随机复位范围。',
                  'box': '长方体（全长）；可作为其他实体的随机复位范围。'}
ARROW_LENGTH = .3
ALIGN = {'保留朝向': 'none', 'XYZ 轴对齐': 'all', 'X 轴同向': 'x', 'Y 轴同向': 'y', 'Z 轴同向': 'z'}
PLACEMENT = {'保持当前位置': 'keep', '对齐到父物体': 'origin', '自定义偏移': 'custom'}
COLORS = {'mesh': (178, 189, 201), 'terrain': (89, 110, 128), 'dex_cube': (217, 102, 26),
          'proxy': (89, 140, 217), 'selected': (255, 170, 40), 'abstract': (51, 230, 102),
          'light': (255, 220, 120), 'sensor': (60, 60, 60)}
NONE = '（未选择）'
ROOT_BODY = '根部'
FREE = '（不安装，固定在世界中）'
ICONS = {'robot': 'robot', 'object': 'box', 'terrain': 'mountain', 'sensor': 'radar', 'light': 'bulb',
         'group': 'stack-2', 'abstract': 'map-pin'}
TREE_INDENT = 14
# viser renders each GUI button as a full-width Mantine button; tree rows restyle
# them by id so the scene list reads like an outliner, not a column of buttons.
TREE_ROW = ('[id="{id}"]{{height:1.65em!important;padding:0 6px 0 {pad}px!important;background:{bg}!important;'
            'color:var(--mantine-color-text)!important;font-weight:{weight}!important;border:0!important}}'
            '[id="{id}"]:hover{{background:{hover}!important}}'
            '[id="{id}"] .mantine-Button-inner,[id="{id}"] > span{{justify-content:flex-start!important}}'
            'div:has(> [id="{id}"]){{padding-bottom:1px!important;margin:0 4px!important}}')
# Isaac pinhole preset of shared/backends/isaac/assets.py:camera (mm). USD/Isaac
# cameras look down -Z with +Y up; viser frustums and renders use OpenCV (+Z, -Y).
CAMERA_FOCAL, CAMERA_APERTURE = 24.0, 20.955
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
GIZMO_SCALE = .35
# Entities a press-and-drag in the view slides over the floor; robots and terrain move only with the gizmo
# so dragging across them still orbits the camera.
DRAGGABLE = ('object', 'abstract')
SHELF_PAGE = Path(__file__).with_name('asset_shelf.html')  # the add tab's asset shelf (an iframe)
OUTLINER_PAGE = Path(__file__).with_name('scene_outliner.html')  # the scene tab's outliner (an iframe)
# Physics of an object row (outliner toggle and inspector): label, outliner hint.
PHYSICS = {'dynamic': ('受力运动', '动态：受重力和接触力'),
           'kinematic': ('不受力', '运动学：位置由场景运动或 Task 给出，能推动别的物体'),
           'fixed': ('固定', '根部固定在原处（fix_root_link）'),
           'static': ('静态', '静态几何：没有刚体；填写质量后才受力运动')}
MOTIONS = {None: '不自行运动', 'wander': '区域内游走', 'path': '沿路径'}
SKETCH_BACKGROUND = (243, 245, 248)  # canvas of the editor's flat-shaded asset sketches


def _gizmo_handles():
    """Unit-scale sample points of viser's pivot controls: axis arrows, plane squares and quarter arcs."""
    steps = np.linspace(.1, 1.0, 10)
    axes = [np.outer(steps, axis) for axis in np.eye(3)]
    square = (1 / 7 + .225 / 2)
    planes = [square * (np.eye(3)[a] + np.eye(3)[b]) for a, b in ((0, 1), (1, 2), (0, 2))]
    angle = np.linspace(0, np.pi / 2, 9)
    arcs = [.65 * (np.outer(np.cos(angle), np.eye(3)[a]) + np.outer(np.sin(angle), np.eye(3)[b]))
            for a, b in ((0, 1), (1, 2), (2, 0))]
    return np.vstack([*axes, np.array(planes), *arcs])


GIZMO_HANDLES = _gizmo_handles()
# The 3D canvas wrapper: the axes overlay is anchored to its top-right corner, inside docked panels.
AXES_OVERLAY = ('div[style*="z-index: 0"]:has(> div > div > canvas)::after{{content:"";position:absolute;top:10px;'
                'right:10px;width:84px;height:84px;z-index:1;pointer-events:none;'
                'background:url("data:image/svg+xml,{svg}") center/contain no-repeat}}')


def axes_svg(wxyz):
    """World axes as seen by a camera (OpenCV frame: +X right, +Y down, +Z forward), farthest drawn first."""
    view = F.quat_matrix(wxyz)[:3, :3].T  # world -> camera
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="-42 -42 84 84">',
             '<circle r="40" fill="white" fill-opacity=".72" stroke="#ced4da"/>']
    axes = [(view @ axis, label, color) for axis, label, color in
            zip(np.eye(3), 'XYZ', ('#e03131', '#2f9e44', '#1971c2'))]
    for direction, label, color in sorted(axes, key=lambda axis: -axis[0][2]):
        x, y = direction[:2] * 26
        parts.append(f'<circle cx="{-x:.1f}" cy="{-y:.1f}" r="2.5" fill="{color}" fill-opacity=".45"/>'
                     f'<line x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
                     f'<text x="{x * 1.3:.1f}" y="{y * 1.3:.1f}" fill="{color}" font-size="11" font-weight="700" '
                     f'font-family="sans-serif" text-anchor="middle" dominant-baseline="central">{label}</text>')
    return ''.join(parts) + '</svg>'
UNDO_LIMIT = 100
THUMBNAILS = ROOT / '.cache' / 'asset_thumbnails'
GALLERY_SCENE = ROOT / '.cache' / 'asset_gallery' / 'scene.yaml'
# User-imported assets: named shared-factory presets, listed in the gallery.
LIBRARY = ROOT / 'assets' / 'library.json'
IMPORT_FACTORIES = ('urdf_robot', 'usd_robot', 'usd_object', 'usd_articulation')
# Tiles written by `run.py asset terrain`: listed on the shelf as usd_object presets.
TERRAINS = ROOT / 'assets' / 'terrains'
IMPORT_KINDS = {'机器人（带关节）': 'robot', '物体（柜门、抽屉等关节自动识别）': 'object'}
# Global polish on top of viser 1.1.1's Mantine panels. Folder classes are viser's
# compiled names; if they change the panels simply fall back to viser's look.
THEME_CSS = """
._3o7djo0{border:1px solid #e6e9ee!important;border-radius:8px!important;margin:6px 8px 10px!important;
  padding:30px 0 2px!important;background:#fff}
._3o7djo1{position:absolute!important;top:0!important;left:0!important;right:0!important;transform:none!important;
  height:30px;display:flex!important;align-items:center;gap:4px;padding:0 10px!important;background:#f6f7f9!important;
  border-bottom:1px solid #e6e9ee;border-radius:8px 8px 0 0;font-size:12.5px!important;font-weight:600!important;color:#343a40!important}
._3o7djo1:hover{background:#eef1f5!important}
._3o7djo1 > svg{margin-left:auto;opacity:.55}
._3o7djo0:has(> ._3o7djo1 + div[aria-hidden="true"]){padding-bottom:0!important}
._3o7djo0:has(> ._3o7djo1 + div[aria-hidden="true"]) > ._3o7djo1{border-bottom:none;border-radius:8px}
._3o7djo0 > div[aria-hidden="true"] + div[aria-hidden="false"]{display:none}
._3o7djo0 ._3o7djo0{margin:6px 6px 8px!important;border-color:#eceef2!important}
._3o7djo0 ._3o7djo0 > ._3o7djo1{background:#fbfbfc!important;font-weight:500!important}
.mantine-Flex-root:has(> div > div > .mantine-Checkbox-root){flex-direction:row-reverse;justify-content:flex-end;
  gap:8px;border-radius:6px;padding:2px 4px;margin:0 -4px}
.mantine-Flex-root:has(> div > div > .mantine-Checkbox-root):hover{background:#f3f5f8}
.mantine-Flex-root:has(> div > div > .mantine-Checkbox-root) > div:first-child{width:auto!important;flex:1 1 0;padding-right:0!important}
.mantine-Flex-root:has(> div > div > .mantine-Checkbox-root) > div:first-child p{color:var(--mantine-color-text)!important;
  letter-spacing:0!important;cursor:pointer}
.mantine-Flex-root:has(> div > div > .mantine-Checkbox-root) > div:last-child{flex:0 0 auto!important;display:flex}
button[style*="--mantine-color-red-filled"]{--button-bg:transparent!important;--button-hover:var(--mantine-color-red-0)!important;
  --button-color:var(--mantine-color-red-7)!important;--button-bd:1px solid var(--mantine-color-red-3)!important}
h3{font-size:16px!important;margin-top:2px!important}
.rl-status{font-size:12px;line-height:1.45;color:#495057;background:#f3f5f8;border-radius:6px;padding:5px 8px;
  overflow-wrap:anywhere}
.rl-status.error{color:#c92a2a;background:#fff5f5}
"""
OBSERVATION_CSS = ('{labels}{{white-space:pre-line;font-size:11.5px!important;line-height:1.35!important;color:#868e96!important}}'
                   '{first}{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#212529}}')
# Group folders holding a ticked input; `:checked` keeps it live without a server round trip.
OBSERVATION_GROUP_CSS = ('._3o7djo0:has({checked}){{border-color:#a5d8ff!important}}'
                         '._3o7djo0:has({checked}) > ._3o7djo1{{background:#e7f5ff!important;color:#1864ab!important;'
                         'box-shadow:inset 3px 0 0 #339af0}}'
                         '._3o7djo0:has({checked}) > ._3o7djo1:hover{{background:#d0ebff!important}}')
# The view command under the save button. A srcdoc page so its copy button runs (viser HTML runs no
# scripts); the clipboard API needs a secure origin, so a remote http editor falls back to execCommand.
COMMAND_PAGE = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font:12px/1.4 system-ui,sans-serif;background:transparent;overflow:hidden}
.c{display:flex;align-items:center;gap:6px;height:30px;padding:0 4px 0 9px;border:1px solid #e3e6ea;border-radius:7px;background:#f6f7f9}
code{flex:1;min-width:0;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#1f2328;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;user-select:all}
button{flex:none;height:22px;padding:0 8px;border:1px solid #cfd4da;border-radius:5px;background:#fff;color:#495057;font:inherit;cursor:pointer}
button:hover{background:#eef3fd}</style></head><body><div class="c" title="在终端运行：用仿真器打开已保存的场景（未保存的修改不包含在内）">
<code id="c" title="__COMMAND__">__COMMAND__</code><button id="b">复制</button></div><script>
const b=document.getElementById('b'),text=document.getElementById('c').textContent;
b.onclick=async()=>{try{await navigator.clipboard.writeText(text)}catch(e){const t=document.createElement('textarea');
t.value=text;document.body.appendChild(t);t.select();document.execCommand('copy');t.remove()}
b.textContent='已复制';setTimeout(()=>b.textContent='复制',1200)};</script></body></html>'''
ROLE_PURPOSE = {'planner': '生成目标', 'executor': '输出控制动作'}
# A button group used as a two-way switch: one button per module, its role and state written
# under the name, the trained one filled. Viser gives the group no id; its (hidden) row label
# points at the uuid.
TRAIN_ROW = '.mantine-Flex-root:has(> div > p > label[for="{id}"])'
TRAIN_SWITCH_CSS = ('{row}{{padding:10px 0 14px}}{row} > div:first-child{{display:none}}'
                    '{row} > div:last-child{{flex:1 1 auto!important;width:100%}}'
                    '{row} > div:last-child > .mantine-Flex-root{{flex-wrap:nowrap;gap:8px}}'
                    '{row} button{{height:auto!important;min-width:0!important;padding:7px 6px!important;'
                    '--button-bg:var(--mantine-color-default)!important;--button-hover:#f3f5f8!important;'
                    '--button-color:var(--mantine-color-text)!important;--button-bd:1px solid #dee2e6!important}}'
                    '{row} .mantine-Button-label{{flex-direction:column;height:auto;line-height:1.4;white-space:normal;'
                    'font-size:13px;font-weight:600}}'
                    '{row} .mantine-Button-label::after{{font-size:11px;font-weight:400;opacity:.8}}')
TRAIN_BUTTON_CSS = '{row} button:nth-child({n}) .mantine-Button-label::after{{content:{caption}}}'
TRAIN_ON_CSS = ('{row} button:nth-child({n}){{--button-bg:var(--mantine-color-orange-filled)!important;'
                '--button-hover:var(--mantine-color-orange-filled-hover)!important;'
                '--button-color:var(--mantine-color-white)!important;--button-bd:1px solid transparent!important}}')
# The ✕ button sits before its folder; lift it onto the middle of that folder's 30px header,
# left of the chevron. The 1px height keeps the folder's top margin (pinned to 6px) from
# collapsing through the wrapper, so the header centre is 1 + 6 + 1 + 15 px below its top.
CONTROL_REMOVE_CSS = ('div:has(> [id="{id}"]){{height:1px;padding:0!important;position:relative;z-index:3}}'
                      'div:has(> [id="{id}"]) + div{{margin-top:6px!important}}'
                      'button[id="{id}"][style]{{position:absolute;right:30px;top:23px;transform:translateY(-50%);'
                      'width:22px!important;height:22px!important;min-width:0;padding:0!important;'
                      '--button-bd:none!important;border-radius:4px}}'
                      '[id="{id}"] .mantine-Button-inner{{justify-content:center}}'
                      '[id="{id}"] .mantine-Button-label{{display:none}}'
                      # Viser places the icon absolutely with a half-size offset; let it flow and centre.
                      '[id="{id}"] .mantine-Button-section,[id="{id}"] .mantine-Button-section > div'
                      '{{margin:0!important;width:auto!important;height:auto!important}}'
                      '[id="{id}"] .mantine-Button-section svg{{position:static!important;transform:none!important;'
                      'display:block;width:15px;height:15px}}')
def sketch_on_white(path):
    """PNG bytes of a shelf picture; editor sketches drawn on their grey canvas are brightened to white."""
    from PIL import Image
    with Image.open(path) as image:
        if image.convert('RGB').getpixel((0, 0)) != SKETCH_BACKGROUND:
            return path.read_bytes()
        table = [min(255, round(v * 255 / c)) for c in SKETCH_BACKGROUND for v in range(256)]
        stream = io.BytesIO()
        image.convert('RGB').point(table).save(stream, format='PNG', optimize=True)
    return stream.getvalue()


def simready_preview_url(entry):
    """The official SimReady preview, as served beside the editor (tools/simready_library.py)."""
    return f"preview/320/{urllib.parse.quote(entry['publisher'])}/{urllib.parse.quote(entry['name'])}.jpg"


def simready_thumbnail(publisher, name):
    """Asset-shelf card image of a SimReady wrapper: the official white-background preview."""
    return THUMBNAILS / f'simready-light_{publisher}_{name}.png'


def environment_thumbnail(key):
    """Asset-shelf card image of a catalog environment (shared.environments): its publisher's render."""
    return THUMBNAILS / f'environment_{key.replace("/", "_")}.png'


def _spawn_text(spawn, shown=3):
    places = '；'.join(f"({', '.join(f'{value:g}' for value in item['position'])})" for item in spawn[:shown])
    return f'可放机器人的空地（场景坐标，地面高度）：{places}' + (f' 等 {len(spawn)} 处' if len(spawn) > shown else '')


def _resolve(owner, reference):
    return posixpath.normpath(posixpath.join(owner, reference)) if reference.startswith('.') else reference


def _body_bound(category, factory, scene):
    """Sensors that exist only on a body; the backend declares which may also stand alone (FREE_SENSORS)."""
    from shared.assembly.assets import backend_module
    return category == 'sensor' and factory not in backend_module(geometry_backend(scene), 'assets').FREE_SENSORS


def _body_pattern_matches(pattern, name):
    try:
        return re.fullmatch(pattern, name) is not None
    except re.error:
        return False


def _body_groups(bodies):
    """Link groups sharing a name suffix, e.g. '所有 *_foot（4 个）' -> '.*_foot', for multi-link sensors."""
    suffixes = {}
    for body in bodies:
        if '_' in body:
            suffixes.setdefault(body.rsplit('_', 1)[1], []).append(body)
    return {f'所有 *_{suffix}（{len(names)} 个）': f'.*_{re.escape(suffix)}'
            for suffix, names in suffixes.items() if len(names) > 1}


def _wxyz(matrix):
    return tuple(float(v) for v in F.matrix_quat(matrix))


def _xyz(matrix):
    return tuple(float(v) for v in np.asarray(matrix)[:3, 3])


def camera_intrinsics(width, height, focal=CAMERA_FOCAL):
    """Vertical field of view (rad) and aspect of the pinhole camera (fixed horizontal aperture)."""
    return 2 * float(np.arctan(CAMERA_APERTURE * height / width / (2 * focal))), width / height


def _ray_box(origin, direction, half):
    direction = np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    low, high = (-half - origin) / direction, (half - origin) / direction
    near, far = np.minimum(low, high).max(), np.maximum(low, high).min()
    if near > far or far < 0:
        return None
    return float(near if near > 0 else far)


def _ray_sphere(origin, direction, centre, radius):
    offset = origin - centre
    b = offset @ direction
    disc = b * b - (offset @ offset - radius * radius)
    if disc < 0:
        return None
    root = np.sqrt(disc)
    t = -b - root if -b - root >= 0 else -b + root
    return float(t) if t >= 0 else None


def _ray_triangles(origin, direction, vertices, faces):
    """Nearest Möller–Trumbore hit over all triangles at once."""
    a = vertices[faces[:, 0]]
    e1, e2 = vertices[faces[:, 1]] - a, vertices[faces[:, 2]] - a
    p = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, p)
    valid = np.abs(det) > 1e-12
    inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    s = origin - a
    u = np.einsum('ij,ij->i', s, p) * inverse
    q = np.cross(s, e1)
    v = (q @ direction) * inverse
    t = np.einsum('ij,ij->i', e2, q) * inverse
    hit = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return float(t[hit].min()) if hit.any() else None


def ray_shape(shape, origin, direction):
    """Distance along a unit ray (in the shape's frame) to an editor shape, else None."""
    kind = shape['type']
    if kind == 'mesh':
        if 'bounds' not in shape:
            shape['bounds'] = (shape['vertices'].min(0), shape['vertices'].max(0))
        low, high = shape['bounds']
        if _ray_box(origin - (low + high) / 2, direction, (high - low) / 2 + 1e-4) is None:
            return None
        return _ray_triangles(origin, direction, shape['vertices'].astype(float), shape['faces'])
    if kind == 'box':
        return _ray_box(origin, direction, np.asarray(shape['size'], float) / 2)
    if kind == 'sphere':
        return _ray_sphere(origin, direction, np.zeros(3), shape['radius'])
    if kind == 'arrow':
        return _ray_sphere(origin, direction, np.array([shape['length'] / 2, 0.0, 0.0]), shape['length'] / 3)
    if kind == 'frustum':
        return _ray_sphere(origin, direction, np.array([0.0, 0.0, -.04]), .06)
    if kind == 'grid':
        if abs(direction[2]) < 1e-9:
            return None
        t = -origin[2] / direction[2]
        x, y = origin[:2] + t * direction[:2]
        return float(t) if t >= 0 and max(abs(x), abs(y)) <= shape['size'] / 2 else None
    return _ray_sphere(origin, direction, np.zeros(3), max(shape['length'] * .5, .03))


def component_period(text):
    """Decision period in seconds from the editor text field."""
    try:
        return float(str(text).strip())
    except ValueError:
        raise ValueError(f'决策周期须为秒数：{text!r}') from None


def component_dimension(text):
    """Output dimension from the editor text field: a positive integer or auto."""
    text = str(text).strip()
    if text == 'auto':
        return text
    try:
        return int(text)
    except ValueError:
        raise ValueError(f'输出维度须为正整数或 auto：{text!r}') from None


def observation_group(key, source, components=()):
    """Folder label and row name: obs.robot.target.linear_velocity -> ('robot', 'target.linear_velocity')."""
    head, _, name = key.partition('.')
    if not name:
        return '其他', key
    note = ('组件接口' if head in components else '仿真' if head == 'simulation' else
            source.get('group_label') if source.get('group_label') not in (None, head, '自定义 / 未声明来源') else '')
    return f'{head} · {note}' if note else head, name


def observation_caption(meta):
    """Description plus tensor shape and unit, as in the observation.py comment."""
    info = meta.get('information') or {}
    shape = 'B×' + '×'.join('N' if size is None else str(size) for size in info.get('shape', ())) if info.get('shape') else ''
    parts = [meta.get('label', ''), shape, meta.get('units') or info.get('unit') or '']
    return ' · '.join(str(part) for part in parts if part)


class Node:
    """Editor transform tree; world poses compose like Blender parenting."""
    def __init__(self, key, local=None, parent=None, shape=None, owner=None):
        self.key, self.parent, self.shape, self.owner = key, parent, shape, owner
        self.local = np.eye(4) if local is None else np.array(local, dtype=float)
        self.name, self.handle = None, None

    def world(self):
        return self.local if self.parent is None else self.parent.world() @ self.local

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


_MESHES = {}


def load_mesh(filename):
    signature = (filename, Path(filename).stat().st_mtime_ns)
    if signature not in _MESHES:
        with gzip.open(filename, 'rt') as stream:
            geometry = json.load(stream)
        parts = [(part['name'], np.asarray(part['vertices'], dtype=np.float32).reshape(-1, 3),
                  triangles(part['faces']).astype(np.uint32)) for part in geometry['parts'] if part['faces']]
        _MESHES[signature] = (parts, geometry.get('frames', {}), geometry.get('joints', []))
    return _MESHES[signature]


def usd_placeholder(item):
    """``(low, high)`` root-frame corners of an imported USD entity (``usd_path`` and ``scale``), or None."""
    from shared.asset_formats import usd_bounds
    from shared.assembly.assets import workspace_file
    parameters = item.get('parameters') or {}
    if not isinstance(parameters.get('usd_path'), str):
        return None
    try:
        bounds = usd_bounds(workspace_file(parameters['usd_path']))
    except Exception:  # noqa: BLE001 - a missing or unreadable file keeps the generic placeholder
        return None
    if bounds is None:
        return None
    scale = parameters.get('scale', 1.)
    scale = np.asarray(scale if isinstance(scale, (list, tuple)) else [scale] * 3, float)
    low, high = np.asarray(bounds[0]) * scale, np.asarray(bounds[1]) * scale
    return np.minimum(low, high), np.maximum(low, high)


def thumbnail_key(factory, parameters):
    """Default-parameter assets share the factory name; anything else is keyed by its parameters."""
    if not parameters:
        return factory
    return factory + '-' + hashlib.sha1(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:10]


def mesh_parts(record, positions=None):
    """Root-frame triangles of an exported native visual at its initial joint state."""
    parts, frames, joints = load_mesh(record['path'])
    positions = {**record.get('initial_state', {}).get('joint_positions', {}), **(positions or {})}
    current = F.forward_kinematics(frames, joints, positions) if joints else \
        {name: F.pose_matrix(pose) for name, pose in frames.items()}
    for name, vertices, faces in parts:
        body = next((body for body in frames if '/' + body + '/' in name), None)
        matrix = current[body] @ np.linalg.inv(F.pose_matrix(frames[body])) if body else np.eye(4)
        yield vertices @ matrix[:3, :3].T + matrix[:3, 3], faces


def render_thumbnail(record, positions=None, size=128):
    """Flat-shaded orthographic sketch (painter's algorithm); PNG bytes, or None without geometry."""
    from PIL import Image, ImageDraw
    pieces = [vertices[faces] for vertices, faces in mesh_parts(record, positions) if len(faces)]
    if not pieces:
        return None
    triangles_ = np.concatenate(pieces).astype(np.float64)
    eye = np.array([1.0, -1.2, .8]) / np.linalg.norm([1.0, -1.2, .8])
    right = np.cross(-eye, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -eye)
    screen = np.stack([triangles_ @ right, -(triangles_ @ up)], axis=-1)
    normals = np.cross(triangles_[:, 1] - triangles_[:, 0], triangles_[:, 2] - triangles_[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    light = eye + .5 * right + .6 * np.array([0.0, 0.0, 1.0])
    shade = .35 + .65 * np.abs(normals[keep] / lengths[keep, None] @ (light / np.linalg.norm(light)))
    screen, depth = screen[keep], (triangles_[keep] @ eye).mean(axis=1)
    canvas = size * 2
    low, high = screen.reshape(-1, 2).min(axis=0), screen.reshape(-1, 2).max(axis=0)
    scale = canvas * .86 / max(float(np.max(high - low)), 1e-6)
    screen = (screen - (low + high) / 2) * scale + canvas / 2
    image = Image.new('RGB', (canvas, canvas), SKETCH_BACKGROUND)
    draw = ImageDraw.Draw(image)
    base = np.array([96, 140, 196])
    for index in np.argsort(depth):
        draw.polygon([tuple(point) for point in screen[index]], fill=tuple(int(c) for c in base * shade[index]))
    stream = io.BytesIO()
    image.resize((size, size), Image.LANCZOS).save(stream, format='PNG', optimize=True)
    return stream.getvalue()


def load_library():
    try:
        return json.loads(LIBRARY.read_text()).get('assets', [])
    except (OSError, ValueError):
        return []


def terrain_tiles():
    """Library-style presets for the dual-backend terrain tiles under assets/terrains."""
    tiles = []
    for path in sorted(TERRAINS.glob('*/ASSET.json')):
        try:
            record = json.loads(path.read_text())
            usd, terrain = record['formats']['usd'], record['terrain']
        except (OSError, ValueError, KeyError):
            continue
        label = terrain.get('label') or path.parent.name
        if any(tile['label'] == label for tile in tiles):  # same kind, other size: tell the cards apart
            label += f' {terrain["size"]:g} m'
        tiles.append({'id': 'terrain_' + path.parent.name, 'label': label,
                      'factory': 'usd_object', 'parameters': {'usd_path': usd}, 'category': 'terrain',
                      'note': f'地形地块 · {terrain["size"]:g} m 见方，以实体原点为中心；'
                              f'参考出生点（相对中心）{[round(v, 3) for v in terrain["spawn_origin"]]}'})
    return tiles


def usd_joints(path):
    """Movable joints of a USD file: its ASSET.json joint_order when recorded, else read from the file."""
    from shared.asset_formats import recorded_joint_order, usd_joint_order
    recorded = recorded_joint_order(path)
    return list(recorded) if recorded is not None else usd_joint_order(path)


def recorded_assets():
    """Library-style presets for objects imported under assets/ (e.g. ``run.py asset simready``):
    usd_object, dynamic as their USD bodies and joints say. Robots stay with their own factories."""
    items = []
    for path in sorted(ROOT.glob('assets/*/ASSET.json')):
        try:
            record = json.loads(path.read_text())
            usd, joints, source = record['formats']['usd'], record.get('joint_order') or [], record['simready']
        except (OSError, ValueError, KeyError):
            continue
        note = f'SimReady {source.get("asset", path.parent.name)}（{source.get("license", "许可见 ASSET.json")}）'
        note += f' · {len(joints)} 个被动关节，可读关节位置/速度' if joints else ' · 无关节'
        item = {'id': 'asset_' + path.parent.name, 'label': path.parent.name, 'category': 'object',
                'factory': 'usd_object', 'parameters': {'usd_path': usd}, 'note': note}
        parts = Path(record.get('source', '')).parts
        if parts[:2] == ('assets', 'simready') and len(parts) > 3:  # the official preview of its mirrored source
            item['simready'] = parts[2], parts[3]
            item['thumbnail'] = simready_thumbnail(*item['simready']).stem
        items.append(item)
    return items


def save_library(entries):
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY.with_suffix('.tmp')
    temporary.write_text(json.dumps({'assets': entries}, ensure_ascii=False, indent=1) + '\n')
    os.replace(temporary, LIBRARY)


def action(method):
    """GUI callbacks run on viser worker threads: serialize and report errors."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return self.guarded(functools.partial(method, self))(*args, **kwargs)
    return wrapper


class Editor:
    def __init__(self, scene, *, host='127.0.0.1', port=8080, native_preview=True, library_port=None):
        import viser
        self.scene = None
        self._edit(scene)
        atexit.register(self.discard_copy)
        self.server = viser.ViserServer(host=host, port=port, label='RLinf 场景编辑器')
        self.lock = threading.RLock()
        self.syncing = 0
        self.dragging = False
        self.selected = None
        self.drag_events, self.body_drag, self.placing = queue.Queue(), None, None
        self.draft = None
        self.backend_problem = None  # Why the project's backend rejects the saved scene, if it does.
        self.export_thread = None
        self.export_again = None
        # Isaac starts only when an asset lacks cached meshes, then serves follow-up imports until idle.
        self.exporter, self.exporter_lock = None, threading.Lock()
        self.preview_restore, self.preview_key = None, None
        self.open_section = None
        self.hidden = set()
        self.inspector_items, self.tree_rows = [], {}
        self.outliner_page, self.outliner_revision, self.tree_changed = False, 0, set()
        self.motion_preview, self.path_marker, self.motion_drafts = None, None, {}
        # Entities waiting for their native mesh: {path: {'target', 'shown', 'start', 'handle'}}; the placeholder
        # box fills up from the bottom as the exporter reports progress (_animate_loading).
        self.loading = {}
        self.shown = (None, None)
        self.camera_image, self.render_wanted = None, threading.Event()
        self.undo, self.redo, self.undo_tag, self.history_scene = [], [], None, None
        self._build_gui()
        self.load()
        self.server.on_client_connect(self._client_connected)
        self.server.scene.on_click()(self._scene_clicked)
        if native_preview and self._needs_native():
            self.start_export()
        threading.Thread(target=self._watch_changes, daemon=True).start()
        threading.Thread(target=self._camera_renderer, daemon=True).start()
        threading.Thread(target=self._drag_worker, daemon=True).start()
        threading.Thread(target=self._animate_loading, daemon=True).start()
        self._start_library(host, library_port or self.server.get_port() + 1)
        threading.Thread(target=self._load_simready, daemon=True).start()
        threading.Thread(target=self._load_environments, args=(True,), daemon=True).start()

    def guarded(self, function):
        def wrapper(*args, **kwargs):
            with self.lock:
                try:
                    return function(*args, **kwargs)
                except Exception as error:  # noqa: BLE001 - surfaced in the page
                    traceback.print_exc()
                    self.report(f'⚠️ {error}', error=True)
        return wrapper

    # ------------------------------------------------------------------ loading
    def _edit(self, target):
        """Edit ``target`` through a fresh working copy (``self.scene``); only save writes ``target``."""
        target = Path(target).resolve()
        text = target.read_text()
        working = editing_copy(target)
        working.write_text(text)
        if self.scene is not None and self.scene != working:
            self.discard_copy()  # the previous environment's copy
        self.target, self.scene, self.saved_text = target, working, text

    def discard_copy(self):
        if self.scene is not None:
            self.scene.unlink(missing_ok=True)

    def load(self, snapshot=None, state=None):
        with self.lock:
            self.snapshot = snapshot or SceneDocument(self.scene).snapshot()
            self.disk_text = self.scene.read_text()
            if self.history_scene != self.scene:  # Undo never crosses into another environment.
                self.undo, self.redo, self.history_scene = [], [], self.scene
            try:
                self.manifest = json.loads(cache_path(self.scene).read_text())['entities']
            except (OSError, ValueError, KeyError):
                self.manifest = {}
            # The same asset exported by other code (e.g. after a backend change), anywhere in the workspace.
            self.previous = previous_visuals(self.scene)
            self.preview_restore = None  # Previews refer to the nodes being replaced.
            self.pending = {'actions': {}, 'resets': {}, 'meshes': {}, 'shapes': {}, 'kinematic': {}, 'shown': None,
                            'components': {}, 'motion': {}, 'visible': {}, 'simulation': {}}
            self.motion_drafts = {}  # motions being edited, not yet complete (e.g. a path of one point)
            self._show_command()
            self._show_simulation()
            self.labels, self.parameters = {}, {}
            self._build_tree()
            if state:
                self._overlay(state)
                self._show_simulation()  # an undone engine or renderer choice
            self._emit()
            self._init_observations()
            if state:
                self.observation_enabled = dict.fromkeys(state['observations'], True)
                self.train_role = state['train']
                self._rebuild_observations()
            self._refresh_lists()
            keep = state['selected'] if state else self.selected
            if keep != self.selected:
                self.selected = None  # The previous selection's node may be gone.
            if keep and self._node_of(keep) is not None:
                # Same owner: select() refreshes it and keeps the section the user is working in.
                self.select(keep)
            else:
                self._rebuild_inspector()
            self._check_backend()
            self.report(f'已加载 {len(self.snapshot["entities"])} 个实体 · {self.target.parent.name}/{self.target.name}')
        threading.Thread(target=self._harvest_thumbnails, daemon=True).start()

    def _check_backend(self):
        """Run the project backend's static scene check on what was just written.

        Saving never waits on it: an edit can pass through a state the backend rejects
        (a scanner whose targets are being chosen), but the page says so until it is fixed."""
        from shared.assembly.backend_scene import prepare
        backend = geometry_backend(self.scene)
        try:
            prepare(self.scene, backend)
            problem = None
        except ImportError:
            problem = None  # A project asset needing the simulator SDK is checked by run.py check.
        except Exception as error:  # noqa: BLE001 - shown in the page, never blocks saving
            problem = f'{backend} 后端不接受当前场景：{error}'
        if problem and problem != self.backend_problem:
            for client in self.server.get_clients().values():
                client.add_notification('已保存，但无法训练', problem, color='yellow', auto_close_seconds=15)
        self.backend_problem = problem

    def visual(self, item):
        """Current native visual, else the last exported one (marked stale)."""
        preview = item.get('preview') or {}
        if str(preview.get('path') or '').endswith('.mesh.json.gz') and preview.get('preview_version'):
            return preview, False
        for stale in (self.manifest.get(item['path']) or {}, self.previous.get(item['path']) or {}):
            path = stale.get('path') or ''
            if path.endswith('.mesh.json.gz') and stale.get('preview_version') and Path(path).is_file():
                return stale, True
        return None, False

    def initial_display(self, path):
        item = self.items.get(path, {})
        record, _ = self.visual(item) if item else (None, False)
        return (record or {}).get('initial_state', {}) or item.get('initial_display', {})

    def bodies(self, path):
        """Link names of an entity, known once its native visual was exported."""
        return self.initial_display(path).get('body_names', []) or list(self.info[path]['bodies'])

    def _needs_native(self):
        return any(item.get('factory') and item['category'] in ('robot', 'object', 'terrain')
                   and self.visual(item)[0] is None or self.visual(item)[1]
                   for item in self.snapshot['entities'] if item.get('factory'))

    def _build_tree(self):
        snapshot = self.snapshot
        self.items = {item['path']: item for item in snapshot['entities']}
        self.info, self.nodes = {}, []
        for item in snapshot['entities']:
            record, stale = self.visual(item)
            pose = item['display_pose']
            if stale and not item.get('pose_authored') and record.get('default_pose'):
                pose = record['default_pose']
            node = self._node(item['path'], F.pose_matrix(pose), owner=('entity', item['path']))
            info = {'node': node, 'base': F.matrix_pose(F.pose_matrix(pose)), 'bodies': {}, 'meshes': [],
                    'fixed': None, 'joint': None, 'grouped': False, 'layout_parent': None, 'pending': None,
                    'note': (item.get('preview') or {}).get('note', ''), 'kinematics': None}
            self.info[item['path']] = info
            if record and record.get('path'):
                self._attach_mesh(item, info, record, stale)
            else:
                self._proxy(item, node)
        for item in snapshot['entities']:
            parent = self.info.get(item['parent'])
            self.info[item['path']]['node'].parent = parent['node'] if parent else None
        self._connect_physical()
        self.groups = {}
        for name, group in snapshot.get('layout_groups', {}).items():
            helper = self._node('group:' + name, F.pose_matrix(group.get('pose', {})),
                                shape={'type': 'axes', 'length': .3}, owner=('group', name))
            self.groups[name] = helper
            for path in group['entities']:
                info = self.info[path]
                if info['fixed']:
                    raise ValueError('物理固定实体应通过其父实体移动，不可另加布局组合')
                self._reparent(info['node'], helper)
                info['grouped'] = True
        pending = dict(snapshot.get('layout_parents', {}))
        while pending:
            ready = [path for path, value in pending.items() if value['entity'] not in pending]
            if not ready:
                raise ValueError('布局父子关系存在循环')
            for path in ready:
                value = pending.pop(path)
                info = self.info[path]
                try:
                    parent = self.body_node(value['entity'], value.get('body', ''))
                except ValueError:
                    # Layout parenting stores world poses; the origin is a safe stand-in.
                    parent = self.info[value['entity']]['node']
                    info['note'] = '等待原生连杆坐标；暂时保留当前世界位置'
                self._reparent(info['node'], parent)
                info['grouped'] = True
                info['layout_parent'] = value['entity']

    def _node(self, key, local=None, parent=None, shape=None, owner=None):
        node = Node(key, local, parent, shape, owner)
        self.nodes.append(node)
        return node

    def _reparent(self, node, parent):
        world = node.world()
        node.parent = parent
        node.local = np.linalg.inv(parent.world()) @ world if parent else world

    def _attach_mesh(self, item, info, record, stale):
        parts, frames, joints = load_mesh(record['path'])
        authored = item.get('initial_state', {}).get('joint_positions', {})
        positions = {**record.get('initial_state', {}).get('joint_positions', {}), **authored}
        current = F.forward_kinematics(frames, joints, positions) if joints else \
            {name: F.pose_matrix(pose) for name, pose in frames.items()}
        for body, matrix in current.items():
            info['bodies'][body] = self._node(f'{item["path"]}#{body}', matrix, info['node'])
        info['kinematics'] = {'frames': frames, 'joints': joints}
        color = COLORS['terrain'] if item['category'] == 'terrain' else \
            COLORS['dex_cube'] if item.get('factory') == 'dex_cube' else COLORS['mesh']
        for index, (name, vertices, faces) in enumerate(parts):
            body = next((body for body in frames if '/' + body + '/' in name), None)
            parent = info['bodies'][body] if body else info['node']
            local = np.linalg.inv(F.pose_matrix(frames[body])) if body else None
            mesh = self._node(f'{item["path"]}@{index}', local, parent, owner=('entity', item['path']),
                              shape={'type': 'mesh', 'vertices': vertices, 'faces': faces, 'color': color})
            info['meshes'].append(mesh)
        if stale:
            info['note'] = '显示上次导出的网格；参数或初始状态已变，点击“刷新原生网格”更新'
        else:
            info['note'] = record.get('note', info['note'])

    def _proxy(self, item, node):
        factory, info = item.get('factory'), self.info[item['path']]
        if item['category'] == 'abstract':
            info['note'] = '抽象体：无模型、无物理；含义由 task.py / 模型决定'
            self._abstract_shape(item['path'])
            return
        info['note'] = info['note'] or '示意代理：尺寸和连杆坐标不代表物理模型'

        def box(size, position=(0, 0, 0), color=COLORS['proxy']):
            info['meshes'].append(self._node(f'{item["path"]}@{len(info["meshes"])}', F.transform(position), node,
                                             owner=('entity', item['path']),
                                             shape={'type': 'box', 'size': size, 'color': color}))
        if factory in ('go2', 'anymal_c'):
            box((.62, .28, .20) if factory == 'anymal_c' else (.5, .25, .17), (0, 0, .48 if factory == 'anymal_c' else .35))
            for x in (-.18, .18):
                for y in (-.14, .14):
                    box((.045, .045, .3), (x, y, .2))
        elif factory == 'seattle_table':
            box((1.2, .8, .05), (0, 0, -.03))
        elif factory == 'dex_cube':
            box((.06, .06, .06), color=COLORS['dex_cube'])
        elif factory == 'box':  # A native cuboid: its parameters are its exact geometry.
            defaults = {name: field.get('default') for name, field in self.asset_entry('box').get('parameters', {}).items()}
            parameters = {**defaults, **item.get('parameters', {})}
            info['note'] = ''
            color = parameters.get('color')
            box(tuple(parameters.get('size') or (1., 1., 1.)),
                color=tuple(round(255 * c) for c in color) if color else COLORS['proxy'])
        elif factory in ('terrain', 'ground_plane'):
            node.shape = {'type': 'grid', 'size': 4.0 if factory == 'terrain' else 2.0}
        elif usd_placeholder(item) is not None:  # an imported USD: a box of its authored extent
            low, high = usd_placeholder(item)
            box(tuple(np.maximum(high - low, .005)), tuple((low + high) / 2))
        elif factory == 'camera':
            fov, aspect = camera_intrinsics(*self.camera_optics(item))
            node.shape = {'type': 'frustum', 'fov': fov, 'aspect': aspect}
        elif item['category'] == 'light':
            node.shape = {'type': 'sphere', 'radius': .05, 'color': COLORS['light']}
        elif item.get('kind') == 'robot':
            box((.3, .3, .6), (0, 0, .3))
        else:
            node.shape = {'type': 'axes', 'length': .08}

    def asset_entry(self, factory):
        return next((entry for entry in self.snapshot['catalog'].get('assets', []) if entry['id'] == factory), {})

    def parameter_values(self, item):
        """Factory defaults, saved parameters, then unsaved edits (when they parse)."""
        defaults = {name: field.get('default') for name, field in self.asset_entry(item['factory']).get('parameters', {}).items()}
        try:
            pending = json.loads(self.parameters.get(item['path'], '{}'))
        except ValueError:
            pending = {}
        return {**defaults, **item.get('parameters', {}), **(pending if isinstance(pending, dict) else {})}

    def camera_optics(self, item):
        parameters = self.parameter_values(item)
        return (float(parameters.get('width') or 160), float(parameters.get('height') or 120),
                float(parameters.get('focal_length') or CAMERA_FOCAL))

    def body_node(self, path, body):
        info = self.info[path]
        if not body:
            return info['node']
        if body in info['bodies']:
            return info['bodies'][body]
        if body == self.initial_display(path).get('root_body'):
            return info['node']  # A rigid object without link frames (a USD cube) is its own root body.
        # A contact sensor may cover a link group such as .*_foot; preview it on the first match.
        matches = [name for name in info['bodies'] if _body_pattern_matches(body, name)]
        if not matches:
            raise ValueError(f'{path}: 连杆 {body} 无预览坐标，请先刷新原生模型')
        return info['bodies'][matches[0]]

    def _connect_physical(self):
        snapshot = self.snapshot
        for item in snapshot['entities']:
            mount = item.get('mount')
            if not mount:
                continue
            node, target = self.info[item['path']]['node'], _resolve(item['path'], mount['entity'])
            body = mount.get('body', '')
            bodies = self.info.get(target, {}).get('bodies', {})
            if target in self.info:
                node.parent = self.info[target]['node']
                with contextlib.suppress(ValueError):
                    node.parent = self.body_node(target, body)
            if body and body != self.initial_display(target).get('root_body') \
                    and not any(_body_pattern_matches(body, name) for name in bodies):
                self.info[item['path']]['note'] = '机身坐标预览不可用（暂用资产原点）；仿真按连杆安装'
        joints = list(snapshot.get('connections', []))
        for item in snapshot['entities']:
            for joint in item.get('connections', []):
                joints.append({**joint, **{end: {**joint[end], 'entity': _resolve(item['path'], joint[end]['entity'])}
                                          for end in ('parent', 'child') if joint.get(end)}})
        project = {entry['name'] for entry in snapshot.get('connections', [])}
        pending = list(joints)
        while pending:
            children = {joint['child']['entity'] for joint in pending}
            ready = [joint for joint in pending if not joint.get('parent') or joint['parent']['entity'] not in children]
            if not ready:
                raise ValueError('Cyclic fixed attachments')
            for joint in ready:
                pending.remove(joint)
                info = self.info[joint['child']['entity']]
                info['joint'] = {**joint, 'project': joint['name'] in project}
                node, parent = info['node'], joint.get('parent')
                try:
                    parent_node = self.body_node(parent['entity'], parent.get('body', '')) if parent else None
                    child_body = self.body_node(joint['child']['entity'], joint['child'].get('body', ''))
                except ValueError:
                    # Cold previews have no link frames: keep the placeholder,
                    # never invent a fixed offset from it.
                    info['pending'] = '等待原生连杆坐标；组合位置将在网格就绪后更新'
                    info['note'] = info['pending']
                    continue
                body_local = np.linalg.inv(node.world()) @ child_body.world()
                node.parent = parent_node
                node.local = F.fixed_child_local(joint.get('parent_pose', {}), joint.get('child_pose', {}), body_local)
                info.update(fixed={**joint, 'project': joint['name'] in project}, child_body_local=body_local,
                            base=F.matrix_pose(node.local), pending=None)

    def abstract_geometry(self, path):
        """(shape, size) of an abstract entity, including unsaved edits."""
        item = self.items[path]
        return self.pending['shapes'].get(path) or (item['shape'], list(item.get('size') or []))

    def _abstract_shape(self, path):
        shape, size = self.abstract_geometry(path)
        color = COLORS['abstract']
        self.info[path]['node'].shape = (
            {'type': 'sphere', 'radius': .04, 'color': color} if shape == 'point' else
            {'type': 'arrow', 'length': ARROW_LENGTH, 'color': color} if shape == 'direction' else
            {'type': 'box', 'size': tuple(size) if len(size) == 3 else (*size, .001), 'color': color, 'region': True})

    # ----------------------------------------------------------------- drawing
    def _emit(self):
        """Draw the node tree in the view, reusing what an earlier load drew.

        Names follow node keys (an entity path, its links and mesh parts), so a reload after a save, undo
        or export updates poses in place and redraws only shapes that changed; clearing the view and
        sending every mesh again made the scene flash."""
        scene = self.server.scene
        if not hasattr(self, 'emitted'):
            scene.reset()
            scene.add_grid('/floor_grid', width=10, height=10, cell_size=.5, section_size=1.0,
                           cell_color=(220, 220, 220), section_color=(170, 170, 170))
            self.gizmo = scene.add_transform_controls('/gizmo', scale=GIZMO_SCALE, depth_test=False, visible=False)
            self.gizmo.on_update(self._gizmo_moved)
            self.emitted = {}
        self.handles = {}
        order, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            if node.parent is not None:
                visit(node.parent)
            seen.add(id(node))
            order.append(node)
        for node in self.nodes:
            visit(node)
        self.hidden = {owner for owner in self.hidden if self._node_of(owner) is not None}
        hidden = {id(self._node_of(owner)) for owner in self.hidden}
        previous, self.emitted = self.emitted, {}
        for node in order:
            token = re.sub(r'[^A-Za-z0-9_]', '_', node.key)[-40:] + '_' + hashlib.sha1(node.key.encode()).hexdigest()[:6]
            node.name = (node.parent.name if node.parent is not None else '/scene') + '/' + token
            old = previous.pop(node.name, None)
            if old is not None:
                frame = old['frame']
                frame.position, frame.wxyz = _xyz(node.local), _wxyz(node.local)
            else:
                frame = scene.add_frame(node.name, show_axes=False, position=_xyz(node.local), wxyz=_wxyz(node.local))
            if frame.visible == (id(node) in hidden):
                frame.visible = id(node) not in hidden
            self.handles[id(node)] = frame
            signature = self._shape_signature(node)
            self.emitted[node.name] = {'frame': frame, 'shape': None, 'signature': signature}
            if old is not None and old['shape'] is not None and old['signature'] == signature:
                node.handle = old['shape']
                self.emitted[node.name]['shape'] = node.handle
                if old.get('highlighted'):  # drop the earlier selection's colour; select() sets it again
                    if self._solid(node):
                        node.handle.color = node.shape.get('color')
                    else:
                        self._draw_shape(node)
            else:
                if old is not None and old['shape'] is not None:
                    with contextlib.suppress(Exception):
                        old['shape'].remove()
                self._draw_shape(node)
        # Deepest first: a parent's removal already takes its children.
        for name in sorted(previous, key=len, reverse=True):
            with contextlib.suppress(Exception), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                previous[name]['frame'].remove()

    @staticmethod
    def _solid(node):
        return (node.shape['type'] in ('mesh', 'sphere', 'box') and not node.shape.get('region')
                and node.handle is not None and node.shape.get('color') is not None)

    def _shape_signature(self, node):
        """What _draw_shape draws for a node; equal signatures reuse the drawn handle."""
        shape = node.shape
        if not shape:
            return None
        faint = bool(node.owner and node.owner[0] == 'entity' and node.owner[1] in self.items
                     and not self.render_visible(node.owner[1]))
        return faint, tuple(sorted((key, id(value) if isinstance(value, np.ndarray) else repr(value))
                                   for key, value in shape.items()))

    def _draw_shape(self, node, highlight=False):
        shape, scene = node.shape, self.server.scene
        if not shape:
            return
        name, color = node.name + '/v', COLORS['selected'] if highlight else shape.get('color')
        kind = shape['type']
        # Hidden from rendering (visible: false): drawn faint, it still collides.
        faint = {'opacity': .3} if node.owner and node.owner[0] == 'entity' and node.owner[1] in self.items \
            and not self.render_visible(node.owner[1]) else {}
        if kind == 'mesh':
            handle = scene.add_mesh_simple(name, shape['vertices'], shape['faces'], color=color, side='double', **faint)
        elif kind == 'box' and shape.get('region'):
            # Abstract rectangle/box: a translucent body plus screen-width edges, so it
            # stays visible inside robots or on the ground. A flat rectangle is lifted
            # 2 mm in the view only, otherwise the ground plane hides it.
            size = np.asarray(shape['size'], float)
            lift = (0, 0, .002) if size[2] <= .001 else (0, 0, 0)
            handle = scene.add_box(name, color=color, dimensions=tuple(size), opacity=.25, side='double',
                                   cast_shadow=False, receive_shadow=False, position=lift)
            corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * size
            edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count('1') == 1]
            scene.add_line_segments(name + '/edges', corners[np.array(edges)], color,
                                    thickness=4. if highlight else 2.5, thickness_units='screen')
        elif kind == 'box':
            handle = scene.add_box(name, color=color, dimensions=shape['size'], opacity=faint.get('opacity', .85))
        elif kind == 'sphere':
            handle = scene.add_icosphere(name, radius=shape['radius'], color=color, **faint)
        elif kind == 'arrow':
            # Local +X: a shaft and a two-line head, as in Isaac's arrow marker.
            length, head = shape['length'], shape['length'] * .25
            points = np.array([[[0, 0, 0], [length, 0, 0]],
                               [[length, 0, 0], [length - head, head * .5, 0]],
                               [[length, 0, 0], [length - head, -head * .5, 0]],
                               [[length, 0, 0], [length - head, 0, head * .5]],
                               [[length, 0, 0], [length - head, 0, -head * .5]]], float)
            handle = scene.add_line_segments(name, points, color, thickness=.012 if highlight else .008)
        elif kind == 'frustum':
            handle = scene.add_camera_frustum(name, fov=shape['fov'], aspect=shape['aspect'], scale=.08,
                                              color=COLORS['selected'] if highlight else COLORS['sensor'],
                                              wxyz=_wxyz(GL_TO_CV))
        elif kind == 'grid':
            handle = scene.add_grid(name, width=shape['size'], height=shape['size'], cell_size=.25,
                                    cell_color=(150, 160, 170), section_color=(90, 100, 110))
        else:
            handle = scene.add_frame(name, show_axes=True, axes_length=shape['length'] * (1.6 if highlight else 1),
                                     axes_radius=shape['length'] * .04)
        entry = getattr(self, 'emitted', {}).get(node.name)
        if entry is not None:
            entry.update(shape=handle, highlighted=highlight)
        # No per-node click handlers: viser outlines hovered clickable nodes and
        # lets them swallow pointer events meant for the transform gizmo. Bodies that can be
        # moved get a drag handler instead: pressing on one and dragging slides it (see _drag).
        node.handle = handle
        if kind in ('mesh', 'box', 'sphere') and node.owner and node.owner[0] == 'entity' \
                and self.items.get(node.owner[1], {}).get('category') in DRAGGABLE:
            async def dragged(event, owner=node.owner):  # async: phases arrive in order
                self.drag_events.put((owner, event.phase, event.start_position, event.end_position,
                                      event.client.camera.position))
            handle.on_drag(dragged)

    def _update_frame(self, node):
        handle = self.handles.get(id(node))
        if handle is not None:
            handle.position, handle.wxyz = _xyz(node.local), _wxyz(node.local)

    def _highlight(self, owner, on):
        node = self._node_of(owner)
        if node is None:
            return
        shapes = self.info[owner[1]]['meshes'] if owner[0] == 'entity' else []
        for shaped in [node, *shapes]:
            if not (shaped.shape and shaped.name):
                continue
            if self._solid(shaped):
                # Recolour in place: re-adding the node would end a drag that is selecting it.
                shaped.handle.color = COLORS['selected'] if on else shaped.shape.get('color')
                entry = self.emitted.get(shaped.name)
                if entry is not None:
                    entry['highlighted'] = on
            else:
                self._draw_shape(shaped, highlight=on)

    def _client_connected(self, client):
        self._frame_camera(client)
        client.camera.on_update(self._draw_axes)
        self._draw_axes(client.camera)

    def _draw_axes(self, camera):
        content = '<style>' + AXES_OVERLAY.format(svg=urllib.parse.quote(axes_svg(camera.wxyz), safe='')) + '</style>'
        if self.axes_style.content != content:
            self.axes_style.content = content

    def _frame_camera(self, client):
        # Ray-cast sensors sit metres above their parent; frame what the user edits.
        points = [info['node'].world()[:3, 3] for path, info in self.info.items()
                  if self.items[path]['category'] in ('robot', 'object')] or [np.zeros(3)]
        centre = np.mean(points, axis=0)
        radius = max(1.0, float(np.max(np.linalg.norm(np.asarray(points) - centre, axis=1))) * 2.5)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        # viser shifts look_at along with position, so the target is set last.
        client.camera.position = tuple(centre + radius * np.array([1.2, -1.2, 0.9]))
        client.camera.look_at = tuple(centre)

    # --------------------------------------------------------------- selection
    def hits(self, origin, direction):
        """(owner, tier) under a world ray, nearest first within a tier: robots, objects and groups;
        then sensors, lights and abstract entities (markers often sit inside a body); then terrain."""
        origin = np.asarray(origin, float)
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        hidden = {id(self._node_of(owner)) for owner in self.hidden}
        best = {}
        for node in self.nodes:
            if not node.shape or not node.owner or hidden and any(id(n) in hidden for n in (node, *node.ancestors())):
                continue
            inverse = np.linalg.inv(node.world())
            distance = ray_shape(node.shape, inverse[:3, :3] @ origin + inverse[:3, 3], inverse[:3, :3] @ direction)
            if distance is None:
                continue
            kind, name = node.owner
            category = self.items[name]['category'] if kind == 'entity' else None
            rank = 2 if category == 'terrain' else 1 if category in ('sensor', 'light', 'abstract') else 0
            best[node.owner] = min(best.get(node.owner, (rank, distance)), (rank, distance))
        return [(owner, best[owner][0]) for owner in sorted(best, key=best.get)]

    def pick(self, origin, direction, current=None):
        """Nearest owner; clicking the selection again cycles to what lies behind it in the same tier."""
        hits = self.hits(origin, direction)
        if not hits:
            return None
        stack = [owner for owner, tier in hits if tier == hits[0][1]]
        if current in stack:
            return stack[(stack.index(current) + 1) % len(stack)]
        return stack[0]

    def _near_gizmo(self, origin, direction):
        """Whether the ray passes over a drawn gizmo handle (arrows, plane squares, rotation arcs)."""
        origin = np.asarray(origin, float)
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        points = (GIZMO_HANDLES * GIZMO_SCALE) @ F.quat_matrix(self.gizmo.wxyz)[:3, :3].T + np.asarray(self.gizmo.position)
        along = np.clip((points - origin) @ direction, 0.0, None)
        miss = np.linalg.norm(origin + along[:, None] * direction - points, axis=1)
        # Handles keep a constant on-screen width: tolerance grows with distance.
        return bool(np.any(miss < .012 + .02 * along))

    @action
    def _scene_clicked(self, event):
        origin = np.asarray(event.ray_origin, float)
        if self.placing:
            point = self.surface_point(origin, event.ray_direction)
            if point is None:
                raise ValueError('没有点到地面或物体表面，请朝下点击')
            placing = self.placing
            self.cancel_placement()
            placing['add']([round(float(value), 4) for value in point])
            return
        if self.gizmo.visible and self._near_gizmo(origin, event.ray_direction):
            return  # Clicks on the transform handles never select what lies behind them.
        self.select(self.pick(origin, event.ray_direction, self.selected))

    def _node_of(self, owner):
        if not owner:
            return None
        kind, name = owner
        if kind == 'entity':
            return self.info.get(name, {}).get('node')
        return self.groups.get(name)

    def _locked(self, owner):
        return owner and owner[0] == 'entity' and bool(self.info[owner[1]]['pending'])

    @action
    def select(self, owner):
        if owner != self.selected:
            self._end_preview()  # Switching away abandons an unapplied preview.
            self.open_section, self.undo_tag = None, None
        if self.selected and self.selected != owner:
            self._highlight(self.selected, False)
        self.selected = owner if self._node_of(owner) is not None else None
        if self.selected:
            self._highlight(self.selected, True)
        self._place_gizmo()
        self._rebuild_inspector()
        self._style_tree()
        self._draw_motion_path()

    @action
    def escape(self):
        if self.placing:
            self.cancel_placement(f'已取消放置 {self.placing["label"]}')
        elif self.preview_restore:
            self._end_preview()
            self._rebuild_inspector()
            self.report('已取消预览，恢复原位')
        else:
            self.select(None)

    @action
    def toggle_hidden(self, owner=None):
        """Editor-only visibility (e.g. terrain hiding what sits on it); never saved."""
        owner = owner or self.selected
        node = self._node_of(owner)
        if node is None:
            return
        self.hidden ^= {owner}
        self.handles[id(node)].visible = owner not in self.hidden
        self.outliner_revision += 1
        self._label_tree()
        if owner == self.selected and self.hidden_box is not None and self.hidden_box.value != (owner in self.hidden):
            self.syncing += 1
            try:
                self.hidden_box.value = owner in self.hidden
            finally:
                self.syncing -= 1

    # -------------------------------------------------- outliner toggles (scene_outliner.html)
    def render_visible(self, path):
        """Whether an entity appears in rendered images: ``visible`` for assets, ``visualization.abstract``
        for abstract entities (drawn only in previews)."""
        item = self.items[path]
        if item['category'] == 'abstract':
            shown = self.pending['shown'] if self.pending['shown'] is not None else \
                self.snapshot.get('visualization', {}).get('abstract', [])
            return item['name'] in shown
        return self.pending['visible'].get(path, item.get('visible', True))

    def physics_state(self, path):
        """``(mode, jointed)`` of an object (see PHYSICS), or None for entities without that switch:
        robots, sensors, terrain, lights, attached or mounted parts and shared (global) props."""
        from shared.assembly.catalog import asset_capabilities, factory_capabilities
        item, info = self.items[path], self.info[path]
        if item['category'] != 'object' or not item.get('factory') or item.get('scope') == 'global' \
                or info['fixed'] or item.get('mount') or item.get('instance_component') is not None:
            return None
        parameters = self.parameter_values(item)
        key = (item['factory'], json.dumps(parameters, sort_keys=True, default=str))
        cache = self.__dict__.setdefault('capability_cache', {})
        if key not in cache:  # a usd_object opens its USD file
            try:
                cache[key] = asset_capabilities(item['factory'], factory_capabilities(item['factory']), parameters)
            except Exception:  # noqa: BLE001 - an unreadable USD has no physics switch until it is fixed
                cache[key] = None
        capabilities = cache[key]
        if capabilities is None:
            return None
        jointed = 'joint_state' in capabilities
        if self.pending['kinematic'].get(path, item.get('kinematic', False)):
            return 'kinematic', jointed
        if parameters.get('fix_root_link'):
            return 'fixed', jointed
        return ('dynamic' if 'rigid_body' in capabilities else 'static'), jointed

    def current_motion(self, path):
        return self.pending['motion'][path] if path in self.pending['motion'] else self.items[path].get('motion')

    def _set_parameter(self, path, name, value):
        item = self.items[path]
        try:
            current = json.loads(self.parameters[path]) if path in self.parameters else dict(item.get('parameters', {}))
        except ValueError:
            current = dict(item.get('parameters', {}))
        if value is None:
            current.pop(name, None)
        else:
            current[name] = value
        self.parameters[path] = json.dumps(current, ensure_ascii=False)

    def toggle_physics(self, path):
        """Outliner switch: an object either moves under forces or not (kinematic, or a fixed root when
        it has joints, which cannot be kinematic)."""
        state = self.physics_state(path)
        if state is None:
            raise ValueError(f'{path} 没有物理开关（机器人、传感器、地形、已安装或固定连接的部件）')
        mode, jointed = state
        if mode == 'static':
            raise ValueError(f'{path} 是静态几何（没有刚体）；在参数里填写质量后才会受力运动')
        self.checkpoint()
        if mode == 'dynamic' and jointed:
            self._set_parameter(path, 'fix_root_link', True)
        elif mode == 'dynamic':
            self.pending['kinematic'][path] = True
        elif mode == 'fixed':
            self._set_parameter(path, 'fix_root_link', None if 'fix_root_link' not in self.items[path].get('parameters', {})
                                else False)
        else:
            self.pending['kinematic'][path] = False
            if self.current_motion(path):
                self.pending['motion'][path] = None
                self.report(f'{path} 改为受力运动，其场景运动已移除')
        mode = self.physics_state(path)[0]
        self.outliner_revision += 1
        if self.selected == ('entity', path):
            self._rebuild_inspector()
        self.report(f'{path}：{PHYSICS[mode][0]}；保存后写入 scene.yaml')

    def toggle_render(self, path):
        """Outliner switch: an asset's ``visible``, or whether an abstract entity is drawn in previews."""
        item = self.items[path]
        if item['category'] in ('sensor', 'light', 'group') or (item['category'] != 'abstract' and not item.get('factory')):
            raise ValueError(f'{path} 不出现在渲染画面中，没有该开关')
        self.checkpoint()
        shown = not self.render_visible(path)
        if item['category'] == 'abstract':
            current = self.pending['shown'] if self.pending['shown'] is not None else \
                list(self.snapshot.get('visualization', {}).get('abstract', []))
            self.pending['shown'] = [n for n in current if n != item['name']] + ([item['name']] if shown else [])
        else:
            if item.get('instance_component') is not None:
                raise ValueError(f'{path} 是模板内部部件；渲染可见性在模板中设置')
            self.pending['visible'][path] = shown
            for node in [self.info[path]['node'], *self.info[path]['meshes']]:
                if node.shape and node.name:
                    self._draw_shape(node, highlight=self.selected == ('entity', path))
        self.outliner_revision += 1
        if self.selected == ('entity', path):
            self._rebuild_inspector()
        what = '仿真中显示标记' if item['category'] == 'abstract' else '出现在相机与录像中'
        self.report(f'{path}：{"" if shown else "不"}{what}；保存后写入 scene.yaml')

    def outliner_get(self, name, query):
        with self.lock:
            if name != 'state':
                raise KeyError(f'未知的场景列表请求 {name}')
            if query.get('revision') == str(self.outliner_revision):
                return {'revision': self.outliner_revision}
            rows = []
            for owner, depth in self.tree_entries():
                kind, path = owner
                row = {'id': f'{kind}:{path}', 'depth': depth, 'icon': self._row_icon(owner), 'changed': owner in self.tree_changed,
                       'hidden': owner in self.hidden, 'selected': owner == self.selected, 'render': None,
                       'physics': None, 'motion': None}
                if kind == 'entity':
                    item, info = self.items[path], self.info[path]
                    row['label'] = (self.labels.get(path) or item['label']).strip() or item['name']
                    row['note'] = '固定' if info['fixed'] else '安装' if item.get('mount') else \
                        '布局' if info['layout_parent'] else ''
                    row['hint'] = f'{path} · {item["category_label"]}'
                    if item['category'] == 'abstract' or (item.get('factory') and item['category'] not in ('sensor', 'light')):
                        row['render'] = self.render_visible(path)
                        row['render_hint'] = (('仿真预览中画出标记' if row['render'] else '仿真预览中不画标记（只是读数）')
                                              if item['category'] == 'abstract' else
                                              '出现在相机与录像中' if row['render'] else '不出现在相机与录像中（仍参与碰撞）') \
                            + '；点击切换'
                    state = self.physics_state(path)
                    if state is not None:
                        row['physics'] = {'mode': state[0], 'label': PHYSICS[state[0]][0], 'hint': PHYSICS[state[0]][1]}
                    motion = self.current_motion(path)
                    if motion:
                        row['motion'] = MOTIONS[motion['type']]
                else:
                    row.update(label=path, note='', hint='布局组合（仅摆放）')
                rows.append(row)
            return {'revision': self.outliner_revision, 'rows': rows}

    def outliner_post(self, name, request):
        with self.lock:
            try:
                kind, _, path = str(request.get('id', '')).partition(':')
                owner = (kind, path)
                if self._node_of(owner) is None:
                    raise ValueError(f'场景中没有 {path}')
                if name == 'select':
                    self.select.__wrapped__(self, owner)
                elif name == 'toggle':
                    what = request.get('what')
                    if what == 'hidden':
                        self.toggle_hidden.__wrapped__(self, owner)
                    elif what == 'render' and kind == 'entity':
                        self.toggle_render(path)
                    elif what == 'physics' and kind == 'entity':
                        self.toggle_physics(path)
                    else:
                        raise KeyError(f'未知的开关 {what}')
                elif name == 'focus':
                    self.select.__wrapped__(self, owner)
                    self.focus_selection.__wrapped__(self)
                else:
                    raise KeyError(f'未知的场景列表请求 {name}')
            except Exception as error:
                self.report(f'⚠️ {error}', error=True)
                raise
            return self.outliner_get('state', {})

    @action
    def focus_selection(self):
        node = self._node_of(self.selected)
        if node is None:
            return
        points = [node.world()[:3, 3]]
        if self.selected[0] == 'entity':
            for mesh in self.info[self.selected[1]]['meshes']:
                if mesh.shape and mesh.shape['type'] == 'mesh':
                    world = mesh.world()
                    points.extend(mesh.shape['vertices'] @ world[:3, :3].T + world[:3, 3])
        points = np.asarray(points)
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
        distance = float(np.clip(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 1.5, .5, 12.0))
        for client in self.server.get_clients().values():
            direction = np.asarray(client.camera.position) - np.asarray(client.camera.look_at)
            direction = direction / max(np.linalg.norm(direction), 1e-6)
            # Keep the current heading but look down on the selection, never up from below the ground.
            if np.linalg.norm(direction[:2]) < .1:
                direction = np.array([1.2, -1.2, 0.9])
            direction[2] = max(direction[2], .3)
            direction = direction / np.linalg.norm(direction)
            # viser shifts look_at along with position, so the target is set last.
            client.camera.position = tuple(centre + direction * distance)
            client.camera.look_at = tuple(centre)

    def _place_gizmo(self):
        node = self._node_of(self.selected)
        if node is None or self._locked(self.selected):
            self.gizmo.visible = False
            return
        if not self.dragging:
            world = node.world()
            self.gizmo.position, self.gizmo.wxyz = _xyz(world), _wxyz(world)
        self.gizmo.visible = True

    def _gizmo_moved(self, event):
        # Never write the gizmo pose while the user drags it (jitter feedback).
        self.dragging = getattr(event, 'phase', 'end') in ('start', 'update')
        if not self.dragging:
            self.request_camera_view()
        with self.lock:
            node = self._node_of(self.selected)
            if node is None or self._locked(self.selected) or self.preview_restore:
                return
            world = F.transform(self.gizmo.position, F.quat_matrix(self.gizmo.wxyz))
            if np.allclose(world, node.world(), atol=1e-7):
                return
            self.checkpoint(('move', self.selected))
            if not self.dragging:
                self.undo_tag = None  # The next drag is a separate step.
            node.local = np.linalg.inv(node.parent.world()) @ world if node.parent else world
            self._update_frame(node)
            self._sync_pose_fields(node)

    def _sync_pose_fields(self, node):
        if not getattr(self, 'pose_fields', None):
            return
        self.syncing += 1
        try:
            self.pose_fields[0].value = tuple(round(v, 5) for v in _xyz(node.local))
            self.pose_fields[1].value = tuple(round(v, 3) for v in F.matrix_euler(node.local))
        finally:
            self.syncing -= 1

    def _drag_worker(self):
        """Body drags in event order (viser callbacks run on a pool): the grab point follows the pointer
        over the horizontal plane at its height, so the entity slides without changing height."""
        while True:
            owner, phase, grab, pointer, camera = self.drag_events.get()
            try:
                self._drag(owner, phase, np.asarray(grab, float), np.asarray(pointer, float), np.asarray(camera, float))
            except Exception as error:  # noqa: BLE001 - reported, the next drag starts fresh
                traceback.print_exc()
                self.body_drag = None
                self.report(f'拖动失败：{error}', error=True)

    def _drag(self, owner, phase, grab, pointer, camera):
        with self.lock:
            if phase == 'start':
                node = self._node_of(owner)
                if node is None or self._locked(owner) or self.preview_restore or self.placing:
                    return
                if owner != self.selected:
                    self.select(owner)
                self.checkpoint(('drag', owner, time.monotonic()))  # each drag is one undo step
                self.body_drag = {'owner': owner, 'grab': grab, 'world': node.world().copy()}
                return
            drag = self.body_drag
            if drag is None or drag['owner'] != owner:
                return
            if phase == 'end':
                self.body_drag, self.undo_tag = None, None
            # `pointer` lies on the pointer ray (viser's camera-facing drag plane); meet the grab height.
            ray = pointer - camera
            if abs(ray[2]) > 1e-6:
                t = (drag['grab'][2] - camera[2]) / ray[2]
                hit = camera + t * ray
                if t > 0 and np.linalg.norm(hit - camera) < 100:
                    node = self._node_of(owner)
                    world = drag['world'].copy()
                    world[:2, 3] += (hit - drag['grab'])[:2]
                    node.local = np.linalg.inv(node.parent.world()) @ world if node.parent else world
                    self._update_frame(node)
                    self._place_gizmo()
                    self._sync_pose_fields(node)
            if phase == 'end':
                self.request_camera_view()
                self.report(f'已移动 {self.items[owner[1]]["name"]}（保存后写入 scene.yaml）')

    def surface_point(self, origin, direction):
        """Where a view ray first meets a robot, object or terrain surface, else the floor (z = 0)."""
        origin = np.asarray(origin, float)
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        hidden = {id(self._node_of(owner)) for owner in self.hidden}
        nearest = None
        for node in self.nodes:
            if not node.shape or not node.owner or node.owner[0] != 'entity' \
                    or self.items[node.owner[1]]['category'] not in ('robot', 'object', 'terrain') \
                    or hidden and any(id(n) in hidden for n in (node, *node.ancestors())):
                continue
            inverse = np.linalg.inv(node.world())
            distance = ray_shape(node.shape, inverse[:3, :3] @ origin + inverse[:3, 3], inverse[:3, :3] @ direction)
            if distance is not None and (nearest is None or distance < nearest):
                nearest = distance
        if nearest is None and direction[2] < -1e-6:
            nearest = -origin[2] / direction[2]
        return None if nearest is None else origin + nearest * direction

    def arm_placement(self, label, add):
        """The next click in the view adds `label` there through ``add(position)``; Esc cancels."""
        self.placing = {'label': label, 'add': add}
        self.place_style.content = '<style>canvas{cursor:crosshair!important}</style>'
        self.report(f'放置 {label}：在 3D 视图中点击位置（落在点到的表面上，Esc 取消）')

    def cancel_placement(self, text=None):
        self.placing = None
        self.place_style.content = ''
        if text:
            self.report(text)

    @action
    def _pose_field_changed(self, _event):
        if self.syncing:
            return
        node = self._node_of(self.selected)
        if node is None:
            return
        self._end_preview()
        self.checkpoint(('move', self.selected))
        node.local = F.transform(self.pose_fields[0].value, F.euler_matrix(self.pose_fields[1].value))
        self._update_frame(node)
        self._place_gizmo()
        self.request_camera_view()

    # ------------------------------------------------------------- camera view
    def request_camera_view(self):
        if self.camera_image is not None:
            self.render_wanted.set()

    def _camera_renderer(self):
        """Latest-request worker: a client render is a network round trip, so never under the lock."""
        while True:
            self.render_wanted.wait()
            self.render_wanted.clear()
            try:
                self._render_camera_view()
            except Exception as error:  # noqa: BLE001 - shown next to the image
                with self.lock:
                    if self.camera_image is not None:
                        self.camera_caption.content = f'<small>渲染失败：{error}</small>'

    def _render_camera_view(self):
        with self.lock:
            clients = list(self.server.get_clients().values())
            owner, image = self.selected, self.camera_image
            if image is None or self.dragging:
                return
            if not clients:
                self.camera_caption.content = '<small>没有打开的网页，无法渲染</small>'
                return
            node = self._node_of(owner)
            fov, aspect = camera_intrinsics(*self.camera_optics(self.items[owner[1]]))
            world = node.world() @ GL_TO_CV  # Isaac camera frame -> OpenCV render convention.
            # The gizmo and the camera's own frustum sit at the eye and would fill the view.
            hidden = [handle for handle in (self.gizmo, node.handle) if handle is not None and handle.visible]
            for handle in hidden:
                handle.visible = False
        try:
            rows = 240
            frame = clients[0].get_render(rows, round(rows * aspect), wxyz=_wxyz(world), position=_xyz(world),
                                          fov=fov, timeout=5)
        finally:
            with self.lock:
                for handle in hidden:
                    handle.visible = True
                self._place_gizmo()
        with self.lock:
            if self.camera_image is image and self.selected == owner:
                image.image = frame
                self.camera_caption.content = ('<small>编辑器示意渲染：位姿与内参同 scene.yaml（复位姿态），'
                                               '画面不是 Isaac 传感器图像</small>')

    # ----------------------------------------------------------------- editing
    def collect(self):
        """Mirror of the Blender add-on's ``collect_edits``; unapplied previews are ignored."""
        with self._committed():
            return self._collect()

    @contextlib.contextmanager
    def _committed(self):
        held = [(node, node.local, node.parent) for node, _, _ in self.preview_restore or []]
        for node, local, parent in self.preview_restore or []:
            node.local, node.parent = local, parent
        try:
            yield
        finally:
            for node, local, parent in held:
                node.local, node.parent = local, parent

    def _collect(self):
        snapshot = self.snapshot
        payload = {'revision': snapshot['revision'], 'entities': [], 'observations': {},
                   'layout_groups': {}, 'actions': dict(self.pending['actions'])}
        project_connections = {entry['name'] for entry in snapshot.get('connections', [])}
        for item in snapshot['entities']:
            path, info = item['path'], self.info[item['path']]
            label = self.labels.get(path)
            if label is not None and label.strip() != item['label']:
                payload.setdefault('labels', {})[path] = label
            node = info['node']
            matrix = node.local
            if info['grouped']:
                matrix = node.world()
                parent = self.info.get(item['parent'])
                if parent:
                    matrix = np.linalg.inv(parent['node'].world()) @ matrix
            pose = F.matrix_pose(matrix)
            changes = {}
            if not F.same_pose(pose, info['base']):
                if info['pending']:
                    raise ValueError(f'{path}: 等待原生连杆坐标后才能调整固定安装位置')
                if info['fixed']:
                    joint = {key: value for key, value in info['fixed'].items() if key != 'project'}
                    if joint['name'] not in project_connections:
                        raise ValueError('模板内部固定连接请在模板中调整')
                    joint['parent_pose'] = F.fixed_parent_pose(node.local, info['child_body_local'], joint.get('child_pose', {}))
                    payload.setdefault('connections', {})[joint.pop('name')] = joint
                else:
                    changes['pose'] = pose
            if path in self.parameters:
                try:
                    parameters = json.loads(self.parameters[path])
                except ValueError as error:
                    raise ValueError(f'{path}: 参数不是有效 JSON：{error}') from None
                if not isinstance(parameters, dict):
                    raise ValueError(f'{path}: 参数必须是 JSON 对象')
                if parameters != item.get('parameters', {}):
                    if item.get('parameters', {}).keys() - parameters.keys():
                        raise ValueError('不支持删除参数；请编辑 YAML 后重新加载')
                    changes['parameters'] = parameters
            if path in self.pending['resets']:
                changes['reset'] = self.pending['resets'][path]
            if self.pending['kinematic'].get(path, item.get('kinematic', False)) != item.get('kinematic', False):
                changes['kinematic'] = self.pending['kinematic'][path]
            if path in self.pending['motion'] and self.pending['motion'][path] != (item.get('motion') or None):
                changes['motion'] = self.pending['motion'][path]
            if self.pending['visible'].get(path, item.get('visible', True)) != item.get('visible', True):
                changes['visible'] = self.pending['visible'][path]
            if path in self.pending['meshes'] and self.pending['meshes'][path] != item.get('meshes', []):
                changes['meshes'] = self.pending['meshes'][path] or None
            if path in self.pending['shapes']:
                shape, size = self.pending['shapes'][path]
                if shape != item['shape'] or list(size) != list(item.get('size') or []):
                    changes.update(shape=shape, size=list(size))
            if changes:
                payload['entities'].append({'path': path, 'frame': 'local', 'changes': changes})
        for name, node in self.groups.items():
            old = snapshot['layout_groups'][name]
            pose = F.matrix_pose(node.world())
            if not F.same_pose(pose, old.get('pose') or F.IDENTITY_POSE):
                payload['layout_groups'][name] = {**old, 'pose': pose}
        shown = self.pending['shown']
        if shown is not None and shown != snapshot.get('visualization', {}).get('abstract', []):
            payload['visualization'] = {'abstract': list(shown)}
        control = snapshot.get('control')
        if control and self.train_role != control['train']:
            payload['control_train'] = self.train_role
        for role, fields in self.pending['components'].items():
            saved, changed = control['components'][role], {}
            if 'period' in fields and (period := component_period(fields['period'])) != saved.get('period'):
                changed['period'] = period
            if 'dimension' in fields and (dimension := component_dimension(fields['dimension'])) != saved.get('dimension', 'auto'):
                changed['dimension'] = dimension
            if changed:
                payload.setdefault('components', {})[role] = changed
        for profile, order in self.observation_order.items():
            selected = [key for key in order if self.observation_enabled.get((profile, key))]
            if selected != list(control['components'][profile].get('observations', [])):
                payload['observations'][profile] = selected
        return payload

    def unsaved(self):
        """Edits not yet in scene.yaml (pending fields, structural edits held in the working copy) or train.yaml."""
        return self.dirty() or self.disk_text != self.saved_text or bool(self.pending.get('simulation'))

    def dirty(self):
        try:
            return any(value for key, value in self.collect().items() if key != 'revision')
        except ValueError:
            return True

    def changed_owners(self):
        """Tree markers for unsaved edits."""
        try:
            payload = self.collect()
        except ValueError:
            return set()
        owners = {('entity', entry['path']) for entry in payload['entities']}
        owners |= {('entity', path) for path in payload.get('labels', {})}
        owners |= {('entity', joint['child']['entity']) for joint in payload.get('connections', {}).values() if joint}
        return owners | {('group', name) for name in payload['layout_groups']}

    def _apply(self, payload, *, select=None, record=True):
        if record:
            self.checkpoint()
        result = apply_edits(self.scene, payload)
        if select is not None:
            self.selected = select
        self.load(result['snapshot'])
        return result

    @action
    def save(self, *, quiet=False):
        # Saving is not an undo step: undo still returns to the edit before it.
        simulation = dict(self.pending.get('simulation') or {})  # applying reloads, which clears pending
        result = self._apply(self.collect(), record=False)
        if self.target.read_text() != self.saved_text:
            raise ValueError('scene.yaml 已在编辑器外被修改，未覆盖；点击“放弃修改并重新加载”读取新内容')
        text = self.scene.read_text()
        result = {**result, 'changed': text != self.saved_text}
        if simulation:
            from shared.assembly.editing import write_simulation_settings
            write_simulation_settings(self.target, simulation)
            self.pending['simulation'] = {}
            result['changed'] = True
            self._show_simulation()
        if result['changed']:
            temporary = self.target.with_name(f'.{self.target.name}.save')
            temporary.write_text(text)
            os.replace(temporary, self.target)
            self.saved_text = text
        if self.draft:
            from shared.assembly.starter import publish_draft
            self.discard_copy()  # not part of the published project
            target, self.draft = publish_draft(self.target, self.draft).resolve(), None
            self.scene = None
            self._edit(target)
            self.load()
            self.report(f'环境已保存：{target.parent}；训练模型和任务可继续配置')
            return result
        if not quiet:
            self.report('已写入 scene.yaml；训练前请运行 run.py check' if result['changed'] else '没有配置变化')
        return result

    def _save_first(self):
        """Structural edits reload the expanded scene: move pending edits into the working copy first."""
        if self.dirty():
            self._apply(self.collect(), record=False)

    @action
    def reload(self):
        self.checkpoint()  # Discarded edits can be brought back with undo.
        self._edit(self.target)  # scene.yaml as saved, or as changed outside the editor
        self.load()

    # -------------------------------------------------------------------- undo
    def _state(self):
        """Everything an edit can change: the saved text plus unsaved editor state."""
        with self.lock, self._committed():
            nodes = {('entity', path): info['node'].local.copy() for path, info in self.info.items()}
            nodes.update({('group', name): node.local.copy() for name, node in self.groups.items()})
            return {'text': self.disk_text, 'nodes': nodes,
                    'labels': dict(self.labels), 'parameters': dict(self.parameters),
                    'pending': copy.deepcopy(self.pending), 'train': self.train_role,
                    'observations': frozenset(key for key, on in self.observation_enabled.items() if on),
                    'selected': self.selected}

    @staticmethod
    def same_state(a, b):
        plain = ('text', 'labels', 'parameters', 'pending', 'train', 'observations')
        if any(a[key] != b[key] for key in plain) or a['nodes'].keys() != b['nodes'].keys():
            return False
        return all(np.allclose(a['nodes'][key], b['nodes'][key], atol=1e-9) for key in a['nodes'])

    def checkpoint(self, tag=None):
        """Record the state before an edit; consecutive edits with one tag (a drag, typing) are one step."""
        with self.lock:
            if tag is not None and tag == self.undo_tag:
                return
            self.undo_tag = tag
            state = self._state()
            if not self.undo or not self.same_state(self.undo[-1], state):
                self.undo.append(state)
                del self.undo[:-UNDO_LIMIT]
            self.redo.clear()

    def _overlay(self, state):
        self.labels, self.parameters = dict(state['labels']), dict(state['parameters'])
        self.pending = copy.deepcopy(state['pending'])
        for path in self.pending['shapes']:
            if path in self.info:
                self._abstract_shape(path)
        for owner, local in state['nodes'].items():
            node = self._node_of(owner)
            if node is not None:
                node.local = local.copy()

    @action
    def undo_edit(self):
        self._step(self.undo, self.redo, '撤回')

    @action
    def redo_edit(self):
        self._step(self.redo, self.undo, '重做')

    def _step(self, source, target, verb):
        current = self._state()
        while source and self.same_state(source[-1], current):
            source.pop()
        if not source:
            self.report(f'没有可{verb}的操作')
            return
        if self.scene.read_text() != self.disk_text:
            self.undo, self.redo = [], []
            raise ValueError('scene.yaml 已在编辑器外被修改；撤回记录已清空，请重新加载')
        state = source.pop()
        target.append(current)
        self.undo_tag, self.open_section = None, None
        if state['text'] != self.disk_text:
            # Structural edits were applied to the working copy when made; undoing restores it too.
            temporary = self.scene.with_name(f'.{self.scene.name}.undo')
            temporary.write_text(state['text'])
            os.replace(temporary, self.scene)
            self.load(state=state)
        else:
            self.load(self.snapshot, state=state)
        self.report(f'已{verb}（剩余 {len(self.undo)} 步可撤回）')

    # ----------------------------------------------------------- native meshes
    def _export_visuals(self, scene, force=False):
        """Lightweight native meshes of ``scene``; the backend SDK starts only for uncached assets (serialised)."""
        with self.exporter_lock:
            if not force and geometry_keys(scene).keys() <= load_visuals(scene).keys():
                return ensure_visuals(scene)  # every mesh is saved already: no SDK needed
            python = exporter_python(geometry_backend(scene))
            if self.exporter is not None and self.exporter.python != python:
                self.exporter.close()  # another project's backend: its own interpreter
                self.exporter = None
            if self.exporter is None:
                self.exporter = ExportWorker(python, cache_path(scene).with_suffix('.exporter.log'))
            return self.exporter.request(scene, force=force, progress=self._export_progress)

    # ------------------------------------------------- loading placeholders that fill up
    def _placeholder_bounds(self, path):
        """Root-frame corners of the boxes standing in for an entity's mesh, or None."""
        corners = []
        for mesh in self.info[path]['meshes']:
            if mesh.shape and mesh.shape['type'] == 'box' and not mesh.shape.get('region'):
                half, centre = np.asarray(mesh.shape['size']) / 2, mesh.local[:3, 3]
                corners += [centre - half, centre + half]
        return (np.min(corners, 0), np.max(corners, 0)) if corners else None

    def _start_loading(self):
        """Entities drawn as placeholders get a fill that rises while their mesh is made."""
        for path, item in self.items.items():
            # A box is drawn exactly from its parameters: nothing to wait for.
            if path in self.loading or item['category'] not in ('robot', 'object') or item.get('factory') in (None, 'box') \
                    or self.visual(item)[0] is not None or self._placeholder_bounds(path) is None:
                continue
            self.loading[path] = {'target': 0., 'shown': 0., 'start': time.time(), 'handle': None}

    def _stop_loading(self):
        for state in self.loading.values():
            if state['handle'] is not None:
                with contextlib.suppress(Exception):
                    state['handle'].remove()
        self.loading = {}

    def _export_progress(self, values):
        """Exporter progress ({path: fraction}) for entities still drawn as placeholders."""
        for path, fraction in values.items():
            state = self.loading.get(path)
            if state is not None and float(fraction) > state['target']:
                state['target'], state['start'] = float(fraction), time.time()

    def _animate_loading(self):
        """Raise each loading placeholder's fill towards its progress. Between reports (the exporter's start-up
        and scene build come as one step without a count) it creeps on, up to 40% or 20% past the last report."""
        import math
        while True:
            time.sleep(.12)
            if not self.loading:
                continue
            with self.lock:
                for path, state in list(self.loading.items()):
                    info = self.info.get(path)
                    bounds = self._placeholder_bounds(path) if info else None
                    if bounds is None or ('entity', path) in self.hidden:
                        continue
                    target = state['target']
                    cap = .4 if target <= 0 else 1. if target >= 1 else min(target + .2, .95)
                    goal = target + (cap - target) * (1 - math.exp(-(time.time() - state['start']) / 4))
                    shown = state['shown'] + (goal - state['shown']) * (.5 if target >= 1 else .3)
                    name = info['node'].name + '/loading'
                    if state['handle'] is not None and state.get('name') == name and abs(shown - state['shown']) < .003:
                        continue  # a reload renames the nodes: draw again under the new name
                    state['shown'] = shown
                    low, high = bounds
                    size = high - low
                    height = max(size[2] * shown, 1e-4)
                    centre = ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] + height / 2)
                    # Slightly wider than the placeholder so it shows through it; replaced in place.
                    state['name'] = name
                    state['handle'] = self.server.scene.add_box(
                        name, color=(46, 110, 220), opacity=.75,
                        dimensions=(size[0] * 1.04 + .002, size[1] * 1.04 + .002, height), position=centre,
                        cast_shadow=False, receive_shadow=False)

    def start_export(self, force=False):
        backend = geometry_backend(self.scene)
        if not exporter_available(backend):
            self.report(f'后端 {backend} 没有原生网格导出；继续显示示意代理')
            return
        python = exporter_python(backend)
        if not python.exists():
            self.report(f'未找到 {backend} 的环境 {python}；继续显示示意代理（可设置 RLINF_EXPORT_PYTHON）')
            return
        with self.lock:
            self._start_loading()
            if self.export_thread and self.export_thread.is_alive():
                self.export_again = bool(self.export_again) or force
                self.report('原生网格导出进行中；结束后会再导出一次')
                return
            scene = self.scene
            self.export_again = None

        def run(force=force):
            while True:
                started = time.time()
                try:
                    self._export_visuals(scene, force)
                    error = None
                    for state in list(self.loading.values()):
                        state['target'], state['start'] = 1., time.time()
                    if self.loading:
                        time.sleep(.5)  # let the fills reach the top before the meshes replace them
                except SceneChanged:
                    continue  # saved again mid-export: export the current file
                except Exception as exception:
                    error = str(exception)[-1500:]
                with self.lock:
                    if self.export_again is None:
                        break
                    force, self.export_again = self.export_again, None
            with self.lock:
                self._stop_loading()
                if error:
                    self.report(f'原生网格导出失败：{error}', error=True)
                elif scene != self.scene:
                    pass
                elif self.preview_restore:
                    self.report(f'原生网格已就绪（{time.time() - started:.0f}s）；保存或下一次添加、删除后显示')
                else:
                    try:  # Pending fields move into the working copy (not scene.yaml) so loading keeps them.
                        self._save_first()
                    except ValueError as problem:
                        self.report(f'原生网格已就绪；修正后显示：{problem}', error=True)
                    else:
                        self.load()
                        self.report(f'原生网格已更新（{time.time() - started:.0f}s）')
        self.export_thread = threading.Thread(target=run, daemon=True)
        self.export_thread.start()
        self.report('后台导出原生网格（通常 5–20 秒），期间可继续编辑')

    # ---------------------------------------------------------------------- GUI
    def _show_command(self):
        """Under the save button: the command that runs this project's scene in the simulator (saved file)."""
        if not hasattr(self, 'command_strip'):
            return
        target = Path(self.target)
        if target.name != 'scene.yaml' or target.parent.parent != ROOT / 'projects':
            self.command_strip.content = ''  # view runs projects; a scene variant or loose file has no command
            return
        command = f'bash run.sh view {target.parent.name}'
        if getattr(self, 'shown_command', None) == command:
            return
        self.shown_command = command
        page = COMMAND_PAGE.replace('__COMMAND__', html.escape(command))
        self.command_strip.content = (f'<iframe srcdoc="{html.escape(page)}" title="运行命令" style="display:block;width:100%;'
                                      'height:34px;border:0;margin:-2px 0 2px"></iframe>')

    def _show_simulation(self):
        """The 项目 tab's physics engine and renderer (train.yaml env.train.init_params), from the backend's
        ``project.SIMULATION_CHOICES``; a choice is pending until saved, like scene edits."""
        if not hasattr(self, 'simulation_folder'):
            return
        from shared.assembly.editing import simulation_settings
        for handle in self.simulation_items:
            handle.remove()
        self.simulation_items = []
        gui = self.server.gui
        try:
            backend, choices, current = simulation_settings(self.target)
        except Exception as error:  # noqa: BLE001 - a project without a readable recipe
            with self.simulation_folder:
                self.simulation_items.append(gui.add_markdown(f'<small>读不到 train.yaml：{html.escape(str(error))}</small>'))
            return
        labels = {'physics': '物理引擎', 'renderer': '渲染（预览窗口与相机）'}
        with self.simulation_folder:
            if not choices:
                self.simulation_items.append(gui.add_markdown(f'<small>后端 `{backend}` 没有可切换的物理引擎或渲染器。</small>'))
                return
            for name, options in choices.items():
                value = self.pending['simulation'].get(name, current[name])
                names = {key: label for key, (label, _) in options.items()}
                dropdown = gui.add_dropdown(labels.get(name, name), list(names.values()), initial_value=names[value],
                                            hint='；'.join(f'{label}：{hint}' for label, hint in options.values()))
                self.simulation_items.append(dropdown)

                def changed(_, name=name, dropdown=dropdown, names=names, saved=current[name]):
                    if self.syncing:
                        return
                    self.checkpoint()
                    value = next(key for key, label in names.items() if label == dropdown.value)
                    if value == saved:
                        self.pending['simulation'].pop(name, None)
                    else:
                        self.pending['simulation'][name] = value
                    self.report(f'{labels.get(name, name)}：{dropdown.value}；保存后写入 train.yaml'
                                f'（env.train.init_params.{name}，评估与 view/record 同样使用）')
                dropdown.on_update(self.guarded(changed))
            self.simulation_items.append(gui.add_markdown(
                f'<small>后端 `{backend}`。训练、评估与 `run.sh view` 都按 train.yaml 运行；'
                '命令行 `+env.train.init_params.physics=…` 可临时覆盖。</small>'))

    def report(self, text, error=False):
        if hasattr(self, 'status'):
            shown = text if not self.backend_problem else f'{text}\n⚠️ {self.backend_problem}'
            shown = shown.replace('&', '&amp;').replace('<', '&lt;').replace('\n', '<br>')
            self.status.content = f'<div class="rl-status{" error" if error else ""}">{shown}</div>'
        if error:
            for client in self.server.get_clients().values():
                client.add_notification('操作未完成', text, color='red', auto_close_seconds=8)

    def _build_gui(self):
        gui = self.server.gui
        gui.configure_theme(dark_mode=False, show_logo=False, brand_color=(46, 110, 220))
        # Left: scene tree and tools. Right: the selection's properties, always visible.
        panel = gui.add_panel()
        with panel.add_tab('场景', icon='list-tree'):
            gui.add_markdown('<small>点击选中 · 双击聚焦 · 行尾开关：物理 / 渲染可见 / 编辑器中显示 · '
                             'Ctrl+S 保存 · Ctrl+Z 撤回</small>')
            self.tree_style = gui.add_html('')
            # The outliner (tools/scene_outliner.html, served with the asset library) replaces these buttons
            # once the library server runs; they remain the tree if it cannot start.
            self.outliner_frame = gui.add_html('')
            self.tree_folder = gui.add_folder(None)
            self.tree_items = []
        with panel.add_tab('添加', icon='plus'):
            self.place_style = gui.add_html('')
            # The shelf is a small page (tools/asset_shelf.html, served with the SimReady library) listing
            # every source: this backend's catalog, the user library and its imports, abstract entities
            # and the SimReady datasets. A card is dragged into the view, or chosen and placed by a click
            # in the view. The fields below hold what it does not show.
            self.shelf_frame = gui.add_html('<small>资产架启动中…</small>')
            self.gallery_entries, self.asset_choice, self.asset_variant, self.shelf_items = {}, None, None, []
            self.shelf_categories, self.shelf_detail, self.shelf_focus = [], None, False
            self.shelf_revision, self.shelf_items_revision, self.entity_name = 0, 0, 'new_object'
            self.mount_targets, self.mount_groups = {}, {}
            self.simready_entries, self.simready_problems, self.simready_loaded = {}, {}, False
            self.environment_entries = {}
            self.jobs, self.library, self.library_url, self.shelf_pictures = [], None, None, {}
            with gui.add_folder('选项 · 按坐标添加', expand_by_default=False):
                self.asset_mount = gui.add_dropdown('安装到', [NONE], hint='传感器随该实体的连杆运动')
                self.asset_mount_body = gui.add_dropdown('安装连杆', [ROOT_BODY])
                self.asset_mount.on_update(lambda _: self.syncing or self.guarded(self._offer_mount_bodies)())
                self.asset_mount_body.on_update(lambda _: self.syncing or self._shelf_changed())
                self.asset_size = gui.add_vector3('尺寸 (m)', (.5, .5, .5), min=(.001, .001, .001), step=.001,
                                                  hint='全长；矩形取前两项')
                self.asset_parameters = gui.add_text('参数 (JSON)', '{}', multiline=True)
                self.asset_placement = gui.add_dropdown('初始位置', ['资产默认初始位置', '指定位置'])
                self.asset_position = gui.add_vector3('位置 (m)', (0.0, 0.0, 0.0), step=.001)
                gui.add_button('按坐标添加选中的资产', icon='plus').on_click(lambda _: self._add_asset())
            with gui.add_folder('更多', expand_by_default=False):
                self.simready_status = gui.add_markdown('<small>SimReady：正在读取目录…</small>')
                gui.add_button('打开完整 SimReady 资产库', icon='layout-grid').on_click(
                    lambda event: self._open_library(event.client))
                self.simready_link = gui.add_markdown('')
                gui.add_button('生成缺失示意图（Isaac 后台）', icon='photo').on_click(lambda _: self.start_thumbnails())
            with gui.add_folder('导入 URDF / USD', expand_by_default=False):
                gui.add_markdown('<small>登记到资产库（assets/library.json），所有项目的陈列栏都可使用；'
                                 '文件留在原处，建议放在工作区内。</small>')
                self.import_path = gui.add_text('文件路径', '', hint='.urdf / .usd / .usda / .usdc；相对路径按工作区根目录解析')
                self.import_kind = gui.add_dropdown('类型', list(IMPORT_KINDS))
                self.import_label = gui.add_text('名称', '', hint='陈列栏中显示的名称；留空用文件名')
                self.import_fixed = gui.add_checkbox('固定底座', True, hint='机械臂通常固定；移动机器人取消')
                gui.add_button('导入到资产库', icon='file-import').on_click(lambda _: self._import_asset())
            with gui.add_folder('布局组合', expand_by_default=False):
                gui.add_markdown('<small>仅编辑器内一起摆放，不产生物理约束。</small>')
                self.group_name = gui.add_text('组合名称', 'workstation')
                self.group_folder = gui.add_folder(None)
                self.group_boxes = {}
                gui.add_button('组成布局组合', icon='stack-2').on_click(lambda _: self._create_group())
        with panel.add_tab('观察', icon='eye'):
            # Rebuilt per scene: training switch, then one page of input checkboxes per module.
            self.observation_folder = gui.add_folder(None)
            self.observation_items, self.observation_tabs, self.observation_boxes = [], {}, {}
            self.period_fields, self.dimension_fields = {}, {}
        with panel.add_tab('控制', icon='steering-wheel'):
            # Like the observation tab: one folder per target, its joints as checkboxes.
            self.control_folder = gui.add_folder(None)
            self.control_items = []
            self.target_add = gui.add_folder('＋ 新增控制目标', expand_by_default=False)
            with self.target_add:
                self.target_asset = gui.add_dropdown('机器人 (asset_name)', [NONE])
                self.target_type = gui.add_dropdown('类型', list(CONTROL_TYPES.values()))
                self.target_name = gui.add_text('名称', 'arm')
                gui.add_button('添加（勾选尚未控制的全部关节）', icon='plus').on_click(lambda _: self._add_target())
            gui.add_markdown('关节逐个列出，勾选顺序即动作维度顺序；实际单位目标，缩放与偏移在 policy.py 中计算。'
                             '增删终点后请同步 decode_action 的返回值。保存后写入 scene.yaml。')
            self.target_asset.on_update(lambda _: None if self.syncing else self._suggest_target_name())
        with panel.add_tab('项目', icon='folder'):
            # Physics engine and renderer: run-time choices of train.yaml (_show_simulation).
            self.simulation_folder = gui.add_folder('仿真（train.yaml）')
            self.simulation_items = []
            gui.add_button('放弃修改并重新加载', icon='refresh').on_click(lambda _: self.reload())
            gui.add_button('刷新原生网格（Isaac 后台导出）', icon='cube').on_click(lambda _: self._refresh_native())
            with gui.add_folder('新建 / 打开环境'):
                from tools.create import MAINTAINED_TEMPLATES
                self.new_name = gui.add_text('新项目名称', 'new_environment')
                self.new_template = gui.add_dropdown('模板（同 run.py new）', MAINTAINED_TEMPLATES)
                gui.add_button('新建环境').on_click(lambda _: self._new_environment())
                self.open_path = gui.add_text('scene.yaml 路径', '')
                gui.add_button('打开').on_click(lambda _: self._open_environment())
        panel.dock_left()
        panel.set_width(340)
        main = gui.main_panel
        main.dock_right()
        main.set_width(340)
        theme, self.axes_style = gui.add_html(''), gui.add_html('')
        self.save_button = gui.add_button('已保存', icon='device-floppy', color='gray')
        self.save_button.on_click(lambda _: self.save())
        self.undo_button = gui.add_button('', icon='arrow-back-up', color='gray', hint='撤回 Ctrl+Z', disabled=True)
        self.undo_button.on_click(lambda _: self.undo_edit())
        self.redo_button = gui.add_button('', icon='arrow-forward-up', color='gray', hint='重做 Ctrl+Shift+Z', disabled=True)
        self.redo_button.on_click(lambda _: self.redo_edit())
        # Save, undo and redo share one toolbar row.
        # viser gives each button wrapper ~9px side margins; keep the outer ones, 4px gaps inside.
        toolbar = ''.join(f'div:has(> [id="{button._impl.uuid}"]){{display:inline-block;vertical-align:top;'
                          f'width:{width};margin-left:{left}!important;margin-right:{right}!important}}'
                          for button, width, left, right in ((self.save_button, 'calc(100% - 104px)', '9.25px', '4px'),
                                                             (self.undo_button, '38px', '0', '4px'),
                                                             (self.redo_button, '38px', '0', '0')))
        toolbar += ''.join(f'[id="{button._impl.uuid}"]{{padding:0!important}}' for button in (self.undo_button, self.redo_button))
        theme.content = '<style>' + THEME_CSS + toolbar + '</style>'
        self.command_strip = gui.add_html('')  # `run.sh view <project>` with a copy button (_show_command)
        self.status = gui.add_markdown('<div class="rl-status">加载中…</div>')
        self.inspector = gui.add_folder(None)
        for label, hotkey, modifier, callback, icon in (
                ('保存到 scene.yaml', 'S', 'cmd/ctrl', lambda: self.save(), 'device-floppy'),
                ('撤回', 'Z', 'cmd/ctrl', lambda: self.undo_edit(), 'arrow-back-up'),
                ('重做', 'Z', 'cmd/ctrl+shift', lambda: self.redo_edit(), 'arrow-forward-up'),
                ('取消预览 / 取消选择', 'escape', None, lambda: self.escape(), 'x'),
                ('聚焦选中对象', 'F', None, lambda: self.focus_selection(), 'focus-2'),
                ('隐藏 / 显示选中对象', 'H', None, lambda: self.toggle_hidden(), 'eye-off')):
            gui.add_command(label, hotkey=hotkey, modifier=modifier, icon=icon).on_trigger(
                lambda _, callback=callback: callback())

    def _refresh_lists(self):
        self._rebuild_tree_gui()
        self._rebuild_group_choices()
        self._refresh_assets()
        self._refresh_control()

    # -------------------------------------------------------------- scene tree
    def _tree_parent(self, owner, members):
        kind, name = owner
        if kind != 'entity':
            return None
        item, info = self.items[name], self.info[name]
        if info['fixed'] and info['fixed'].get('parent'):
            return ('entity', info['fixed']['parent']['entity'])
        if item.get('mount'):
            target = _resolve(name, item['mount']['entity'])
            if target in self.items:
                return ('entity', target)
        if info['layout_parent']:
            return ('entity', info['layout_parent'])
        if name in members:
            return ('group', members[name])
        return ('entity', item['parent']) if item['parent'] in self.items else None

    def tree_entries(self):
        """(owner, depth): physical, mount, layout and YAML parents nest their children."""
        members = {path: name for name, group in self.snapshot.get('layout_groups', {}).items()
                   for path in group['entities'] if name in self.groups}
        owners = [('entity', path) for path in self.items] + [('group', name) for name in self.groups]
        children, roots = {}, []
        for owner in owners:
            parent = self._tree_parent(owner, members)
            (children.setdefault(parent, []) if parent in owners else roots).append(owner)
        rows, seen = [], set()

        def walk(owner, depth):
            if owner in seen:
                return
            seen.add(owner)
            rows.append((owner, depth))
            for child in children.get(owner, []):
                walk(child, depth + 1)
        for owner in roots + owners:  # Owners left over from a cycle fall back to the top level.
            walk(owner, 0)
        return rows

    def _row_text(self, owner, changed):
        kind, name = owner
        if kind == 'entity':
            item, info = self.items[name], self.info[name]
            text = (self.labels.get(name) or item['label']).strip() or item['name']
            if info['fixed']:
                text += '  · 固定'
            elif item.get('mount'):
                text += '  · 安装'
            elif info['layout_parent']:
                text += '  · 布局'
        else:
            text = name
        if owner in self.hidden:
            text += '  · 隐藏'
        return text + ('  •' if owner in changed else '')

    def _row_icon(self, owner):
        kind, name = owner
        if kind != 'entity':
            return ICONS[kind]
        item = self.items[name]
        return 'camera' if item.get('factory') == 'camera' else ICONS.get(item['category'], 'box')

    def _rebuild_tree_gui(self):
        gui = self.server.gui
        for handle in self.tree_items:
            handle.remove()
        self.tree_items, self.tree_rows = [], {}
        self.outliner_revision += 1
        if self.outliner_page:  # the outliner page lists the tree (scene_outliner.html)
            return
        changed = self.changed_owners()
        entries = self.tree_entries()
        with self.tree_folder:
            for owner, depth in entries:
                kind, name = owner
                hint = f'{name} · {self.items[name]["category_label"]}' if kind == 'entity' else '布局组合（仅摆放）'
                row = gui.add_button(self._row_text(owner, changed), icon=self._row_icon(owner), hint=hint)
                row.on_click(lambda _, owner=owner: self.select(owner))
                self.tree_items.append(row)
                self.tree_rows[owner] = (row, depth)
            if not entries:
                self.tree_items.append(gui.add_markdown('场景为空：在“添加”中加入地面、机器人和物体。'))
        self._style_tree()

    def _style_tree(self):
        self.outliner_revision += 1
        rules = []
        for owner, (row, depth) in self.tree_rows.items():
            selected = owner == self.selected
            rules.append(TREE_ROW.format(id=row._impl.uuid, pad=6 + TREE_INDENT * depth,
                                         bg='rgba(255,170,40,.28)' if selected else 'transparent',
                                         hover='rgba(255,170,40,.36)' if selected else 'rgba(128,128,128,.14)',
                                         weight=600 if selected else 400))
        self.tree_style.content = '<style>' + ''.join(rules) + '</style>'

    def _label_tree(self, changed=None):
        changed = self.changed_owners() if changed is None else changed
        if changed != self.tree_changed:
            self.tree_changed = changed
            self.outliner_revision += 1
        for owner, (row, _) in self.tree_rows.items():
            text = self._row_text(owner, changed)
            if row.label != text:
                row.label = text

    def _watch_changes(self):
        """Keep the save button and tree markers in step with unsaved edits."""
        while True:
            time.sleep(.8)
            try:
                with self.lock:
                    changed, dirty = self.changed_owners(), self.unsaved()
                    self._label_tree(changed)
                    label, color = ('保存到 scene.yaml', 'green') if dirty else ('已保存', 'gray')
                    if self.save_button.label != label:
                        self.save_button.label, self.save_button.color = label, color
                    if self.undo_button.disabled != (not self.undo):
                        self.undo_button.disabled = not self.undo
                    if self.redo_button.disabled != (not self.redo):
                        self.redo_button.disabled = not self.redo
            except Exception:  # noqa: BLE001 - cosmetic; errors surface on the next action
                pass

    # ----------------------------------------------------------------- inspector
    def _clear_inspector(self):
        for handle in self.inspector_items:
            handle.remove()
        self.inspector_items, self.pose_fields, self.hidden_box = [], None, None
        self.camera_image = None

    def _add(self, handle):
        self.inspector_items.append(handle)
        return handle

    def _section(self, title):
        return self._add(self.server.gui.add_folder(title, expand_by_default=self.open_section == title))

    def _rebuild_inspector(self):
        self._clear_inspector()
        gui, owner = self.server.gui, self.selected
        with self.inspector:
            if owner is None:
                self._add(gui.add_markdown('在左侧场景树或 3D 视图中点击对象，即可在这里编辑位置、安装关系、'
                                           '初始状态和复位。\n\n<small>拖动坐标轴移动；数值框可精确输入。</small>'))
                return
            node = self._node_of(owner)
            kind, name = owner
            if kind == 'entity':
                item, info = self.items[name], self.info[name]
                factory = '' if item['category'] == 'abstract' else f' · `{item.get("factory") or "group"}`'
                self._add(gui.add_markdown(f'### {item["label"]}\n{item["category_label"]}{factory}'
                                           f' · `{name}`  \n{self._relation_text(item, info)}'
                                           + (f'  \n<small>{info["note"]}</small>' if info['note'] else '')))
                label = self._add(gui.add_text('显示名称', self.labels.get(name, item['label'])))
                label.on_update(lambda _: (self.checkpoint(('label', name)), self.labels.__setitem__(name, label.value)))
                if item.get('factory') == 'camera':
                    width, height, _ = self.camera_optics(item)
                    blank = np.full((240, round(240 * width / height), 3), 225, np.uint8)
                    self.camera_image = self._add(gui.add_image(blank, label='相机视角', format='jpeg'))
                    self.camera_caption = self._add(gui.add_markdown('<small>渲染中…</small>'))
                    self._add(gui.add_button('刷新相机视角', icon='refresh')).on_click(lambda _: self.request_camera_view())
                    self.request_camera_view()
            else:
                self._add(gui.add_markdown(f'### {name}\n布局组合 · 仅编辑器摆放，不产生物理约束'))
            locked = self._locked(owner)
            position = self._add(gui.add_vector3('位置 (m)', tuple(round(v, 5) for v in _xyz(node.local)), step=.001,
                                                 disabled=locked))
            rotation = self._add(gui.add_vector3('旋转 XYZ (°)', tuple(round(v, 3) for v in F.matrix_euler(node.local)),
                                                 step=.1, disabled=locked))
            self.pose_fields = (position, rotation)
            position.on_update(self._pose_field_changed)
            rotation.on_update(self._pose_field_changed)
            self.hidden_box = self._add(gui.add_checkbox('在编辑器中隐藏', owner in self.hidden,
                                                         hint='仅影响网页显示（快捷键 H），不写入 scene.yaml'))
            self.hidden_box.on_update(lambda _: None if self.syncing else self.toggle_hidden(owner))
            if kind == 'entity':
                self._entity_controls(name)
            else:
                self._add(gui.add_button('解散布局组合（保留位置）', icon='stack-pop')).on_click(lambda _: self._ungroup(name))

    def _relation_text(self, item, info):
        if info['fixed']:
            parent = info['fixed']['parent']
            return f'物理固定于 `{parent["entity"]}` · {parent.get("body") or ROOT_BODY}（连接 {info["fixed"]["name"]}）'
        if item.get('mount') and item['category'] == 'abstract':
            return f'跟随 `{_resolve(item["path"], item["mount"]["entity"])}` 根部（仅读数，不是物理约束）'
        if item.get('mount'):
            return f'安装于 `{_resolve(item["path"], item["mount"]["entity"])}` · {item["mount"].get("body") or ROOT_BODY}（随连杆运动）'
        if info['layout_parent']:
            return f'共同摆放于 `{info["layout_parent"]}`（仅布局）'
        if info['grouped']:
            return '布局组合成员（数值相对组合）'
        return '数值相对父级' if item['parent'] in self.items else '世界坐标'

    def _entity_controls(self, path):
        gui, item, info = self.server.gui, self.items[path], self.info[path]
        if item['category'] == 'abstract':
            self._abstract_controls(path)
            self._attach_section(path)
            if not item.get('mount'):
                if (self.current_motion(path) or {}).get('type') != 'path':  # a path decides the reset pose
                    self._reset_section(path)
                self._motion_section(path)
            self._add(gui.add_button('删除抽象体', color='red', icon='trash')).on_click(lambda _: self._delete(path))
            return
        with self._section('参数'):
            schema = self.asset_entry(item['factory']).get('parameters', {})
            if schema and not item.get('parameters', {}).keys() - schema.keys():
                self._parameter_fields(path, schema)
            else:  # Factories taking **kwargs, or authored keys outside the signature.
                text = self.parameters.get(path, json.dumps(item.get('parameters', {}), ensure_ascii=False, indent=1))
                parameters = gui.add_text('参数 (JSON)', text, multiline=True, hint='几何相关参数修改后需刷新原生网格')
                parameters.on_update(lambda _: (self.checkpoint(('parameters', path)),
                                                self.parameters.__setitem__(path, parameters.value)))
        self._attach_section(path)
        if item.get('factory') and item['category'] in ('robot', 'object'):
            self._initial_section(path)
        if item['category'] not in ('sensor', 'terrain', 'light') and not info['fixed'] \
                and (self.current_motion(path) or {}).get('type') != 'path':  # a path decides the reset pose
            self._reset_section(path)
        if item['category'] == 'object' and item.get('scope') != 'global' and not info['fixed'] and not item.get('mount'):
            self._motion_section(path)
        if item.get('meshes') or 'height_scan' in self.asset_entry(item['factory']).get('capabilities', []):
            self._meshes_section(path)
        self._add(gui.add_button('删除实体', color='red', icon='trash')).on_click(lambda _: self._delete(path))

    def _parameter_fields(self, path, schema):
        """One field per factory parameter, typed by the catalog; edits the same pending parameters as the JSON form."""
        gui, item = self.server.gui, self.items[path]
        values, self.parameter_fields = self.parameter_values(item), {}

        def store(name, value):
            self.checkpoint(('parameters', path, name))
            try:
                current = json.loads(self.parameters[path]) if path in self.parameters else dict(item.get('parameters', {}))
            except ValueError:
                current = dict(item.get('parameters', {}))
            current[name] = value
            self.parameters[path] = json.dumps(current, ensure_ascii=False)
            if item.get('factory') == 'camera':
                self.request_camera_view()

        def parsed(name, text, convert):
            try:
                store(name, convert(text))
            except ValueError as error:
                self.report(f'{schema[name]["label"]}：{error}', error=True)

        for name, field in schema.items():
            label, kind, value = field['label'], field['type'], values.get(name)
            names = field.get('option_labels', {})
            if kind == 'choices':
                chosen = set(value or ())
                gui.add_markdown(f'<small>{label}</small>')
                boxes = {}
                for option in field['options']:
                    boxes[option] = gui.add_checkbox(names.get(option, option), option in chosen)
                    boxes[option].on_update(lambda _, n=name, b=boxes, o=field['options']: store(
                        n, [option for option in o if b[option].value]))
                self.parameter_fields[name] = boxes
                continue
            mounted = _resolve(path, item['mount']['entity']) if field.get('mount_body') and item.get('mount') else None
            if mounted in self.info and value in self.bodies(mounted):
                handle = gui.add_dropdown(label + '（连杆）', self.bodies(mounted), initial_value=value)
                handle.on_update(lambda _, n=name, h=handle: store(n, h.value))
            elif kind == 'choice' and value in field['options']:
                handle = gui.add_dropdown(label, [names.get(o, o) for o in field['options']],
                                          initial_value=names.get(value, value))
                handle.on_update(lambda _, n=name, h=handle, f=field: store(
                    n, next(o for o in f['options'] if f['option_labels'].get(o, o) == h.value)))
            elif kind == 'boolean' and isinstance(value, bool):
                handle = gui.add_checkbox(label, value)
                handle.on_update(lambda _, n=name, h=handle: store(n, bool(h.value)))
            elif kind in ('integer', 'number') and not field.get('nullable') and isinstance(value, (int, float)):
                handle = gui.add_number(label, value, step=1 if kind == 'integer' else .001)
                handle.on_update(lambda _, n=name, h=handle, k=kind: store(n, int(h.value) if k == 'integer' else float(h.value)))
            elif kind == 'vector' and field['length'] in (2, 3) and isinstance(value, (list, tuple)) and len(value) == field['length']:
                handle = (gui.add_vector2 if field['length'] == 2 else gui.add_vector3)(label, tuple(map(float, value)), step=.001)
                handle.on_update(lambda _, n=name, h=handle: store(n, [float(v) for v in h.value]))
            elif kind in ('integer', 'number'):  # Nullable: empty text is null.
                handle = gui.add_text(label, '' if value is None else str(value), hint='留空为 null')
                handle.on_update(lambda _, n=name, h=handle, k=kind: parsed(
                    n, h.value.strip(), lambda t, k=k: None if not t else int(t) if k == 'integer' else float(t)))
            elif kind == 'text' and isinstance(value, str):
                handle = gui.add_text(label, value)
                handle.on_update(lambda _, n=name, h=handle: store(n, h.value))
            else:  # Structured or unset values stay JSON.
                handle = gui.add_text(label + ' (JSON)', json.dumps(value, ensure_ascii=False), multiline=True,
                                      hint='必填' if field.get('required') else None)
                handle.on_update(lambda _, n=name, h=handle: parsed(n, h.value, json.loads))
            self.parameter_fields[name] = handle
        gui.add_markdown('<small>参数在“保存到 scene.yaml”时校验；几何相关参数修改后需刷新原生网格。</small>')

    def _abstract_controls(self, path):
        gui = self.server.gui
        shape, size = self.abstract_geometry(path)
        with self._section('形状'):
            pick = gui.add_dropdown('形状', list(ABSTRACT_SHAPES.values()), initial_value=ABSTRACT_SHAPES[shape])
            field = None
            if shape in ('rectangle', 'box'):
                field = gui.add_vector3('尺寸 (m)', tuple(size), min=(.001,) * 3, step=.001) if len(size) == 3 else \
                    gui.add_vector2('尺寸 (m)', tuple(size), min=(.001,) * 2, step=.001)
            gui.add_markdown('<small>矩形/长方体为全长；方向读取本地 +X 轴。观察字段：position、orientation'
                             + ('、size' if field else '、direction' if shape == 'direction' else '') + '。</small>')
        saved = self.snapshot.get('visualization', {}).get('abstract', [])
        shown = self.pending['shown'] if self.pending['shown'] is not None else saved
        visible = self._add(gui.add_checkbox('仿真中显示', self.items[path]['name'] in shown,
                                             hint='写入 visualization.abstract；Isaac 预览/评估时绘制标记'))

        @self.guarded
        def reshape(_):
            new = next(key for key, label in ABSTRACT_SHAPES.items() if label == pick.value)
            if new == shape and field is not None:
                values = [float(v) for v in field.value]
            else:
                # Keep the current footprint when switching between areas.
                values = {'point': [], 'direction': [], 'rectangle': (list(size) + [.5, .5])[:2],
                          'box': (list(size) + [.5, .5, .5])[:3]}[new]
            self.checkpoint(('shape', path))
            self.pending['shapes'][path] = (new, values)
            self._abstract_shape(path)
            self._draw_shape(self.info[path]['node'], highlight=True)
            if new != shape:
                self.open_section = '形状'
                self._rebuild_inspector()
        pick.on_update(reshape)
        if field is not None:
            field.on_update(reshape)

        @self.guarded
        def toggle(_):
            if self.syncing:
                return
            self.checkpoint()
            name = self.items[path]['name']
            current = self.pending['shown'] if self.pending['shown'] is not None else saved
            names = [n for n in current if n != name] + ([name] if visible.value else [])
            self.pending['shown'] = names
        visible.on_update(toggle)

    # ---------------------------------------------------------------- entities
    def deletion_plan(self, path):
        """Deleting keeps dependants where they are: scene-owned children move up a level,
        attached or mounted entities are detached; only sensors measuring the deleted body go with it."""
        item = self.items[path]
        promoted = [child for child, entry in self.items.items() if entry['parent'] == path
                    and item['instance_component'] is None and entry['instance_component'] is None]
        removed = {other for other in self.items if (other == path or other.startswith(path + '/'))
                   and not any(other == kept or other.startswith(kept + '/') for kept in promoted)}
        detached, dropped, released, unplaced, retargeted = [], [], [], [], []
        for other, entry in self.items.items():
            if other in removed:
                continue
            meshes = self.pending['meshes'].get(other, entry.get('meshes', []))
            if any(_resolve(other, ref) in removed for ref in meshes):
                retargeted.append(other)  # SceneDocument drops the ray targets that go away.
            reset = self.pending['resets'][other] if other in self.pending['resets'] else entry.get('reset')
            if (reset or {}).get('region') in removed:
                unplaced.append(other)  # Its placement region goes away; it stays at its authored pose.
            info = self.info[other]
            mounted = bool(entry.get('mount')) and _resolve(other, entry['mount']['entity']) in removed
            joint = info['joint']
            fixed = bool(joint and joint.get('parent')) and joint['parent']['entity'] in removed
            if fixed and not info['fixed']:
                raise ValueError(f'{other} 的连杆坐标尚未就绪，无法确定保留位置；请稍候再删除')
            if mounted and _body_bound(entry['category'], entry.get('factory'), self.scene):
                dropped.append(other)  # Contact sensors and ray casters only exist on their body.
            elif mounted or fixed:
                if fixed and not info['fixed']['project']:
                    raise ValueError(f'{other} 由模板内部连接固定；请在模板中编辑')
                detached.append(other)
            elif info['layout_parent'] in removed:
                released.append(other)
        return {'removed': sorted(removed), 'promoted': promoted, 'detached': detached,
                'dropped': dropped, 'released': released, 'unplaced': unplaced, 'retargeted': retargeted}

    @action
    def _delete(self, path):
        plan, label = self.deletion_plan(path), self.items[path]['label']
        names = lambda paths: '、'.join(f'`{self.items[p]["label"]}`' for p in paths)
        lines = [f'删除 **{label}**？']
        parts = [p for p in plan['removed'] if p != path]
        if parts or plan['dropped']:
            lines.append(f'一起删除：{names(parts + plan["dropped"])}')
        if plan['promoted']:
            lines.append(f'上移一级、保持原位：{names(plan["promoted"])}（观察字段名随路径变化，'
                         'policy.py / task.py 中的引用需同步）')
        if plan['detached'] or plan['released']:
            lines.append(f'解除依附、保持原位：{names(plan["detached"] + plan["released"])}')
        if plan['unplaced']:
            lines.append(f'不再在该区域内随机复位、固定在当前位置：{names(plan["unplaced"])}')
        effects = SceneDocument(self.scene).removal_effects(set(plan['removed']) | set(plan['dropped']))
        if effects['outputs']:
            lines.append('删除控制目标：' + '、'.join(f'`{name}`' for name in effects['outputs']) +
                         '（policy.py 的 decode_action 需同步；没有控制目标时无法训练）')
        if effects['observations']:
            lines.append('移除观察：' + '、'.join(f'`{term}`' for term in effects['observations']) +
                         '（policy.py / task.py 中对应的 obs 属性需同步删除）')
        if plan['retargeted']:
            gone = set(plan['removed'])
            blind = [other for other in plan['retargeted'] if all(
                _resolve(other, ref) in gone for ref in self.pending['meshes'].get(other, self.items[other].get('meshes', [])))]
            kept = [other for other in plan['retargeted'] if other not in blind]
            if kept:
                lines.append(f'不再以其为射线目标：{names(kept)}')
            if blind:
                lines.append(f'⚠️ {names(blind)} 将没有射线目标，后端检查不通过、无法训练；'
                             '之后添加的地面会自动成为它的目标，也可在检查器“射线目标”中选择')
        lines.append('可用 Ctrl-Z 撤回。')
        modal = self.server.gui.add_modal('删除实体')
        with modal:
            self.server.gui.add_markdown('\n\n'.join(lines))
            self.server.gui.add_button('删除', color='red', icon='trash').on_click(
                lambda _: (modal.close(), self._remove(path, plan)))
            self.server.gui.add_button('取消', icon='x').on_click(lambda _: modal.close())

    @action
    def _remove(self, path, plan):
        payload = self.collect()
        for other in plan['detached']:
            item, info = self.items[other], self.info[other]
            logical = self.info.get(item['parent'])
            world = info['node'].world()
            changes = {'pose': F.matrix_pose(np.linalg.inv(logical['node'].world()) @ world if logical else world)}
            if item.get('mount'):
                changes['mount'] = None
            else:
                payload.setdefault('connections', {})[info['fixed']['name']] = None
            payload['entities'].append({'path': other, 'changes': changes})
        for other in plan['unplaced']:
            entry = next((e for e in payload['entities'] if e['path'] == other), None)
            if entry is None:
                payload['entities'].append(entry := {'path': other, 'frame': 'local', 'changes': {}})
            entry['changes']['reset'] = None
        for entry in payload['entities']:
            if entry['changes'].get('meshes'):  # Staged targets lose the removed entities too.
                entry['changes']['meshes'] = [ref for ref in entry['changes']['meshes']
                                              if _resolve(entry['path'], ref) not in plan['removed']] or None
        payload['remove_entities'] = [path, *plan['dropped']]
        self.selected = None
        self._apply(payload)
        kept = len(plan['promoted']) + len(plan['detached']) + len(plan['released']) + len(plan['unplaced'])
        self.report(f'已删除 {path}' + (f'；{kept} 个相关实体保持原位' if kept else ''))

    @action
    def _detach(self, path):
        world = self.info[path]['node'].world()
        layout = bool(self.info[path]['layout_parent'])
        self._save_first()
        item, info = self.items[path], self.info[path]
        parent = self.info.get(item['parent'])
        pose = F.matrix_pose(np.linalg.inv(parent['node'].world()) @ world if parent else world)
        payload = {'revision': self.snapshot['revision'], 'entities': [{'path': path, 'changes': {'pose': pose}}]}
        if layout:
            payload['layout_parents'] = {path: None}
        elif item.get('mount'):
            payload['entities'][0]['changes']['mount'] = None
        elif info['fixed'] and info['fixed']['project']:
            payload['connections'] = {info['fixed']['name']: None}
        else:
            raise ValueError('没有可解除的项目级连接；模板内部连接请编辑模板')
        self.open_section = '父物体'
        self._apply(payload)
        self.report('已解除，世界位置保留')

    @action
    def _add_asset(self, position=None):
        return self._add_choice(position)

    def _add_choice(self, position=None):
        """Add the shelf's choice: at `position` (a click or drop in the view), else as the form says."""
        key = self._simready_choice()
        if key:
            if position is None:
                position = list(self.asset_position.value) if self.asset_placement.value == '指定位置' else [0, 0, 0]
            return self.library_add(key, self.entity_name, position)
        if (self.asset_choice or '').startswith('environment:'):
            return self._environment_add(self.asset_choice.split(':', 1)[1], self.entity_name)
        choice = self.gallery_entries.get(self.asset_choice)
        if not choice:
            raise ValueError('请在资产架中选择资产')
        if choice.get('shape'):
            return self._add_abstract(choice['shape'], position)
        factory = choice['factory']
        parameters = json.loads(self.asset_parameters.value or '{}')
        if not isinstance(parameters, dict):
            raise ValueError('参数必须是 JSON 对象')
        self._save_first()
        entry = next(item for item in self.snapshot['catalog']['assets'] if item['id'] == factory)
        category = entry.get('category', 'object')
        name = self.entity_name
        entity = {'name': name, 'factory': factory, 'kind': category if category in ('robot', 'sensor') else 'entity',
                  'parameters': parameters}
        if category == 'sensor' and self.asset_mount.value != FREE:
            entity['mount'] = self._new_mount()
            for key, field in entry.get('parameters', {}).items():
                if field.get('mount_body') and key not in parameters:  # e.g. the frame a transformer reports.
                    parameters[key] = self._root_body(entity['mount']['entity'])
        elif position is not None or self.asset_placement.value == '指定位置':
            entity['pose'] = {'position': list(position or self.asset_position.value), 'orientation': [1, 0, 0, 0]}
        self._add_entity(entity, entry)

    def _add_entity(self, entity, entry):
        retargeted = self._aim_rays(entity, 'height_scan' in entry.get('capabilities', []))
        self._apply({'revision': self.snapshot['revision'], 'add_entities': [entity], 'entities': retargeted})
        added = next((item for item in self.snapshot['entities']
                      if item['name'] == entity['name'] and item['factory'] == entity['factory']), None)
        if added:
            self.select(('entity', added['path']))
            if self.visual(added)[0] is None and added['category'] in ('robot', 'object', 'terrain'):
                self.start_export()
        self.report(f'已添加 {entity["name"]}')
        self._describe_asset()  # a fresh name for the next one

    def _offer_mounts(self):
        """Robots and objects a new sensor can go on; the selected one, else the first robot.

        Only a camera may also stand fixed in the world (some backends still require a body).
        A static object (a table, a box without mass) has no body once its native export says so."""
        def static(path):
            return self.items[path].get('exported') and not self.initial_display(path).get('root_body')
        self.mount_targets = {label: path for label, path in self._choices(None).items()
                              if self.items[path]['category'] in ('robot', 'object') and not static(path)}
        selected = self.selected[1] if self.selected and self.selected[0] == 'entity' else None
        ranked = sorted(self.mount_targets, key=lambda label: (self.mount_targets[label] != selected,
                                                               self.items[self.mount_targets[label]]['category'] != 'robot'))
        choice = self.gallery_entries.get(self.asset_choice) or {}
        options = [*ranked, FREE] if not _body_bound('sensor', choice.get('factory'), self.scene) else ranked or [NONE]
        self.syncing += 1
        try:
            self.asset_mount.options = options
            self.asset_mount.value = options[0]
        finally:
            self.syncing -= 1
        self._offer_mount_bodies()

    def _offer_mount_bodies(self):
        target = self.mount_targets.get(self.asset_mount.value)
        bodies = self.bodies(target) if target else []
        choice = self.gallery_entries.get(self.asset_choice) or {}
        # Contact sensors may cover every link sharing a name suffix (the four feet).
        self.mount_groups = _body_groups(bodies) if 'contact_forces' in choice.get('entry', {}).get('capabilities', []) else {}
        self.syncing += 1
        try:
            self.asset_mount_body.options = [ROOT_BODY, *bodies, *self.mount_groups]
            self.asset_mount_body.value = ROOT_BODY
        finally:
            self.syncing -= 1
        self.asset_mount_body.visible = target is not None
        # Mounted: the pose is an offset on the body, set afterwards in the inspector.
        self.asset_placement.visible = self.asset_position.visible = target is None
        self._shelf_changed()

    def _root_body(self, target):
        body = self.initial_display(target).get('root_body', '')
        if not body:
            busy = self.export_thread and self.export_thread.is_alive()
            raise ValueError(f'`{target}` 的连杆名尚未解析' +
                             ('：原生网格正在导出，完成后再添加' if busy else '：请先“刷新原生网格”，再添加'))
        return body

    def _new_mount(self):
        """The add panel's 安装到 / 安装连杆 choice for a sensor that only exists on a body."""
        target = self.mount_targets.get(self.asset_mount.value)
        if target is None:
            raise ValueError('该传感器只能装在机器人或物体的连杆上；请先添加机器人或物体')
        body = self.asset_mount_body.value
        return {'entity': target, 'body': self._root_body(target) if body == ROOT_BODY else self.mount_groups.get(body, body)}

    def _aim_rays(self, entity, casts):
        """Ray targets that keep covering the ground once ``entity`` is added.

        Ray sensors hit global entities with geometry (grounds, terrains): a new one aims at all of
        them, and a sensor already aiming at all of them also aims at a new ground."""
        def ground(snapshot):
            return {entry['path'] for entry in snapshot['catalog']['entities'] if entry.get('scope') == 'global'
                    and entry.get('category') not in ('sensor', 'light', 'abstract', 'group')}
        before = ground(self.snapshot)
        if casts:
            entity['meshes'] = sorted(before)
            return []
        trial = SceneDocument(self.scene)
        added = trial.add_entity(**entity)
        if added not in ground(trial.snapshot()):
            return []
        rays = {entry['path'] for entry in self.snapshot['catalog']['entities'] if 'height_scan' in entry.get('capabilities', [])}
        return [{'path': item['path'], 'changes': {'meshes': [*item.get('meshes', []), added]}}
                for item in self.snapshot['entities'] if item['path'] in rays
                and {_resolve(item['path'], ref) for ref in item.get('meshes', [])} == before]

    def _gallery(self):
        """Catalog assets of this scene plus user-library presets of shared factories."""
        catalog = {entry['id']: entry for entry in self.snapshot['catalog'].get('assets', []) if entry.get('available', True)}
        entries = {}
        for factory, entry in catalog.items():
            if factory in IMPORT_FACTORIES or not entry.get('on_shelf', True):  # Imports appear as library presets.
                continue
            entries[factory] = {'label': entry.get('label', factory), 'factory': factory, 'parameters': None,
                                'category': entry.get('category', 'object'), 'entry': entry,
                                'key': thumbnail_key(factory, {})}
        for item in load_library() + terrain_tiles() + recorded_assets():
            entry = catalog.get(item.get('factory'))
            if entry:
                if item.get('note'):
                    entry = {**entry, 'description': item['note']}
                if item.get('category') in CATEGORY_LABELS:  # e.g. a terrain tile spawned by usd_object
                    entry = {**entry, 'category_label': CATEGORY_LABELS[item['category']]}
                entries['library:' + item['id']] = {
                    'label': item.get('label') or item['id'], 'factory': item['factory'], 'library': item,
                    'parameters': item.get('parameters', {}), 'category': item.get('category') or entry.get('category', 'object'),
                    'entry': entry, 'key': item.get('thumbnail') or thumbnail_key(item['factory'], item.get('parameters', {}))}
        for shape, label in ABSTRACT_SHAPES.items():
            # Abstract entities: no model or physics, listed alongside assets.
            entries['abstract:' + shape] = {
                'label': label, 'factory': None, 'parameters': None, 'category': 'abstract', 'shape': shape,
                'key': 'abstract_' + shape, 'icon': ABSTRACT_ICONS[shape],
                'entry': {'description': '抽象体 · 无模型、无物理；' + ABSTRACT_NOTES[shape], 'category_label': '抽象体'}}
        return entries

    def _refresh_assets(self):
        """Rebuild the shelf's items and categories (scene catalog, library, SimReady); the shelf page refetches."""
        self.gallery_entries = self._gallery()
        items = []
        for choice, entry in self.gallery_entries.items():
            source = (entry.get('library') or {}).get('simready')
            if source and '/'.join(source) in self.simready_entries:
                continue  # A wrapped SimReady asset is shown once, as its SimReady card.
            label = CATEGORY_LABELS.get(entry['category'], entry['category'])
            meta = entry['entry'].get('category_label') or label
            picture = THUMBNAILS / (entry['key'] + '.png')
            items.append({'id': choice, 'label': entry['label'], 'category': entry['category'], 'members': None,
                          'meta': meta + (' · 资产库' if entry.get('library') else ''), 'badge': None,
                          'shape': entry.get('shape'), 'local': True,
                          'image': f"thumb/{urllib.parse.quote(entry['key'])}.png" if picture.is_file() else None,
                          'search': f"{entry['label']} {choice} {entry['category']} {label}".lower()})
        families = {}
        for key, entry in sorted(self.simready_entries.items()):
            base = re.sub(r'_[a-z]{0,2}\d{1,3}$', '', entry['name'])  # e.g. the 20 apple variants
            families.setdefault(f"{entry['publisher']}/{base}", []).append(entry)
        for family, members in families.items():
            first = next((entry for entry in members if entry['local']), members[0])
            base, category = family.split('/', 1)[1], members[0]['category']
            shown = base[len(category) + 1:] if base.startswith(category + '_') and len(base) > len(category) + 1 else base
            local = sum(entry['local'] for entry in members)
            items.append({'id': 'simready:' + family, 'label': shown.replace('_', ' '), 'category': 'simready:' + category,
                          'members': [{'key': f"{entry['publisher']}/{entry['name']}", 'name': entry['name'],
                                       'local': entry['local'], 'image': simready_preview_url(entry)} for entry in members],
                          'first': f"{first['publisher']}/{first['name']}", 'shape': None,
                          'local': local == len(members) or ('partial' if local else False),
                          'badge': f'×{len(members)}' if len(members) > 1 else None, 'image': simready_preview_url(first),
                          'meta': category + ('' if first['publisher'] == 'nvidia' else f" · {first['publisher']}"),
                          'search': f"{family} {category} {' '.join(entry['name'] for entry in members)}".lower()})
        for entry in self.environment_entries.values():
            picture = environment_thumbnail(entry['key'])
            state = '已包装' if entry['wrapped'] else '本地' if entry['local'] else '在线，添加时下载'
            items.append({'id': 'environment:' + entry['key'], 'label': entry['label'], 'category': 'environment',
                          'members': None, 'shape': None, 'local': entry['local'], 'badge': None,
                          'meta': ('室外' if entry['kind'] == 'outdoor' else '室内') + f' · {state}'
                                  + ('' if entry['verified'] else ' · 未验证'),
                          'image': f'thumb/{urllib.parse.quote(picture.stem)}.png' if picture.is_file() else None,
                          'search': f"{entry['label']} {entry['key']} environment 场景 环境".lower()})
        counts = {}
        for item in items:
            counts[item['category']] = counts.get(item['category'], 0) + 1
        own = [c for c in CATEGORY_LABELS if c in counts] + sorted(c for c in counts if c not in CATEGORY_LABELS
                                                                   and not c.startswith('simready:'))
        categories = [{'value': c, 'label': CATEGORY_LABELS.get(c, c), 'count': counts[c], 'group': '本项目可用'} for c in own]
        simready_count = sum(n for c, n in counts.items() if c.startswith('simready:'))
        if simready_count:
            categories.append({'value': 'simready:', 'label': 'SimReady 全部', 'count': simready_count, 'group': 'SimReady'})
            categories += [{'value': c, 'label': c[9:], 'count': counts[c], 'group': 'SimReady'}
                           for c in sorted((c for c in counts if c.startswith('simready:')), key=lambda c: (-counts[c], c))]
        self.shelf_items, self.shelf_categories = items, categories
        self.shelf_items_revision += 1
        if self._shelf_item() is None:
            self.asset_choice = self.asset_variant = None
        self._describe_asset()

    def _shelf_item(self, choice=None):
        choice = self.asset_choice if choice is None else choice
        return next((item for item in self.shelf_items if item['id'] == choice), None)

    def _simready_choice(self):
        """publisher/asset of the SimReady card (and variant) chosen on the shelf, else None."""
        item = self._shelf_item()
        if not item or not item['members']:
            return None
        return self.asset_variant if any(member['key'] == self.asset_variant for member in item['members']) else item['first']

    def _shelf_changed(self):
        self.shelf_revision += 1

    def _mounting(self):
        """The entity a sensor about to be added is mounted on (it then needs no place in the view)."""
        return self.mount_targets.get(self.asset_mount.value) if self.asset_mount.visible else None

    @action
    def _shelf_arm(self):
        item = self._shelf_item()
        if item is None:
            raise ValueError('请先在资产架中选择资产')
        if self._mounting() is not None or item['category'] == 'environment':
            return self._add_choice()  # a mounted sensor or an environment has no place to click
        self.arm_placement(self.entity_name or item['label'], lambda position: self._add_asset(position))
        self._shelf_changed()

    @action
    def _choose_asset(self, choice):
        """Select `choice` (e.g. a just imported asset); the shelf page clears its filters to show it."""
        self.asset_choice, self.asset_variant, self.shelf_focus = choice, None, True
        self._describe_asset()

    # The shelf page's API (tools/asset_shelf.html through tools/simready_library.py).
    def shelf_get(self, name, query):
        with self.lock:
            if name == 'items':
                return {'items': self.shelf_items, 'categories': self.shelf_categories,
                        'loading': not self.simready_loaded, 'revision': self.shelf_items_revision}
            if name == 'state':
                return self._shelf_state()
        raise KeyError(f'未知的资产架请求 {name}')

    def shelf_post(self, name, request):
        """Shelf actions; a rejected one is also shown in the editor and comes back to the page as an error."""
        with self.lock:
            try:
                if name == 'select':
                    self._shelf_select(request.get('id'), request.get('variant'))
                elif name == 'option':
                    self._shelf_option(request)
                elif name == 'arm':
                    self._shelf_name(request)
                    self._shelf_arm.__wrapped__(self)
                elif name == 'cancel':
                    if self.placing:
                        self.cancel_placement(f'已取消放置 {self.placing["label"]}')
                elif name == 'drop':
                    self._shelf_drop(request)
                else:
                    raise KeyError(f'未知的资产架请求 {name}')
            except Exception as error:
                self.report(f'⚠️ {error}', error=True)
                raise
            self.shelf_revision += 1
            return self._shelf_state()

    def _shelf_state(self):
        mount = self._mounting()
        options = {}
        if self.asset_mount.visible:
            options['mount'] = {'options': list(self.asset_mount.options), 'value': self.asset_mount.value}
        if self.asset_mount_body.visible:
            options['body'] = {'options': list(self.asset_mount_body.options), 'value': self.asset_mount_body.value}
        if self.asset_size.visible:
            options['size'] = [float(v) for v in self.asset_size.value]
        environment = (self.asset_choice or '').startswith('environment:')
        state = {'items': self.shelf_items_revision, 'revision': self.shelf_revision, 'choice': self.asset_choice,
                 'variant': self._simready_choice(), 'name': self.entity_name, 'detail': self.shelf_detail,
                 'focus': self.shelf_focus, 'options': options, 'mounted': mount is not None or environment,
                 'button': f'安装到 {self.asset_mount.value}' if mount is not None else
                 '添加到场景' if environment else '点击放置',
                 'placing': self.placing and self.placing['label'], 'job': self._shown_job()}
        self.shelf_focus = False
        return state

    def _shelf_select(self, choice, variant=None):
        if self._shelf_item(choice) is None:
            raise KeyError(f'资产架中没有 {choice}')
        if self.placing:
            self.cancel_placement()
        if choice != self.asset_choice or variant != self.asset_variant:
            self.asset_choice, self.asset_variant = choice, variant
            self._describe_asset()

    def _shelf_name(self, request):
        name = str(request.get('name') or '').strip()
        if name:
            if not name.isidentifier():
                raise ValueError(f'实体名称 {name!r} 须是 Python 标识符（字母、数字、下划线）')
            self.entity_name = name

    def _shelf_option(self, request):
        self._shelf_name(request)
        self.syncing += 1
        try:
            if request.get('mount') in (self.asset_mount.options or ()):
                self.asset_mount.value = request['mount']
            if request.get('size'):
                self.asset_size.value = tuple(float(v) for v in request['size'])
        finally:
            self.syncing -= 1
        if 'mount' in request:
            self._offer_mount_bodies()
        if request.get('body') in (self.asset_mount_body.options or ()):
            self.asset_mount_body.value = request['body']

    def _shelf_drop(self, request):
        """A card dropped on the view at (u, v) of its canvas: add it where that pixel's ray meets a surface."""
        self._shelf_select(request.get('id'), request.get('variant'))
        self._shelf_name(request)
        if self._mounting() is not None:
            return self._add_choice()
        u, v, aspect = (float(request[key]) for key in ('u', 'v', 'aspect'))
        clients = list(self.server.get_clients().values())
        if not clients:
            raise ValueError('没有打开的 3D 视图')
        # The browser that dropped it: the one whose view has this canvas' proportions.
        camera = min(clients, key=lambda client: abs(client.camera.aspect - aspect)).camera
        half = np.tan(camera.fov / 2)
        ray = F.quat_matrix(camera.wxyz)[:3, :3] @ [(2 * u - 1) * half * aspect, (2 * v - 1) * half, 1.0]  # OpenCV camera
        point = self.surface_point(camera.position, ray)
        if point is None:
            raise ValueError('没有落在地面或物体表面上，请拖到地面上')
        self._add_choice([round(float(value), 4) for value in point])

    def shelf_picture(self, key):
        """A catalog sketch for the shelf page, brightened to its white cards."""
        path = THUMBNAILS / f'{key}.png'
        if '/' in key or '..' in key or not path.is_file():
            raise KeyError(f'没有示意图 {key}')
        stamp = (path, path.stat().st_mtime)
        if stamp not in self.shelf_pictures:
            self.shelf_pictures[stamp] = sketch_on_white(path)
        return self.shelf_pictures[stamp]

    def _describe_asset(self):
        """Fit the option fields to the chosen asset, suggest a name and publish the shelf's detail."""
        self.shelf_detail = None
        key = self._simready_choice()
        if key:
            self._describe_simready(key)
        elif (self.asset_choice or '').startswith('environment:'):
            self._describe_environment(self.asset_choice.split(':', 1)[1])
        elif self.gallery_entries.get(self.asset_choice):
            self._describe_gallery()
        else:
            for field in (self.asset_mount, self.asset_mount_body, self.asset_size, self.asset_parameters):
                field.visible = False
        self._shelf_changed()

    def _detail(self, title, meta, lines, image=None):
        self.shelf_detail = {'title': title, 'meta': meta, 'lines': [line for line in lines if line], 'image': image}

    def _describe_gallery(self):
        choice, item = self.gallery_entries[self.asset_choice], self._shelf_item()
        shape = choice.get('shape')
        sensor = shape is None and choice['category'] == 'sensor'
        self.asset_parameters.visible = shape is None
        self.asset_placement.visible, self.asset_position.visible = shape is None, True
        self.asset_mount.visible = self.asset_mount_body.visible = sensor
        if sensor:
            self._offer_mounts()
        self.asset_size.visible = shape in ('rectangle', 'box')
        if shape:
            self._detail(choice['label'], f'抽象体 · {shape}', [
                '无模型、无物理；可选作观察、跟随物体（mount）、随机复位。' + ABSTRACT_NOTES[shape] + '含义由 task.py / 模型决定。'],
                item['image'])
            self._suggest_name({'point': 'target', 'direction': 'heading', 'rectangle': 'area', 'box': 'volume'}[shape], {})
            return
        factory, entry = choice['factory'], choice['entry']
        required = [name for name, field in entry.get('parameters', {}).items() if field.get('required')]
        fields = ', '.join(entry.get('parameters', {}))
        library = choice.get('library')
        source = (f'{"地形地块" if library.get("category") == "terrain" else "资产库"} · {factory}' if library
                  else factory)
        self._detail(choice['label'], f'{entry.get("category_label") or CATEGORY_LABELS.get(choice["category"], "")} · {source}', [
            entry.get('description', ''),
            f'能力：{", ".join(entry.get("capabilities", []))}' if entry.get('capabilities') else '',
            f'必填参数：{", ".join(required)}' if required and not library else '',
            f'参数：{fields}（“选项 · 按坐标添加”中填写）' if fields else ''], item['image'])
        parameters = choice['parameters'] if choice['parameters'] is not None else \
            {name: field.get('default', field.get('example')) for name, field in entry.get('parameters', {}).items()
             if field.get('required') and ('default' in field or 'example' in field)}
        self._suggest_name(re.sub(r'\W', '_', library['id'] if library else factory), parameters)

    def _describe_simready(self, key):
        entry = self.simready_entries[key]
        self.asset_parameters.visible = self.asset_size.visible = False
        self.asset_mount.visible = self.asset_mount_body.visible = False
        self.asset_placement.visible = self.asset_position.visible = True
        size = f" · {entry['bytes'] / 1e6:.1f} MB" if entry.get('bytes') else ''
        usage = self._simready_usage().get(key)
        self._detail(entry['name'], f"SimReady · {entry['category']} · " + ('本地' if entry['local'] else '在线，添加时下载') + size, [
            f'作为 usd_object 加入：刚体与关节按 USD 原样，默认动态 · {simready.LICENSE}',
            f'场景中已有：{", ".join(usage)}' if usage else ''], simready_preview_url(entry))
        base = re.sub(r'_[a-z]{0,2}\d{1,3}$', '', entry['name']).split('_')
        self._suggest_name(base[-1] if len(base) > 1 else base[0], {})

    def _describe_environment(self, key):
        entry, item = self.environment_entries[key], self._shelf_item()
        for field in (self.asset_parameters, self.asset_size, self.asset_mount, self.asset_mount_body,
                      self.asset_placement, self.asset_position):
            field.visible = False
        state = '已包装' if entry['wrapped'] else '本地，添加时包装' if entry['local'] else '在线，添加时下载并包装'
        spawn = environments.recorded_spawn(key)
        self._detail(entry['label'], f"场景环境 · {key} · {state}", [
            '作为全局静态 usd_object 加入（所有环境共用一份，地面放在 z = 0）；放置位置取场景目录，不取点击位置。',
            entry['note'], '尚未用机器人 smoke 验证' if not entry['verified'] else '',
            _spawn_text(spawn) if spawn else ('包装时会找出可放机器人的空地' if spawn is None else '没找到可放机器人的空地'),
            f"许可：{entry['license']}"], item and item['image'])
        self._suggest_name('environment', {})

    def _unused_name(self, base):
        names = {item['name'] for item in self.snapshot['entities']} | {job['name'] for job in self.jobs if job['running']}
        candidate, index = base, 2
        while candidate in names:
            candidate, index = f'{base}_{index}', index + 1
        return candidate

    def _suggest_name(self, base, parameters):
        candidate = self._unused_name(base)
        self.syncing += 1
        try:
            self.entity_name = candidate
            self.asset_parameters.value = json.dumps(parameters, ensure_ascii=False)
        finally:
            self.syncing -= 1

    # ----------------------------------------------------------------- SimReady
    def _start_library(self, host, port):
        """The full-page SimReady browser (tools/simready_library.py) on the editor's host."""
        import types
        from tools.simready_library import Library
        try:
            self.library = Library(host, port, types.SimpleNamespace(
                catalog=self.library_catalog, asset=self.library_asset, status=self.library_status, add=self.library_add,
                shelf_get=self.shelf_get, shelf_post=self.shelf_post, shelf_picture=self.shelf_picture,
                outliner_get=self.outliner_get, outliner_post=self.outliner_post,
                origin_port=self.server.get_port()))
        except OSError as error:
            self.report(f'⚠️ SimReady 资产库未启动：{error}', error=True)
            return
        shown = 'localhost' if host in ('127.0.0.1', '0.0.0.0', '::', '') else host
        self.library_url = f'http://{shown}:{self.library.port}/'
        self.simready_link.content = f'<small>[在新标签页打开]({self.library_url}) · {self.library_url}</small>'
        print(f'SimReady asset library: {self.library_url}')
        # srcdoc: the page shares the editor's origin, so it can reach the view for drops; it calls the
        # library server on the editor's host name (a remote browser needs that port forwarded too).
        page = SHELF_PAGE.read_text().replace('__LIBRARY_PORT__', str(self.library.port))
        self.shelf_frame.content = (f'<iframe srcdoc="{html.escape(page)}" title="资产架" style="display:block;width:100%;'
                                    'height:calc(100vh - 250px);min-height:380px;border:1px solid #e3e6ea;'
                                    'border-radius:8px;background:#f6f7f9"></iframe>')
        page = OUTLINER_PAGE.read_text().replace('__LIBRARY_PORT__', str(self.library.port))
        with self.lock:
            self.outliner_page = True
            self._rebuild_tree_gui()  # drops the button rows
            self.tree_style.content = ''
            self.outliner_frame.content = (f'<iframe srcdoc="{html.escape(page)}" title="场景列表" style="display:block;'
                                           'width:100%;height:calc(100vh - 190px);min-height:300px;border:0;'
                                           'background:transparent"></iframe>')

    def _open_library(self, client):
        """The browser in a near-full-screen dialog over the editor."""
        if not self.library_url:
            raise ValueError('SimReady 资产库未启动')
        modal = client.gui.add_modal('SimReady 资产库')
        with modal:
            client.gui.add_html(
                '<style>.mantine-Modal-content{flex:0 0 96vw!important;max-width:96vw!important}'
                '.mantine-Modal-inner{padding:2vh 2vw!important}</style>'
                f'<iframe src="{self.library_url}" style="width:100%;height:84vh;border:0;border-radius:6px"></iframe>')
            client.gui.add_button('关闭（新增的实体已在场景中，保存后写入 scene.yaml）').on_click(lambda _: modal.close())

    def _load_simready(self):
        """List SimReady assets (mirror, else online) and give wrapped ones their official preview."""
        try:
            entries, problems = simready.catalog()
            for item in recorded_assets():
                target = simready_thumbnail(*item['simready']) if item.get('simready') else None
                if target and not target.is_file():
                    with contextlib.suppress(Exception):  # offline: the card keeps its icon
                        simready.thumbnail(*item['simready'], target)
        except Exception as error:  # noqa: BLE001 - surfaced in the browser
            traceback.print_exc()
            entries, problems = [], {'目录': str(error)}
        with self.lock:
            self.simready_entries = {f"{entry['publisher']}/{entry['name']}": entry for entry in entries}
            self.simready_problems, self.simready_loaded = problems, True
            local = sum(entry['local'] for entry in entries)
            self.simready_status.content = f'<small>{len(entries)} 个资产（本地 {local}）' + ''.join(
                f'<br>⚠️ {source} 未能在线列出（只显示本地）：{text[:160]}' for source, text in problems.items()) + '</small>'
            self._refresh_assets()

    def _simready_usage(self):
        """{publisher/asset: names of scene entities whose usd_object loads its wrapper}."""
        sources = {item['parameters']['usd_path']: '/'.join(item['simready'])
                   for item in recorded_assets() if item.get('simready')}
        usage = {}
        for entity in self.snapshot['entities']:
            key = sources.get((entity.get('parameters') or {}).get('usd_path'))
            if key:
                usage.setdefault(key, []).append(entity['name'])
        return usage

    def library_catalog(self):
        with self.lock:
            usage = self._simready_usage()
            entries = [{**entry, 'scene': usage.get(key, [])} for key, entry in self.simready_entries.items()]
        return {'entries': entries, 'problems': self.simready_problems, 'loading': not self.simready_loaded,
                'license': simready.LICENSE,
                'sources': {name: {'repo': source['repo'], 'revision': source['revision']}
                            for name, source in simready.SOURCES.items()}}

    def library_asset(self, key):
        entry = self.simready_entries.get(key)
        if entry is None:
            raise KeyError(f'未知的 SimReady 资产 {key!r}')
        publisher, name = entry['publisher'], entry['name']
        listed = simready.files(publisher, name, online=False)
        if listed is None:
            with contextlib.suppress(Exception):  # offline or delisted: details stay partial
                listed = simready.files(publisher, name)
        skipped = simready.duplicates(listed) if listed else set()
        absent = simready.missing(publisher, name, listed) if listed and entry['local'] else None
        root = simready.root_usd(name, listed) if listed else None
        if root is None and entry['local']:
            found = simready.local_root(publisher, name)
            root = found.relative_to(simready.MIRROR / publisher).as_posix() if found else None
        info = {'local': entry['local'], 'root': root.split('/', 1)[1] if root else None,
                'files': len(listed) - len(skipped) if listed else None,
                'bytes': sum(item['size'] for item in listed if item['path'] not in skipped) if listed else entry['bytes'],
                'missing': None if absent is None else {'count': len(absent), 'bytes': sum(item['size'] for item in absent)},
                'wrapper': None, 'wrapper_other': False, 'joints': None}
        record_path = ROOT / 'assets' / name / 'ASSET.json'
        if record_path.is_file():
            with contextlib.suppress(OSError, ValueError):
                record = json.loads(record_path.read_text())
                info['wrapper'] = f'assets/{name}'
                info['wrapper_other'] = bool(root) and record.get('source') != f'assets/simready/{publisher}/{root}'
                info['joints'] = len(record.get('joint_order') or [])
        with self.lock:
            info['scene'] = self._simready_usage().get(key, [])
            info['suggested'] = self._unused_name(re.sub(r'\W', '_', name).lower())
        return info

    def library_status(self):
        return self._shown_job()

    def _start_job(self, key, name, message):
        """Register a background addition; other assets may be prepared meanwhile, each asset once at a time."""
        if not name.isidentifier():
            raise ValueError(f'实体名称 {name!r} 须是 Python 标识符（字母、数字、下划线）')
        running = [job for job in self.jobs if job['running']]
        if any(job['key'] == key for job in running):
            raise ValueError(f'{key} 仍在准备中')
        if name in {item['name'] for item in self.snapshot['entities']} | {job['name'] for job in running}:
            raise ValueError(f'场景中已有（或正在添加）实体 {name}')
        job = {'key': key, 'name': name, 'running': True, 'error': False, 'done': 0, 'total': 0, 'message': message}
        self.jobs = [*running, job]
        self.report(message)
        self._describe_asset()  # the next addition gets a fresh name
        return job

    def _shown_job(self):
        """The job the shelf shows: the newest running one (with how many others run), else the newest."""
        running = [job for job in self.jobs if job['running']]
        if not running:
            return self.jobs[-1] if self.jobs else None
        others = len(running) - 1
        return {**running[-1], 'message': running[-1]['message'] + (f'（另有 {others} 项在后台准备）' if others else '')}

    def library_add(self, key, name, position):
        """Download/wrap ``key`` in the background, then add it to the scene as a usd_object at ``position``."""
        entry = self.simready_entries.get(key)
        if entry is None:
            raise KeyError(f'未知的 SimReady 资产 {key!r}')
        position = [float(value) for value in position]
        if len(position) != 3:
            raise ValueError('位置须是 3 个数')
        with self.lock:
            job = self._start_job(key, name, f'准备 {key}（{"本地镜像" if entry["local"] else "从 Hugging Face 下载"}，再包装）…')
        threading.Thread(target=self._prepare_simready, args=(job, entry, position), daemon=True).start()
        return job

    def _prepare_simready(self, job, entry, position):
        shown = [0.0]

        def progress(done, total):
            job.update(done=done, total=total, message=f'下载 {job["key"]}：{done / 1e6:.1f}/{total / 1e6:.1f} MB')
            if time.time() - shown[0] > 1 or done == total:
                shown[0] = time.time()
                self.report(job['message'])

        try:
            record = simready.prepare(entry['publisher'], entry['name'], progress)
            entry['local'] = True
            with contextlib.suppress(Exception):  # the shelf card falls back to its icon
                simready.thumbnail(entry['publisher'], entry['name'], simready_thumbnail(entry['publisher'], entry['name']))
            with self.lock:
                factory = next((item for item in self.snapshot['catalog']['assets'] if item['id'] == 'usd_object'), None)
                if factory is None:
                    raise ValueError(f'已准备 {record["formats"]["usd"]}，但当前后端的资产目录没有 usd_object')
                self._save_first()
                self._add_entity({'name': job['name'], 'factory': 'usd_object', 'kind': 'entity',
                                  'parameters': {'usd_path': record['formats']['usd']},
                                  'pose': {'position': position, 'orientation': [1, 0, 0, 0]}}, factory)
                self._refresh_assets()  # the wrapper also joins the asset shelf
                job.update(scene=self._simready_usage().get(job['key'], []), running=False,
                           message=f'已添加 {job["name"]}（{job["key"]}）；在编辑器中保存后写入 scene.yaml')
        except Exception as error:  # noqa: BLE001 - surfaced in both pages
            traceback.print_exc()
            job.update(running=False, error=True, message=f'{job["key"]} 未添加：{error}')
            self.report(f'⚠️ {job["message"]}', error=True)

    def _environment_add(self, key, name):
        """Download/wrap the catalog environment ``key`` in the background, then add it as a global usd_object."""
        job = self._start_job(key, name, f'准备场景环境 {key}…')
        threading.Thread(target=self._prepare_environment, args=(job,), daemon=True).start()
        return job

    def _prepare_environment(self, job):
        def progress(message):
            job['message'] = message
            self.report(message)

        try:
            record = environments.prepare(job['key'], progress)
            with self.lock:
                factory = next((item for item in self.snapshot['catalog']['assets'] if item['id'] == 'usd_object'), None)
                if factory is None:
                    raise ValueError(f'已准备 {record["formats"]["usd"]}，但当前后端的资产目录没有 usd_object')
                self._save_first()
                self._add_entity({'name': job['name'], 'factory': 'usd_object', 'kind': 'entity', 'scope': 'global',
                                  'parameters': {'usd_path': record['formats']['usd']},
                                  'pose': {'position': record['environment']['pose'], 'orientation': [1, 0, 0, 0]}},
                                 factory)
                self._load_environments()
                spawn = record['environment'].get('spawn')
                job.update(running=False, message=f'已添加 {job["name"]}（{job["key"]}）；在编辑器中保存后写入 scene.yaml'
                           + (f'。{_spawn_text(spawn, 1)}' if spawn else ''))
        except Exception as error:  # noqa: BLE001 - surfaced in the shelf and the editor
            traceback.print_exc()
            job.update(running=False, error=True, message=f'{job["key"]} 未添加：{error}')
            self.report(f'⚠️ {job["message"]}', error=True)

    def _load_environments(self, thumbnails=False):
        """Catalog environments for the shelf (part downloads are left to the command line)."""
        entries = {entry['key']: entry for entry in environments.catalog() if not entry.get('part')}
        with self.lock:
            self.environment_entries = entries
            self._refresh_assets()
        if not thumbnails:
            return
        missing = []
        for key in entries:
            if environment_thumbnail(key).is_file():
                continue
            source = None
            with contextlib.suppress(Exception):  # offline: drawn below from the local USD, if any
                source = environments.thumbnail(key, draw=False)
            if source:
                THUMBNAILS.mkdir(parents=True, exist_ok=True)
                environment_thumbnail(key).write_bytes(source.read_bytes())
            elif entries[key]['local']:
                missing.append(key)
        with self.lock:
            self._refresh_assets()
        if not missing:
            return
        # Floor plans load whole scenes (several GB each): drawn in a separate process, one after another.
        process = subprocess.Popen([sys.executable, '-m', 'shared.environments', *missing], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in process.stdout:
            key, _, path = line.strip().partition(' ')
            path = path.split(' ', 1)[0]
            if path and Path(path).is_file():
                THUMBNAILS.mkdir(parents=True, exist_ok=True)
                environment_thumbnail(key).write_bytes(Path(path).read_bytes())
                with self.lock:
                    self._refresh_assets()
        process.wait()
        # Another editor's drawer may hold the machine-wide lock: take what it has drawn by now.
        found = False
        for key in missing:
            source = environments.thumbnail(key, online=False, draw=False)
            if source and not environment_thumbnail(key).is_file():
                environment_thumbnail(key).write_bytes(source.read_bytes())
                found = True
        if found:
            with self.lock:
                self._refresh_assets()

    # --------------------------------------------------------------- thumbnails
    def _harvest_thumbnails(self):
        """Sketch assets that already have native meshes in the open scene."""
        with self.lock:
            jobs = []
            for item in self.snapshot['entities']:
                record, _ = self.visual(item)
                if not record or not item.get('factory'):
                    continue
                defaults = json.loads(json.dumps({name: field.get('default') for name, field in
                            self.asset_entry(item['factory']).get('parameters', {}).items() if 'default' in field}))
                changed = {key: value for key, value in item.get('parameters', {}).items() if defaults.get(key, object()) != value}
                key = thumbnail_key(item['factory'], changed)
                if not (THUMBNAILS / (key + '.png')).is_file():
                    jobs.append((key, record, item.get('initial_state', {}).get('joint_positions', {})))
        for key, record, positions in jobs:
            self._write_thumbnail(key, record, positions)
        if jobs:
            with self.lock:
                self._refresh_assets()

    @staticmethod
    def _write_thumbnail(key, record, positions=None):
        try:
            data = render_thumbnail(record, positions)
        except Exception:  # noqa: BLE001 - a missing sketch falls back to an icon
            traceback.print_exc()
            return
        if data:
            THUMBNAILS.mkdir(parents=True, exist_ok=True)
            temporary = THUMBNAILS / (key + '.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, THUMBNAILS / (key + '.png'))

    @action
    def start_thumbnails(self, keys=None):
        """Export one instance of each asset lacking a sketch in a throwaway scene (shared factories only)."""
        if getattr(self, 'thumbnail_thread', None) and self.thumbnail_thread.is_alive():
            self.report('示意图正在生成中')
            return
        from shared.backends.isaac.assets import FACTORIES  # noqa: F401 - availability check below
        entities, wanted = [], {}
        for choice, entry in self._gallery().items():
            if keys is not None and entry['key'] not in keys:
                continue
            if (THUMBNAILS / (entry['key'] + '.png')).is_file() or entry['category'] not in ('robot', 'object', 'terrain') \
                    or entry['factory'] not in FACTORIES:
                continue
            parameters = entry['parameters'] or {}
            if any(field.get('required') and name not in parameters
                   for name, field in entry['entry'].get('parameters', {}).items()):
                continue
            name = f'a{len(entities)}'
            wanted['world/' + name] = entry['key']
            entities.append({'name': name, 'factory': entry['factory'], 'parameters': parameters,
                             'kind': 'robot' if entry['category'] == 'robot' else 'entity',
                             'pose': {'position': [3.0 * len(entities), 0.0, 0.0], 'orientation': [1, 0, 0, 0]}})
        if not entities:
            self.report('没有缺少示意图、且可在独立场景中生成的资产')
            return
        GALLERY_SCENE.parent.mkdir(parents=True, exist_ok=True)
        GALLERY_SCENE.write_text(json.dumps({'schema_version': 1, 'name': 'world', 'entities': entities}, indent=1))

        def run():
            try:
                self._export_visuals(GALLERY_SCENE)
                manifest = json.loads(cache_path(GALLERY_SCENE).read_text())['entities']
            except Exception as error:
                manifest = {}
                self.report(f'示意图生成失败：{str(error)[-1200:]}', error=True)
            done = 0
            for path, key in wanted.items():
                record = manifest.get(path) or {}
                if str(record.get('path') or '').endswith('.mesh.json.gz'):
                    self._write_thumbnail(key, record)
                    done += 1
            with self.lock:
                self._refresh_assets()
                if done:
                    self.report(f'已生成 {done}/{len(wanted)} 张资产示意图' +
                                ('；其余为几何体等无原生网格的资产，保留图标' if done < len(wanted) else ''))
        self.thumbnail_thread = threading.Thread(target=run, daemon=True)
        self.thumbnail_thread.start()
        self.report(f'后台生成 {len(entities)} 个资产的示意图（Isaac 导出）')

    @action
    def _import_asset(self):
        raw = self.import_path.value.strip()
        if not raw:
            raise ValueError('请填写 URDF / USD 文件路径')
        path = Path(raw).expanduser()
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if not path.is_file():
            raise ValueError(f'找不到文件：{path}')
        suffix, kind = path.suffix.lower(), IMPORT_KINDS[self.import_kind.value]
        try:
            stored = str(path.relative_to(ROOT))
        except ValueError:
            stored = str(path)  # Outside the workspace: works here, not portable.
        if suffix == '.urdf':
            if kind != 'robot':
                raise ValueError('URDF 请按“机器人（带关节）”导入')
            factory, parameters = 'urdf_robot', {'urdf_path': stored, 'fix_base': self.import_fixed.value}
        elif suffix in ('.usd', '.usda', '.usdc', '.usdz'):
            # Objects follow their USD bodies: dynamic, with passive joints (cabinet doors, drawers) if any.
            factory = 'usd_robot' if kind == 'robot' else 'usd_object'
            parameters = {'usd_path': stored, **({'fix_root_link': self.import_fixed.value} if kind == 'robot' else {})}
        else:
            raise ValueError('仅支持 .urdf / .usd / .usda / .usdc / .usdz')
        identifier = re.sub(r'\W', '_', path.stem).strip('_') or 'asset'
        if identifier[0].isdigit():
            identifier = 'asset_' + identifier
        library = load_library()
        same = next((item for item in library if item['factory'] == factory and item.get('parameters') == parameters), None)
        if same is None:
            taken, candidate, index = {item['id'] for item in library}, identifier, 2
            while candidate in taken:
                candidate, index = f'{identifier}_{index}', index + 1
            same = {'id': candidate, 'label': self.import_label.value.strip() or path.stem,
                    'factory': factory, 'parameters': parameters}
            library.append(same)
            save_library(library)
        self._refresh_assets()
        self._choose_asset('library:' + same['id'])
        joints = f'，带 {len(usd_joints(path))} 个被动关节' if factory == 'usd_object' and usd_joints(path) else ''
        self.report(f'已登记资产 {same["label"]}{joints}；添加到场景后加载。示意图在后台生成')
        self.start_thumbnails(keys={thumbnail_key(factory, parameters)})

    @action
    def _refresh_native(self):
        self._save_first()  # exports the working copy; scene.yaml waits for save
        self.start_export(force=True)

    @action
    def _new_environment(self):
        name = self.new_name.value.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name):
            raise ValueError('名称请使用英文字母、数字和下划线，不能以数字开头')
        if (ROOT / 'projects' / name).exists():
            raise ValueError('项目目录已经存在，请换一个名称')
        if self.unsaved():
            raise ValueError('请先保存当前编辑，再新建环境')
        from shared.assembly.starter import create_draft
        self._edit(create_draft(name, self.new_template.value))
        self.draft, self.selected = name, None
        self.load()
        self.report(f'未保存环境（模板 {self.new_template.value}）；保存后生成 projects/{name}')

    @action
    def _open_environment(self):
        if self.unsaved():
            raise ValueError('请先保存当前编辑，再打开其他环境')
        path = Path(self.open_path.value).expanduser().resolve()
        if not path.is_file():
            raise ValueError('找不到 scene.yaml')
        old = self.target
        try:
            self._edit(path)
            self.draft, self.selected = None, None
            self.load()
        except Exception:
            self._edit(old)
            self.load()
            raise
        if self._needs_native():
            self.start_export()

    # ------------------------------------------------------ inline edit sections
    def _preview(self, key, nodes):
        """Start a live preview on first change; returns True when it just began."""
        if self.preview_restore is not None and self.preview_key != key:
            self._end_preview()
        if self.preview_restore is not None:
            return False
        self.preview_restore, self.preview_key = [(node, node.local.copy(), node.parent) for node in nodes], key
        self.gizmo.visible = False
        return True

    def _end_preview(self):
        for node, local, parent in self.preview_restore or []:
            node.local, node.parent = local, parent
            self._update_frame(node)
        self.preview_restore = None
        self._place_gizmo()

    def _section_buttons(self, title, apply_label='应用并保存'):
        gui = self.server.gui
        apply = gui.add_button(apply_label, color='green', icon='check')
        gui.add_button('取消', icon='x').on_click(lambda _: self._cancel_section(title))
        return apply

    @action
    def _cancel_section(self, title):
        self._end_preview()
        self.open_section = title
        self._rebuild_inspector()

    def _choices(self, exclude):
        labels = [item['label'] for item in self.snapshot['entities']]
        return {(item['label'] if labels.count(item['label']) == 1 else f'{item["label"]}  ({item["path"]})')
                + f' · {item["category_label"]}': item['path']
                for item in self.snapshot['entities'] if item['path'] != exclude}

    def _attach_section(self, path):
        gui, item, info = self.server.gui, self.items[path], self.info[path]
        title = '父物体'
        node = info['node']
        abstract, sensor = item['category'] == 'abstract', item['category'] == 'sensor'
        # Abstract entities only follow a root pose for readings; no physical joint.
        modes = ({'跟随父物体（仅读数）': 'physical', '共同摆放（仅布局）': 'layout'} if abstract else
                 {'物理安装（随连杆运动）': 'physical', '共同摆放（仅布局）': 'layout'})
        targets = self._choices(path)
        if abstract:
            targets = {label: target for label, target in targets.items()
                       if self.items[target]['category'] in ('robot', 'object')}
        capabilities = {entry['path']: entry.get('capabilities', []) for entry in self.snapshot['catalog']['entities']}
        relation = self.snapshot.get('layout_parents', {}).get(path) or item.get('mount')
        current = 'layout' if info['layout_parent'] else 'physical' if relation or info['fixed'] else None
        if info['fixed']:
            relation = info['fixed']['parent']
        if relation:
            relation = {**relation, 'entity': _resolve(path, relation['entity'])}
        target_label = {value: key for key, value in targets.items()}
        linked = bool(info['fixed'] or item.get('mount') or info['layout_parent'])

        def default_mode(target):
            if current:
                return current
            if abstract or sensor:
                return 'physical'
            # Two physical assets are fixed together; static props are only laid out.
            physical = all('root_state' in capabilities.get(p, []) for p in (path, target))
            return 'physical' if physical else 'layout'

        with self._section(title):
            if info['grouped'] and not info['layout_parent']:
                gui.add_markdown('先解散该实体所在的布局组合，再建立父子关系。')
                return
            if not targets:
                gui.add_markdown('场景中没有其他实体。')
                return
            target_pick = gui.add_dropdown('父物体', [NONE, *targets],
                                           initial_value=target_label.get((relation or {}).get('entity'), NONE))
            mode_label = {value: key for key, value in modes.items()}
            mode_pick = gui.add_dropdown('方式', list(modes), initial_value=mode_label[current or 'physical'])
            body_pick = gui.add_dropdown('父连杆', [ROOT_BODY])
            place = gui.add_dropdown('位置', list(PLACEMENT))
            offset = gui.add_vector3('偏移 (m)', (0.0, 0.0, 0.0), step=.001)
            rotation = gui.add_vector3('角度 XYZ (°)', (0.0, 0.0, 0.0), step=.1)
            align = gui.add_dropdown('朝向对齐', list(ALIGN))
            explain = gui.add_markdown('')
            apply = gui.add_button('应用', color='green', icon='check')
            cancel = gui.add_button('取消', icon='x')
            cancel.on_click(lambda _: self._cancel_section(title))
            detach = gui.add_button('解除父物体（保留当前位置）', icon='unlink')
            detach.on_click(lambda _: self._detach(path))
        state = {'syncing': False, 'start': node.world(), 'mode_touched': bool(current)}
        groups = {}

        def mode():
            return modes[mode_pick.value]

        def resolve_body():
            if target_pick.value == NONE:
                raise ValueError('请选择父物体')
            target = targets[target_pick.value]
            if abstract and mode() == 'physical':
                return target, ''  # Abstract entities follow the parent's root pose.
            if mode() == 'physical' and body_pick.value != ROOT_BODY:
                return target, groups.get(body_pick.value, body_pick.value)
            return target, (self.initial_display(target).get('root_body', '') if mode() == 'physical' else '')

        def offset_matrix(parent):
            choice = PLACEMENT[place.value]
            if choice == 'keep':
                return np.linalg.inv(parent.world()) @ state['start']
            if choice == 'origin':
                return np.eye(4)
            return F.align_rotation(F.transform(offset.value, F.euler_matrix(rotation.value)), ALIGN[align.value])

        def signature():
            return (target_pick.value, mode_pick.value, body_pick.value, place.value,
                    tuple(offset.value), tuple(rotation.value), align.value)

        def layout_fields():
            chosen = target_pick.value != NONE
            # Nothing is written until 应用; the section only shows what would change.
            changed = signature() != state['initial']
            target = targets.get(target_pick.value)
            mode_pick.visible = chosen
            body_pick.visible = chosen and mode() == 'physical' and not abstract and len(body_pick.options) > 1
            place.visible = chosen
            offset.visible = rotation.visible = align.visible = chosen and PLACEMENT[place.value] == 'custom'
            apply.visible = cancel.visible = changed
            detach.visible = linked and not changed
            if not chosen:
                explain.content = ('<small>点“应用”解除父物体，世界位置保留。</small>' if linked else
                                   '<small>选择父物体和方式后点“应用”；之后可拖动 3D 坐标轴调整相对位置。</small>')
            elif mode() == 'layout':
                explain.content = '<small>仅一起摆放，仿真中不固定。</small>'
            elif abstract:
                explain.content = '<small>读数跟随父物体根部，不是物理约束；跟随后不再随机复位。</small>'
            elif sensor:
                explain.content = '<small>传感器随父连杆运动。</small>'
            else:
                note = '；子实体自身的随机复位会移除，由父物体带动' if item.get('reset') else ''
                explain.content = f'<small>创建原生固定约束，随 `{target}` 运动{note}。</small>'

        def update_bodies():
            target = targets.get(target_pick.value)
            bodies = self.bodies(target) if target else []
            # Contact sensors may cover every link sharing a name suffix (the four feet).
            groups.clear()
            if 'contact_forces' in capabilities.get(path, []):
                groups.update(_body_groups(bodies))
            state['syncing'] = True
            try:
                body_pick.options = [ROOT_BODY, *bodies, *groups]
                current_body = (relation or {}).get('body')
                if relation and relation['entity'] == target and current_body in bodies:
                    body_pick.value = current_body
                elif relation and relation['entity'] == target and current_body in groups.values():
                    body_pick.value = {value: key for key, value in groups.items()}[current_body]
                if target and not state['mode_touched']:
                    mode_pick.value = mode_label[default_mode(target)]
            finally:
                state['syncing'] = False

        @self.guarded
        def preview(_=None):
            if state['syncing']:
                return
            self.open_section = title
            layout_fields()
            if target_pick.value == NONE or PLACEMENT[place.value] == 'keep':
                self._end_preview()  # Nothing moves; the gizmo stays usable.
                return
            if self._preview(title, [node]):
                state['start'] = node.world()
            try:
                parent = self.body_node(*resolve_body())
            except ValueError as error:
                explain.content = f'⚠️ {error}'
                return
            if parent is node or node in parent.ancestors():
                explain.content = '⚠️ 不能安装到自身或其子节点'
                return
            world = parent.world() @ offset_matrix(parent)
            node.local = np.linalg.inv(node.parent.world()) @ world if node.parent else world
            self._update_frame(node)

        def mode_changed(_):
            if not state['syncing']:
                state['mode_touched'] = True
            preview()

        update_bodies()
        if relation:
            try:
                relative = np.linalg.inv(self.body_node(*resolve_body()).world()) @ state['start']
                offset.value = tuple(round(v, 5) for v in _xyz(relative))
                rotation.value = tuple(round(v, 3) for v in F.matrix_euler(relative))
            except ValueError:
                pass
        state['initial'] = signature()
        layout_fields()
        target_pick.on_update(lambda _: (update_bodies(), preview()))
        mode_pick.on_update(mode_changed)
        for field in (body_pick, place, offset, rotation, align):
            field.on_update(preview)

        @self.guarded
        def execute(_):
            self._end_preview()
            if target_pick.value == NONE:
                self._detach(path)
                return
            state['start'] = node.world()
            target, body = resolve_body()
            parent = self.body_node(target, body)
            target_item = self.items[target]
            physical = mode() == 'physical'
            if physical and target_item['category'] == 'robot' and not body and not abstract:
                raise ValueError('请先刷新原生预览，解析机器人连杆')
            if parent is node or node in parent.ancestors():
                raise ValueError('不能安装到自身或其子节点')
            matrix = offset_matrix(parent)
            payload = self.collect()
            changes = {}
            payload['layout_parents'] = {path: None}
            payload['entities'].append({'path': path, 'changes': changes})
            if not physical:
                # Switching from a physical relation drops it in the same edit.
                if item.get('mount'):
                    changes['mount'] = None
                if info['fixed']:
                    if not info['fixed']['project']:
                        raise ValueError('共享组合内部连接请在模板中编辑')
                    payload['connections'] = {info['fixed']['name']: None}
                payload['layout_parents'][path] = {'entity': target, 'body': body}
                world = parent.world() @ matrix
                logical = self.info.get(item['parent'])
                if logical:
                    world = np.linalg.inv(logical['node'].world()) @ world
                changes['pose'] = F.matrix_pose(world)
            elif sensor or abstract:
                changes.update(mount={'entity': target} if abstract else {'entity': target, 'body': body},
                               pose=F.matrix_pose(matrix))
                if abstract:
                    changes['reset'] = None  # A followed entity is positioned by its parent.
            else:
                if any('root_state' not in capabilities.get(p, []) for p in (path, target)):
                    raise ValueError('物理固定需要两个物理资产；静态桌子请选择“共同摆放”')
                child_body = self.initial_display(path).get('root_body', '')
                if item['category'] == 'robot' and not child_body:
                    raise ValueError('请先刷新原生预览，解析机器人根连杆')
                if info['fixed'] and not info['fixed']['project']:
                    raise ValueError('共享组合内部连接请在模板中编辑')
                if item.get('reset'):
                    changes['reset'] = None  # The parent's reset carries the attached child.
                asset = next((entry for entry in self.snapshot['catalog']['assets'] if entry['id'] == item.get('factory')), {})
                if 'fix_root_link' in asset.get('parameters', {}):
                    changes['parameters'] = {'fix_root_link': False}
                name = info['fixed']['name'] if info['fixed'] else item['native_key'] + '_attachment'
                payload['connections'] = {name: {'parent': {'entity': target, 'body': body},
                                                 'child': {'entity': path, 'body': child_body},
                                                 'parent_pose': F.matrix_pose(matrix)}}
            self.open_section = title  # Keep the section open on the applied relation.
            self._apply(payload)
            self.report(f'已设置父物体 {target_item["label"]}；拖动坐标轴可调整相对位置'
                        + ('（已移除子实体自身的随机复位）' if changes.get('reset', 0) is None and not abstract else ''))
        apply.on_click(execute)

    def _initial_section(self, path):
        gui, item, info = self.server.gui, self.items[path], self.info[path]
        title = '初始状态'
        initial, authored = self.initial_display(path), item.get('initial_state', {})
        kinematics = info['kinematics'] or {'frames': {}, 'joints': []}
        limits = {joint['name']: joint.get('limits') for joint in kinematics['joints']}
        rows = {}
        with self._section(title):
            restore = gui.add_checkbox('恢复资产默认初始状态', False)
            linear = gui.add_vector3('机身线速度 (m/s)', tuple(authored.get('linear_velocity', initial.get('linear_velocity', (0, 0, 0)))), step=.001)
            angular = gui.add_vector3('机身角速度 (rad/s)', tuple(authored.get('angular_velocity', initial.get('angular_velocity', (0, 0, 0)))), step=.001)
            if initial.get('joint_positions'):
                gui.add_markdown('<small>关节位置 rad / m；速度 rad/s / m/s。拖动滑块实时预览。</small>')
                for name, value in initial['joint_positions'].items():
                    value = float(authored.get('joint_positions', {}).get(name, value))
                    bounds = limits.get(name)
                    if bounds and all(np.isfinite(bounds)) and bounds[0] < bounds[1]:
                        low, high = float(bounds[0]), float(bounds[1])
                        position = gui.add_slider(f'{name}', low, high, (high - low) / 500, min(max(value, low), high))
                    else:
                        position = gui.add_number(f'{name}', value, step=.001)
                    velocity = gui.add_number(f'{name} 速度', float(authored.get('joint_velocities', {}).get(
                        name, initial.get('joint_velocities', {}).get(name, 0.0))), step=.001)
                    rows[name] = (position, velocity)
            else:
                gui.add_markdown('关节列表需原生网格导出完成后显示')
            gui.add_markdown('<small>应用后保存；任务复位仍可随机化。</small>')
            apply = self._section_buttons(title)

        @self.guarded
        def preview(_):
            self.open_section = title
            if not kinematics['joints']:
                return
            self._preview(title, list(info['bodies'].values()))
            current = F.forward_kinematics(kinematics['frames'], kinematics['joints'],
                                           {name: fields[0].value for name, fields in rows.items()})
            for body, matrix in current.items():
                if body in info['bodies']:
                    info['bodies'][body].local = matrix
                    self._update_frame(info['bodies'][body])
        for fields in rows.values():
            fields[0].on_update(preview)

        @self.guarded
        def execute(_):
            self._end_preview()
            values = dict(authored)
            values.update(linear_velocity=list(linear.value), angular_velocity=list(angular.value))
            if rows:
                values['joint_positions'] = {name: float(fields[0].value) for name, fields in rows.items()}
                values['joint_velocities'] = {name: float(fields[1].value) for name, fields in rows.items()}
            if restore.value:
                values = {}
            payload = self.collect()
            entry = next((e for e in payload['entities'] if e['path'] == path), None)
            if entry is None:
                payload['entities'].append({'path': path, 'changes': {'initial_state': values}})
            else:
                entry['changes']['initial_state'] = values
            self.open_section = None
            self._apply(payload)
            if restore.value or not kinematics['joints']:
                self.start_export()
            self.report('初始状态已保存')
        apply.on_click(execute)

    def _reset_section(self, path):
        gui, title = self.server.gui, '随机复位'
        abstract = self.items[path]['category'] == 'abstract'
        # Any static abstract rectangle/box can bound a placement; Task code
        # decides what else it means.
        areas = [item['path'] for item in self.snapshot['entities']
                 if item['category'] == 'abstract' and item['path'] != path and not item.get('mount')
                 and not item.get('reset')
                 and self.abstract_geometry(item['path'])[0] in ('rectangle', 'box')]
        current = self.pending['resets'][path] if path in self.pending['resets'] else self.items[path].get('reset')
        current = current or {}
        with self._section(title):
            if current.get('region'):
                gui.add_markdown(f'当前：随机放置于 `{current["region"]}`，偏航 {current.get("yaw_range", [0, 0])}'
                                 + (f'，每 {current["interval"]} s 重采样' if current.get('interval') else ''))
            elif current.get('states') or current.get('pose_range'):
                gui.add_markdown('当前：YAML 中的 states / pose_range 复位')
            if current.get('joint_positions'):
                gui.add_markdown('关节随机范围（scene.yaml 中编辑，此处保留）：' + '，'.join(
                    f'`{joint}` {list(bounds)}' for joint, bounds in current['joint_positions'].items()))
            if not areas:
                gui.add_markdown('先在“添加 › 资产”陈列栏中添加抽象矩形或长方体，作为放置范围。')
            else:
                region = gui.add_dropdown('随机放置于', areas, initial_value=current.get('region')
                                          if current.get('region') in areas else areas[0])
                yaw = gui.add_vector2('偏航范围 (rad)', tuple(current.get('yaw_range', (0.0, 0.0))), step=.001)
                interval = gui.add_number('回合内重采样周期 (s，0 为不重采样)', float(current.get('interval') or 0.0),
                                          min=0.0, step=.01) if abstract else None
                stage = gui.add_button('暂存（保存时写入）', color='green', icon='check')
            clear = gui.add_button('取消随机复位', icon='x') if current else None

        @self.guarded
        def execute(_):
            self.checkpoint()
            value = {'region': region.value, 'yaw_range': [float(v) for v in yaw.value]}
            if interval is not None and interval.value > 0:
                value['interval'] = float(interval.value)
            self.pending['resets'][path] = value
            self.open_section = title
            self._rebuild_inspector()
            self.report(f'{path} 的随机复位已暂存；保存后写入 scene.yaml')

        @self.guarded
        def remove(_):
            self.checkpoint()
            self.pending['resets'][path] = None
            self.open_section = title
            self._rebuild_inspector()
            self.report(f'{path} 将不再随机复位；保存后写入 scene.yaml')
        if areas:
            stage.on_click(execute)
        if clear is not None:
            clear.on_click(remove)

    def _regions(self):
        """Scene paths of abstract entities other entities use as reset or wander regions (kept static)."""
        found = set()
        for path, item in self.items.items():
            reset = self.pending['resets'][path] if path in self.pending['resets'] else item.get('reset')
            if isinstance(reset, dict) and reset.get('region'):
                found.add(_resolve(path, reset['region']))
            motion = self.current_motion(path)
            if motion and motion.get('type') == 'wander':
                region = motion.get('region') or (reset or {}).get('region')
                if region:
                    found.add(_resolve(path, region))
        return found

    def _areas(self):
        """Abstract rectangles/boxes by scene path, as placed in the editor now (``abstract.reset_areas``)."""
        areas = {}
        for path, item in self.items.items():
            if item['category'] != 'abstract':
                continue
            shape, size = self.abstract_geometry(path)
            if shape in ('rectangle', 'box'):
                areas[path] = {'shape': shape, 'size': list(size), 'pose': F.matrix_pose(self.info[path]['node'].world())}
        return areas

    def _motion_section(self, path):
        gui, title, item = self.server.gui, '运动', self.items[path]
        abstract = item['category'] == 'abstract'
        state = None if abstract else self.physics_state(path)
        draft = self.motion_drafts.get(path) or copy.deepcopy(self.current_motion(path) or {})
        with self._section(title):
            if state is not None:
                mode, jointed = state
                gui.add_markdown(f'<small>物理：**{PHYSICS[mode][0]}**（{PHYSICS[mode][1]}）。'
                                 '在场景列表行尾的图标切换。</small>')
                if jointed or mode == 'static':
                    gui.add_markdown('<small>' + ('带关节的物体不能设为运动学，不能自行运动；可在场景列表中固定其根部。'
                                                  if jointed else '静态几何没有刚体，不能自行运动。') + '</small>')
                    return
            if path in self._regions():
                gui.add_markdown('<small>它是别的实体的复位或游走区域，保持静止。</small>')
                return
            pick = gui.add_dropdown('自行运动', list(MOTIONS.values()), initial_value=MOTIONS[draft.get('type')])
            kind = draft.get('type')
            fields = {}
            if kind:
                speed = draft.get('speed', [.1, .1])
                speed = speed if isinstance(speed, (list, tuple)) else [speed, speed]
                fields['speed'] = gui.add_vector2('速度范围 (m/s)', tuple(float(v) for v in speed), min=(0., 0.), step=.01,
                                                  hint='每回合在范围内取一个速度；两值相同即固定速度')
                fields['face'] = gui.add_checkbox('朝向运动方向', bool(draft.get('face')),
                                                  hint='绕竖直轴转向；当前朝向视为面朝 +X')
            if kind == 'wander':
                reset = self.pending['resets'][path] if path in self.pending['resets'] else item.get('reset')
                own = (reset or {}).get('region')
                areas = sorted(p for p in self._areas() if p != path)
                options = ([f'随机复位区域（{own}）'] if own else []) + areas or ['（先添加抽象矩形或长方体）']
                current = draft.get('region')
                fields['region'] = gui.add_dropdown('区域', options, initial_value=current if current in options else options[0])
                top = max(float(v) for v in speed)
                fields['turn'] = gui.add_number('掉头加速度 (m/s²)', float(draft.get('turn', round(top * top / .04, 4))),
                                                min=.001, step=.05, hint='越出区域边缘后以此加速度折返；默认最高速时越出约 2 cm')
            if kind == 'path':
                points = draft.get('points', [])
                text = '  \n'.join(f'{i + 1}. ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})' for i, p in enumerate(points))
                gui.add_markdown(f'<small>路径点（环境坐标，m）：  \n{text or "尚无，至少需要两个"}</small>')
                gui.add_button('在视图中点击添加路径点', icon='map-pin-plus',
                               hint='连续点击添加；高度取实体当前高度；Esc 结束').on_click(lambda _: self._arm_path_point(path))
                remove = gui.add_button_group('', ['删除最后一点', '清空'])
                fields['loop'] = gui.add_checkbox('循环（不勾为往返）', draft.get('loop', True) is not False)
                fields['start'] = gui.add_dropdown('复位时位置', ['第一个点', '路径上随机'],
                                                   initial_value='路径上随机' if draft.get('start') == 'random' else '第一个点')
            if kind:
                gui.add_button('预览运动（8 秒）', icon='player-play').on_click(lambda _: self.preview_motion(path))
            note = ('选了运动后自动设为不受力（运动学）。' if not abstract else '') + \
                '每步由场景运动给出位置；task.py 的 set 若给出该实体，以 set 为准。'
            gui.add_markdown(f'<small>{note}</small>')

        def collect():
            kind = next(key for key, label in MOTIONS.items() if label == pick.value)
            if kind is None:
                return None
            motion = {'type': kind, 'speed': .1}
            if 'speed' in fields:
                low, high = (round(float(v), 4) for v in fields['speed'].value)
                motion['speed'] = high if low == high else [low, high]
                if fields['face'].value:
                    motion['face'] = True
            if kind == 'wander' and 'region' in fields:
                if fields['region'].value in self.items:  # else the entity's own reset region
                    motion['region'] = fields['region'].value
                motion['turn'] = round(float(fields['turn'].value), 4)
            if kind == 'path':
                motion['points'] = draft.get('points', [])
                if 'loop' in fields and not fields['loop'].value:
                    motion['loop'] = False
                if 'start' in fields and fields['start'].value == '路径上随机':
                    motion['start'] = 'random'
            return motion

        @self.guarded
        def changed(_, rebuild=False):
            self.checkpoint(('motion', path))
            self._store_motion(path, collect())
            if rebuild:
                self.open_section = title
                self._rebuild_inspector()
        pick.on_update(lambda event: changed(event, rebuild=True))
        for name, field in fields.items():
            field.on_update(lambda event, name=name: changed(event, rebuild=name == 'speed' and 'turn' not in fields))
        if draft.get('type') == 'path':
            @self.guarded
            def drop(event):
                self.checkpoint()
                points = list(self.motion_drafts.get(path, draft).get('points', []))
                points = points[:-1] if event.target.value == '删除最后一点' else []
                self._store_motion(path, {**collect(), 'points': points})
                self.open_section = title
                self._rebuild_inspector()
            remove.on_click(drop)

    def _store_motion(self, path, motion):
        """Keep the edited motion; a complete one becomes the pending scene edit (kinematic for an asset)."""
        from shared.assembly.motion import check
        self.motion_drafts[path] = motion
        if motion is None:
            self.pending['motion'][path] = None
        else:
            reset = self.pending['resets'][path] if path in self.pending['resets'] else self.items[path].get('reset')
            if motion.get('type') == 'path' and reset:
                # A path decides where each reset puts the entity.
                self.pending['resets'][path], reset = None, None
                self.report(f'{path} 沿路径运动：复位位置由路径决定，已移除其随机复位')
            try:
                check(path, motion, reset_region=(reset or {}).get('region'), has_reset=bool(reset))
            except ValueError as error:
                self.report(f'运动尚未生效：{error}')
                self._draw_motion_path()
                return
            self.pending['motion'][path] = motion
            if self.items[path]['category'] != 'abstract':
                self.pending['kinematic'][path] = True
        self.outliner_revision += 1
        self._draw_motion_path()
        self.report(f'{path} 的运动已暂存；保存后写入 scene.yaml')

    def _arm_path_point(self, path):
        def add(position):
            draft = dict(self.motion_drafts.get(path) or self.current_motion(path) or {'type': 'path', 'speed': .3})
            height = float(self.info[path]['node'].world()[2, 3])
            draft['points'] = [*draft.get('points', []), [round(position[0], 4), round(position[1], 4), round(height, 4)]]
            self.checkpoint()
            self._store_motion(path, draft)
            self.open_section = '运动'
            self._rebuild_inspector()
            self.arm_placement(f'{path} 的下一个路径点', add)  # keep adding until Esc
        self.arm_placement(f'{path} 的路径点', add)

    def _draw_motion_path(self):
        """The selected entity's path (points and segments) in the view."""
        if self.path_marker is not None:
            for handle in self.path_marker:
                handle.remove()
            self.path_marker = None
        if not self.selected or self.selected[0] != 'entity' or self.selected[1] not in self.items:
            return
        path = self.selected[1]
        motion = self.motion_drafts.get(path) or self.current_motion(path)
        if not motion or motion.get('type') != 'path' or not motion.get('points'):
            return
        points = np.asarray(motion['points'], float)
        closed = np.vstack([points, points[:1]]) if motion.get('loop', True) is not False and len(points) > 2 else points
        scene, handles = self.server.scene, []
        if len(closed) > 1:
            handles.append(scene.add_line_segments('/motion_path/segments', np.stack([closed[:-1], closed[1:]], 1),
                                                   (230, 120, 20), thickness=3., thickness_units='screen'))
        handles.append(scene.add_point_cloud('/motion_path/points', points, colors=(230, 120, 20), point_size=.03,
                                             point_shape='circle'))
        self.path_marker = handles

    @action
    def preview_motion(self, path):
        """Play the motion of ``path`` in the view for 8 s with the shared motion code, then put it back."""
        try:
            import torch
            from shared.assembly.motion import Motion, check
        except ImportError:
            raise ValueError('预览运动需要 torch；编辑器运行在没有 torch 的环境中') from None
        motion = self.motion_drafts.get(path) if path in self.motion_drafts else self.current_motion(path)
        if not motion:
            raise ValueError(f'{path} 没有设置运动')
        reset = self.pending['resets'][path] if path in self.pending['resets'] else self.items[path].get('reset')
        motion = check(path, motion, reset_region=(reset or {}).get('region'), has_reset=bool(reset))
        if motion['type'] == 'wander' and motion['region'] not in self._areas():
            raise ValueError(f'区域 {motion["region"]} 不是抽象矩形或长方体')
        node = self.info[path]['node']
        original, world = node.local.copy(), node.world()
        pose = F.matrix_pose(world)
        state = Motion(motion, pose, self._areas(), 1, 'cpu')
        state.reset([0])
        start = state.start([0])
        current = start if start is not None else torch.tensor([[*pose['position'], *pose['orientation']]])
        if self.motion_preview:
            self.motion_preview.set()
        stop = self.motion_preview = threading.Event()

        def run(current=current):
            try:
                for _ in range(240):
                    if stop.wait(1 / 30):
                        break
                    current = state.step(current, 1 / 30, torch.ones(1, dtype=torch.bool))
                    target = F.pose_matrix({'position': current[0, :3].tolist(), 'orientation': current[0, 3:7].tolist()})
                    with self.lock:
                        parent = node.parent.world() if node.parent is not None else np.eye(4)
                        node.local = np.linalg.inv(parent) @ target
                        self._update_frame(node)
            finally:
                with self.lock:
                    node.local = original
                    self._update_frame(node)
                    self._place_gizmo()
        threading.Thread(target=run, daemon=True).start()
        self.report(f'预览 {path} 的运动 8 秒（只在视图中，不改位置）')

    def _meshes_section(self, path):
        gui, title = self.server.gui, '射线目标'
        item = self.items[path]
        current = self.pending['meshes'].get(path, item.get('meshes', []))
        resolved = {_resolve(path, ref): ref for ref in current}
        candidates = [entry['path'] for entry in self.snapshot['entities']
                      if entry['category'] not in ('abstract', 'sensor', 'light') and entry['path'] != path]
        with self._section(title):
            gui.add_markdown('射线命中的实体（地面、地形）。Isaac Lab 3 要求恰好列出全部有形状的全局实体。'
                             + ('' if current else '  \n**当前没有目标，保存前需选择。**'))
            boxes = {other: gui.add_checkbox(self.items[other]['label'], other in resolved, hint=other)
                     for other in candidates}
            stage = gui.add_button('暂存（保存时写入）', color='green', icon='check')

        @self.guarded
        def execute(_):
            self.checkpoint()
            self.pending['meshes'][path] = [resolved.get(other, other) for other, box in boxes.items() if box.value]
            self.open_section = title
            self._rebuild_inspector()
            self.report(f'{path} 的射线目标已暂存；保存后写入 scene.yaml')
        stage.on_click(execute)

    def _rebuild_group_choices(self):
        for box in self.group_boxes.values():
            box.remove()
        self.group_boxes = {}
        with self.group_folder:
            for item in self.snapshot['entities']:
                info = self.info[item['path']]
                if not info['fixed'] and not item.get('mount') and not info['grouped']:
                    self.group_boxes[item['path']] = self.server.gui.add_checkbox(item['label'], False, hint=item['path'])

    @action
    def _create_group(self):
        paths = [path for path, box in self.group_boxes.items() if box.value]
        name = self.group_name.value.strip()
        if len(paths) < 2:
            raise ValueError('至少勾选两个实体，例如桌子和机械臂')
        if not name.isidentifier():
            raise ValueError('组合名称需是 Python 标识符')
        self._save_first()
        if name in self.snapshot.get('layout_groups', {}):
            raise ValueError('组合名称已存在')
        self._apply({'revision': self.snapshot['revision'], 'layout_groups': {name: {'entities': paths}}},
                    select=('group', name))
        self.report(f'已组成布局组合 {name}')

    @action
    def _ungroup(self, name):
        self._save_first()
        self._apply({'revision': self.snapshot['revision'], 'layout_groups': {name: None}}, select=None)

    # ---------------------------------------------------------- abstract entities
    def _add_abstract(self, shape, position=None):
        name = self.entity_name
        size = [float(v) for v in self.asset_size.value][:{'rectangle': 2, 'box': 3}.get(shape, 0)]
        self._save_first()
        entity = {'name': name, 'kind': 'abstract', 'shape': shape, 'size': size,
                  'pose': {'position': list(position or self.asset_position.value), 'orientation': [1, 0, 0, 0]}}
        root = self.snapshot['name']
        self._apply({'revision': self.snapshot['revision'], 'add_entities': [entity]},
                    select=('entity', f'{root}/{name}'))
        self.report(f'已添加抽象体 {name}（{ABSTRACT_SHAPES[shape]}）')

    # ------------------------------------------------------------ observations
    def _init_observations(self):
        from shared.assembly.control import observation_order
        snapshot = self.snapshot
        catalog = {item['id']: item for item in snapshot['catalog'].get('observations', [])}
        self.observation_meta = catalog
        self.observation_order, self.observation_enabled = {}, {}
        control = snapshot.get('control') or {}
        for role, component in control.get('components', {}).items():
            saved = list(component.get('observations', []))
            self.observation_order[role] = observation_order(saved, catalog)
            for key in saved:
                self.observation_enabled[(role, key)] = True
        self.train_role = control.get('train')
        self._rebuild_observations()

    @action
    def _set_train(self, role):
        if role == self.train_role:
            return
        self.checkpoint()
        self.train_role = role
        self._show_train()
        self.report(f'训练模块改为 {role}；保存后生效')

    def _show_train(self):
        """Mark the trained module on the switch and on its tab icon."""
        roles = list(self.observation_order)
        row = TRAIN_ROW.format(id=self.train_switch._impl.uuid)
        rules = [TRAIN_SWITCH_CSS.format(row=row)]
        for index, role in enumerate(roles, 1):
            trained = role == self.train_role
            self.observation_tabs[role].icon = 'flame' if trained else 'lock'
            caption = ' · '.join(filter(None, (ROLE_PURPOSE.get(role), '训练中' if trained else '固定')))
            rules.append(TRAIN_BUTTON_CSS.format(row=row, n=index, caption=json.dumps(caption, ensure_ascii=False)))
            if trained:
                rules.append(TRAIN_ON_CSS.format(row=row, n=index))
        self.train_style.content = '<style>' + ''.join(rules) + '</style>'

    def _rebuild_observations(self):
        for handle in self.observation_items:
            handle.remove()
        self.observation_items, self.observation_tabs, self.observation_boxes = [], {}, {}
        self.period_fields, self.dimension_fields = {}, {}
        gui, roles = self.server.gui, list(self.observation_order)
        components = (self.snapshot.get('control') or {}).get('components', {})
        boxes = []
        with self.observation_folder:
            if not roles:
                self.observation_items.append(gui.add_html(
                    '<div class="rl-status error" style="margin:0 8px 6px">此场景没有 control.components，'
                    '不是当前模板格式；编辑器只编辑新模板项目的组件观察。</div>'))
                return
            self.train_switch = gui.add_button_group('训练', roles, hint=(
                '橙色为被训练的模块：PPO 更新其参数。其余模块固定，只运行 policy.py 中的模型。'))
            self.train_switch.on_click(lambda event: self._set_train(event.target.value))
            self.train_style = gui.add_html('')
            tabs = gui.add_tab_group()
            self.observation_items += [self.train_switch, self.train_style, tabs]
            for index, role in enumerate(roles):
                with tabs.add_tab(f'{role} 的输入') as tab:
                    self.observation_tabs[role] = tab
                    self._period_field(role, components[role])
                    # A module reads earlier modules' interfaces, not its own or later ones.
                    boxes += self._observation_boxes(role, components, hidden=roles[index:])
            self._show_train()
            # Attribute path on the first line in monospace, the description below it.
            if boxes:
                selectors = [f'p:has(> label[for="{box._impl.uuid}"])' for box in boxes]
                self.observation_items.append(gui.add_html('<style>' + OBSERVATION_CSS.format(
                    labels=','.join(selectors), first=','.join(item + '::first-line' for item in selectors))
                    + OBSERVATION_GROUP_CSS.format(checked=','.join(f'input[id="{box._impl.uuid}"]:checked' for box in boxes))
                    + '</style>'))

    def _period_field(self, role, saved):
        period = self.pending['components'].get(role, {}).get('period', saved.get('period', ''))
        field = self.server.gui.add_text('决策周期 (s)', str(period),
                                         hint='须为物理步 simulation.dt 的整数倍；executor 周期须整除 planner 周期')
        field.on_update(lambda _: (self.checkpoint(('component', role, 'period')),
                                   self.pending['components'].setdefault(role, {}).__setitem__('period', field.value)))
        self.period_fields[role] = field
        dimension = self.pending['components'].get(role, {}).get('dimension', saved.get('dimension', 'auto'))
        size = self.server.gui.add_text('输出维度', str(dimension), hint=(
            '模型 forward 返回的末维；planner 须为正整数，executor 可写 auto（等于控制目标关节总数）'))
        size.on_update(lambda _: (self.checkpoint(('component', role, 'dimension')),
                                  self.pending['components'].setdefault(role, {}).__setitem__('dimension', size.value)))
        self.dimension_fields[role] = size

    def _observation_boxes(self, profile, components, hidden=()):
        """One folder per obs object (obs.robot), a checkbox per attribute of the given module.

        Rows follow the catalogue order on every page; interfaces of `hidden` modules are
        left out unless already selected."""
        gui, groups, boxes = self.server.gui, {}, []
        order = self.observation_order[profile]
        for key in [key for key in self.observation_meta if key in order] + [key for key in order if key not in self.observation_meta]:
            if key.partition('.')[0] in hidden and not self.observation_enabled.get((profile, key)):
                continue
            meta = self.observation_meta.get(key, {})
            source = meta.get('source_groups', {}).get('default', {})
            group, name = observation_group(key, source, components)
            groups.setdefault(group, []).append((key, name, meta, source))
        for label, rows in groups.items():
            with gui.add_folder(label, expand_by_default=any(self.observation_enabled.get((profile, key)) for key, *_ in rows)):
                for key, name, meta, source in rows:
                    enabled = bool(self.observation_enabled.get((profile, key)))
                    available = source.get('available', meta.get('available', enabled)) and meta.get('observation_selectable', True)
                    hint = f'obs.{key}：' + meta.get('description', '')
                    if source.get('missing'):
                        hint += '　缺少：' + ', '.join(source['missing'])
                    # Rows show the attribute inside the folder's object in English, then the
                    # description, shape and unit.
                    box = gui.add_checkbox(f'{name}\n{observation_caption(meta)}'.strip(), enabled,
                                           disabled=not available and not enabled, hint=hint)
                    boxes.append(box)
                    self.observation_boxes[(profile, key)] = box
                    box.on_update(lambda _, key=key, box=box: (
                        self.checkpoint(), self.observation_enabled.__setitem__((profile, key), box.value)))
        return boxes

    # ----------------------------------------------------------------- control
    def bindings(self):
        from shared.assembly.control import overlay_outputs
        snapshot = self.snapshot
        saved = (snapshot.get('control') or {}).get('output', {})
        return overlay_outputs(saved, self.pending['actions'])

    def measured_joints(self):
        result = {item['native_key']: list(self.initial_display(item['path']).get('joint_positions', {}))
                  for item in self.snapshot['entities'] if item.get('native_key')}
        result.update({name: item['joint_names'] for name, item in
                       (self.snapshot.get('interface') or {}).get('articulations', {}).items()})
        return result

    def _refresh_control(self):
        for handle in self.control_items:
            handle.remove()
        self.control_items = []
        gui, measured = self.server.gui, self.measured_joints()
        self.syncing += 1
        try:
            robots = {item['native_key'] for item in self.snapshot['entities'] if item['category'] == 'robot'}
            keys = sorted(key for key, joints in measured.items() if joints or key in robots)
            self.target_asset.options = keys or [NONE]
            if self.target_asset.value not in self.target_asset.options:
                self.target_asset.value = self.target_asset.options[0]
        finally:
            self.syncing -= 1
        self._suggest_target_name()
        with self.control_folder:
            if not self.snapshot.get('control'):
                self.control_items.append(gui.add_html(
                    '<div class="rl-status error" style="margin:0 8px 6px">此场景没有 control 段，不是当前模板格式。</div>'))
                return
            bindings = self.bindings()
            if not bindings:
                self.control_items.append(gui.add_markdown('尚无控制目标；在下方新增。'))
            css = []
            for name, binding in bindings.items():
                joints = list(binding['joint_names'])
                # The ✕ sits on the folder's header line, so a target is removed without expanding it.
                remove = gui.add_button('', icon='x', color='red', hint=f'删除控制目标 {name}')
                remove.on_click(lambda _, name=name: self._remove_target(name))
                css.append(CONTROL_REMOVE_CSS.format(id=remove._impl.uuid))
                mark = ' *' if name in self.pending['actions'] else ''
                folder = gui.add_folder(f'{name} · {CONTROL_TYPES.get(binding["type"], binding["type"]).split(" ")[0]}'
                                        f' · {binding["asset_name"]} · {len(joints)} 维{mark}', expand_by_default=False)
                self.control_items += [remove, folder]
                with folder:
                    kind = gui.add_dropdown('类型', list(CONTROL_TYPES.values()), initial_value=CONTROL_TYPES[binding['type']])
                    kind.on_update(lambda _, name=name, kind=kind: None if self.syncing else self._set_target_type(name, kind.value))
                    available = measured.get(binding['asset_name'], [])
                    if not available:
                        gui.add_markdown('尚无实测关节列表（等待原生网格导出）')
                    for joint in [*available, *(j for j in joints if j not in available)]:
                        index = joints.index(joint) if joint in joints else None
                        box = gui.add_checkbox(joint, index is not None,
                                               hint=f'动作第 {index} 维' if index is not None else '未控制')
                        box.on_update(lambda _, name=name, joint=joint, box=box:
                                      None if self.syncing else self._toggle_joint(name, joint, box.value))
            if css:
                self.control_items.append(gui.add_html('<style>' + ''.join(css) + '</style>'))

    def _suggest_target_name(self):
        from shared.assembly.control import free_target_name
        self.syncing += 1
        try:
            asset = self.target_asset.value
            self.target_name.value = free_target_name(self.bindings(), asset if asset != NONE else 'arm')
        finally:
            self.syncing -= 1

    def _stage_targets(self, changes, message):
        """Validate the effective bindings with these changes, then stage them for saving."""
        from shared.assembly.control import (overlay_outputs, require_listed_joints, validate_outputs,
                                             validate_target_conflicts)
        if not self.snapshot.get('control'):
            raise ValueError('此场景没有 control 段，不是当前模板格式')
        proposed = overlay_outputs(self.bindings(), changes)
        require_listed_joints({name: binding for name, binding in changes.items() if binding})
        validate_outputs(proposed)
        validate_target_conflicts(proposed, self.measured_joints())
        self.checkpoint()
        self.pending['actions'].update(changes)
        self._refresh_control()
        self.report(message)

    @action
    def _toggle_joint(self, name, joint, enabled):
        try:
            binding = self.bindings()[name]
            joints = list(binding['joint_names'])
            if enabled and joint not in joints:
                measured = self.measured_joints().get(binding['asset_name'], [])
                # A list kept in articulation order stays in that order; a custom order gets the joint appended.
                if joints == [item for item in measured if item in joints] and joint in measured:
                    joints = [item for item in measured if item in joints or item == joint]
                else:
                    joints.append(joint)
            elif not enabled:
                joints = [item for item in joints if item != joint]
            if not joints:
                raise ValueError(f'{name} 至少要控制一个关节；不需要该目标时点 ✕ 删除')
            self._stage_targets({name: {**binding, 'joint_names': joints}},
                                f'控制目标 {name}：{len(joints)} 维；保存后写入 scene.yaml')
        except Exception:
            self._refresh_control()  # Restore the checkbox to the effective binding.
            raise

    @action
    def _set_target_type(self, name, label):
        binding = self.bindings()[name]
        kind = next(key for key, value in CONTROL_TYPES.items() if value == label)
        try:
            self._stage_targets({name: {**binding, 'type': kind}}, f'控制目标 {name} 改为{label}；保存后写入 scene.yaml')
        except Exception:
            self._refresh_control()
            raise

    @action
    def _add_target(self):
        asset, name = self.target_asset.value, self.target_name.value.strip()
        if asset == NONE:
            raise ValueError('场景中没有可控制的机器人')
        if name in self.bindings():
            raise ValueError(f'控制目标 {name} 已存在')
        from shared.assembly.control import default_target_joints
        controlled = {joint for binding in self.bindings().values() if binding['asset_name'] == asset
                      for joint in binding['joint_names']}
        path = next((item['path'] for item in self.snapshot['entities'] if item.get('native_key') == asset), None)
        kinematics = (self.info.get(path) or {}).get('kinematics') or {}
        joints = default_target_joints(self.measured_joints().get(asset, []), controlled, kinematics.get('joints', ()))
        if not joints:
            raise ValueError(f'{asset} 没有尚未控制的关节' if controlled else
                             f'{asset} 的关节名尚未读出（等待原生网格导出）')
        kind = next(key for key, value in CONTROL_TYPES.items() if value == self.target_type.value)
        rest = len(self.measured_joints().get(asset, [])) - len(controlled) - len(joints)
        self._stage_targets({name: {'type': kind, 'asset_name': asset, 'joint_names': joints}},
                            f'控制目标 {name} 已添加（{len(joints)} 维）'
                            + (f'，另有 {rest} 个类型不同的关节未选，可勾选或另建目标' if rest > 0 else '')
                            + '；同步 policy.py 的 decode_action，保存后写入 scene.yaml')

    @action
    def _remove_target(self, name):
        if name not in self.bindings():
            raise ValueError(f'控制目标 {name} 不存在')
        self.checkpoint()
        self.pending['actions'][name] = None
        self._refresh_control()
        self.report(f'控制目标 {name} 将在保存后删除；记得同步 policy.py 的 decode_action')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--project')
    target.add_argument('--scene', type=Path)
    parser.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 only on a trusted network')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--no-native-preview', action='store_true', help='Do not start the Isaac geometry export')
    parser.add_argument('--library-port', type=int, help='SimReady asset library page (default: editor port + 1, '
                                                         'or the next free one)')
    args = parser.parse_args(argv)
    scene = args.scene or ROOT / 'projects' / args.project / 'scene.yaml'
    if not Path(scene).is_file():
        parser.error(f'No scene.yaml: {scene}')
    editor = Editor(scene, host=args.host, port=args.port, native_preview=not args.no_native_preview,
                    library_port=args.library_port)
    print(f'RLinf scene editor: http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{editor.server.get_port()}')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        editor.server.stop()
        editor.discard_copy()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
