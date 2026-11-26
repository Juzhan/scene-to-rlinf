"""Small, source-preserving scene editing functions shared by Blender and CLI.

An editing session tracks the scene and referenced assemblies. Expanded entities
are for display only: changes to an assembly's child become instance overrides.
No simulator, Blender, training process or runtime state is involved.
"""
from copy import deepcopy
from hashlib import sha256
from io import StringIO
from math import isfinite
import os
from pathlib import Path
import re
import tempfile

from ruamel.yaml import YAML


class EditConflict(ValueError):
    """An input changed since the editor loaded it; reload before saving."""


def editing_copy(scene):
    """The editor's working copy of ``scene``: beside it, so project-relative files resolve.

    Every edit, native export and backend check reads this copy; scene.yaml changes only on save.
    """
    scene = Path(scene).resolve()
    return scene.with_name(f".{scene.name}.{os.getpid()}.editing")


def edited_source(path):
    """The scene file a working copy edits; any other path is returned unchanged."""
    path = Path(path)
    match = re.fullmatch(r"\.(.+)\.\d+\.editing", path.name)
    return path.with_name(match.group(1)) if match else path


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    # ruamel preserves scalar formatting with subclasses; OmegaConf accepts
    # built-in primitive types only (not ScalarFloat/DoubleQuotedScalarString).
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    return value


def _merge(target, patch):
    for key, value in patch.items():
        if key in ("reset", "initial_state", "motion"):
            target[key] = deepcopy(value)
        elif key == "parameters" and isinstance(value, dict):
            # Each parameter is one value (a joint -> coupling map is replaced, not merged); None removes it.
            parameters = target[key] = deepcopy(target.get(key) or {})
            for name, item in value.items():
                if item is None:
                    parameters.pop(name, None)
                else:
                    parameters[name] = deepcopy(item)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            # Detach YAML aliases: one instance's edit must not edit another.
            target[key] = deepcopy(target[key])
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _pose(value):
    from .spec import Pose
    return {"position": list(Pose(**value).position), "orientation": list(Pose(**value).orientation)}


def _rename_in_place(value, rename):
    """Rewrite strings and mapping keys of a round-trip YAML tree without losing order or comments."""
    if isinstance(value, dict):
        for index, key in enumerate(list(value)):
            item = value[key]
            replaced = _rename_in_place(item, rename)
            new = rename(key) if isinstance(key, str) else key
            if new != key:
                del value[key]
                if hasattr(value, 'insert'):
                    value.insert(index, new, replaced)
                else:
                    value[new] = replaced
            elif replaced is not item:
                value[key] = replaced
    elif isinstance(value, list):
        for index, item in enumerate(value):
            replaced = _rename_in_place(item, rename)
            if replaced is not item:
                value[index] = replaced
    elif isinstance(value, str):
        new = rename(value)
        return value if new == value else new
    return value


def _check_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("Names must start with a letter or underscore and contain only letters, digits or underscores")


def policy_section(data):
    """``(section, layers, inputs key)`` of a scene: the ``policy:`` with its ``layers`` (``inputs``), or the older
    ``control:`` with its ``components`` (``observations``); ``(None, {}, None)`` without either."""
    if 'policy' in data:
        return data['policy'], data['policy'].get('layers', {}), 'inputs'
    if 'control' in data:
        return data['control'], data['control'].get('components', {}), 'observations'
    return None, {}, None


class SceneDocument:
    """Edit one scene.yaml, retaining its comments, templates and unknown fields."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.yaml = YAML(typ="rt")
        self.yaml.preserve_quotes = True
        self.yaml.width = 120
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.files = {}
        self.data = self._read(self.path)
        if self.data.get("schema_version") != 1 or not isinstance(self.data.get("entities"), list):
            raise ValueError("Expected schema_version: 1 and an entities list")
        self.original = deepcopy(self.data)
        self._index()

    def _read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        previous = self.files.get(str(path))
        if previous is not None and previous != digest:
            raise EditConflict(f"Changed outside the editor: {path}. Reload before saving.")
        self.files[str(path)] = digest
        data = self.yaml.load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return data

    def _expand(self, raw, directory, stack=()):
        if "assembly" in raw:
            path = (directory / raw["assembly"]).resolve()
            if path in stack:
                raise ValueError(f"Cyclic assembly: {path}")
            template = self._read(path)
            node = self._expand(template["root"], path.parent, (*stack, path))
            _merge(node, {key: value for key, value in raw.items()
                          if key not in ("assembly", "overrides", "connection_overrides")})
            targets = {}
            connections = {}

            def index(item, relative):
                targets[relative or "."] = item
                for joint in item.get("connections", []):
                    connections[(relative + "/" if relative else "") + joint["name"]] = joint
                for child in item.get("children", []):
                    index(child, (relative + "/" if relative else "") + child["name"])
            index(node, "")
            for key, values in raw.get("overrides", {}).items():
                if key not in targets:
                    raise ValueError(f"Unknown assembly override: {key}")
                _merge(targets[key], values)
            for key, values in raw.get("connection_overrides", {}).items():
                if key not in connections:
                    raise ValueError(f"Unknown connection override: {key}")
                _merge(connections[key], values)
            return node
        result = _plain(deepcopy(raw))
        result["children"] = [self._expand(child, directory, stack) for child in raw.get("children", [])]
        return result

    def _index(self):
        self.entities = {}
        self.targets = {}
        world = self.data.get("name", "world")

        def visit(raw, parent, source=None, instance_path=None):
            path = parent + "/" + raw["name"]
            expanded = self._expand(raw, self.path.parent) if source is None else raw
            if path in self.entities:
                raise ValueError(f"Duplicate entity: {path}")
            self.entities[path] = expanded
            if source is not None:
                self.targets[path] = (source, path[len(instance_path)+1:])
            elif "assembly" in raw:
                source, instance_path = raw, path
                self.targets[path] = (raw, ".")
            else:
                self.targets[path] = (raw, None)
            children = expanded.get("children", []) if source is not None else raw.get("children", [])
            for child in children:
                visit(child, path, source, instance_path)
        for raw in self.data["entities"]:
            visit(raw, world)

    @property
    def revision(self):
        return dict(self.files)

    def check_revision(self, expected=None):
        expected = self.files if expected is None else expected
        if set(expected) != set(self.files):
            raise EditConflict("Scene references changed; reload the project before saving")
        for filename, digest in expected.items():
            path = Path(filename)
            if not path.exists() or sha256(path.read_bytes()).hexdigest() != digest:
                raise EditConflict(f"Changed outside the editor: {filename}. Reload before saving.")

    def set_entity(self, path, changes, *, frame="local"):
        if frame != "local":
            raise ValueError("Entity edits require parent-local poses; mounted sensors require body-local poses")
        if path not in self.targets:
            raise ValueError(f"Unknown entity: {path}")
        allowed = {"pose", "parameters", "tags", "initial_state", "reset", "mount", "meshes", "kinematic",
                   "motion", "visible", "kind", "gravity_compensation"}
        if changes.get("kind", "robot") not in ("robot", "entity"):
            raise ValueError("kind can only switch between robot (driven by the policy) and entity")
        if self.entities[path].get("kind") == "abstract":
            allowed = {"pose", "tags", "reset", "mount", "shape", "size", "motion"}
        if changes.keys() - allowed:
            raise ValueError(f"Unsupported entity edits: {sorted(changes.keys() - allowed)}")
        changes = deepcopy(changes)
        if "shape" in changes or "size" in changes:
            from .abstract import check_geometry
            shape = changes.get("shape", self.entities[path].get("shape"))
            size = list(check_geometry(path, shape, changes.get("size", self.entities[path].get("size", []))))
            raw = self.targets[path][0]
            raw["shape"] = shape
            raw.pop("size", None)
            if size:
                raw["size"] = size
            changes.pop("shape", None)
            changes.pop("size", None)
        if "meshes" in changes:
            meshes = changes["meshes"] or None
            if meshes is not None and (not isinstance(meshes, list) or not all(isinstance(m, str) for m in meshes)):
                raise ValueError("Sensor mesh targets must be a list of entity paths")
            changes["meshes"] = meshes
        if "reset" in changes:
            # Editors place roots; joint ranges written in scene.yaml survive a placement edit.
            joints = (self.entities[path].get("reset") or {}).get("joint_positions")
            if joints and "joint_positions" not in (changes["reset"] or {}):
                changes["reset"] = {**(changes["reset"] or {}), "joint_positions": deepcopy(_plain(joints))}
        if "kinematic" in changes:
            if self.targets[path][1] is not None:
                raise ValueError(f"{path}: kinematic is set on scene-owned entities, not on assembly parts")
            if not isinstance(changes["kinematic"], bool):
                raise ValueError("kinematic is true or false")
            if not changes["kinematic"]:  # The default: written only when true.
                changes["kinematic"] = None
        if "motion" in changes and changes["motion"] is not None:
            from .motion import check
            reset = changes.get("reset", self.entities[path].get("reset")) or None
            check(path, changes["motion"], reset_region=(reset or {}).get("region"), has_reset=reset is not None)
        if "visible" in changes:
            if self.targets[path][1] is not None:
                raise ValueError(f"{path}: visible is set on scene-owned entities, not on assembly parts")
            if not isinstance(changes["visible"], bool):
                raise ValueError("visible is true or false")
            if changes["visible"]:  # The default: written only when false.
                changes["visible"] = None
        if "gravity_compensation" in changes:
            if not isinstance(changes["gravity_compensation"], bool):
                raise ValueError("gravity_compensation is true or false")
            if not changes["gravity_compensation"]:  # The default: written only when true.
                changes["gravity_compensation"] = None
        for key in ("reset", "mount", "meshes", "kinematic", "motion", "visible", "gravity_compensation"):
            # None clears an optional rule (static, unmounted entity, no ray targets, dynamic body, no motion,
            # rendered).
            if key in changes and changes[key] is None and self.targets[path][1] is None:
                self.targets[path][0].pop(key, None)
                del changes[key]
        if "pose" in changes:
            _pose(changes["pose"])
        if "parameters" in changes and not isinstance(changes["parameters"], dict):
            raise ValueError("Entity parameters must be a mapping")
        raw, relative = self.targets[path]
        if relative is not None:
            direct = {key: value for key, value in changes.items() if relative == "." and key in ("pose", "parameters")}
            indirect = {key: value for key, value in changes.items() if key not in direct}
            _merge(raw, direct)
            if indirect:
                overrides = raw.setdefault("overrides", {})
                target = deepcopy(overrides.get(relative, {}))
                overrides[relative] = _merge(target, indirect)
        else:
            _merge(raw, changes)
        self._index()

    def set_connection(self, name, value):
        _check_name(name)
        if value is not None and (not isinstance(value, dict) or value.keys() - {'child', 'parent', 'parent_pose', 'child_pose'}):
            raise ValueError('Connection requires child/parent endpoints and optional anchor poses')
        joints = self.data.setdefault("connections", [])
        existing = next((item for item in joints if item["name"] == name), None)
        if existing is not None:
            joints.remove(existing)
        if value is not None:
            joints.append({"name": name, **deepcopy(value)})
        self._index()

    def _removal_filter(self, removed):
        """Entity keys and abstract names in ``removed``, and whether an observation term reads one."""
        from .entity_information import parse_signal
        keys = {path.split('/', 1)[1].replace('/', '__') for path in removed}
        abstract = {self.entities[path]['name'] for path in removed
                    if path in self.entities and self.entities[path].get('kind') == 'abstract'}

        def dropped(term):
            return bool(parse_signal(term) and parse_signal(term)[0] in keys) or term.split('.', 1)[0] in abstract
        return keys, abstract, dropped

    def removal_effects(self, removed):
        """Control targets and observations that removing the ``removed`` paths takes with it."""
        keys, _, dropped = self._removal_filter(removed)
        control, layers, key = policy_section(self.data)
        control = control or {}
        terms = [term for component in layers.values() for term in component.get(key, [])]
        for value in self.data.get('observations', {}).values():
            terms += value.get('terms', []) if isinstance(value, dict) else value
        if 'output' in control:
            outputs = sorted(name for name, binding in control['output'].items() if binding['asset_name'] in keys)
        else:  # automatic targets: the removed robots (control.automatic_outputs)
            from .control import actuated
            modes = control.get('modes', {})
            outputs = sorted(key for path, entry in self.entities.items()
                             if (key := path.split('/', 1)[1].replace('/', '__')) in keys
                             and actuated(entry) and modes.get(key) != 'none') if control else []
        return {'outputs': outputs,
                'observations': sorted({term for term in terms if dropped(term)})}

    def remove_entities(self, paths):
        """Remove scene-owned entities, cleaning generated bindings.

        Scene-owned children move up one level and keep their world pose; parts of an
        assembly instance belong to its template and go with it. Layout children only
        lose their authoring parent. Mounted or physically attached entities must be
        detached by the caller first, since only it knows their live world pose.
        """
        from posixpath import normpath, join, dirname
        from .spec import Pose
        requested = set(paths)
        roots = {path for path in requested if not any(path.startswith(other + '/') for other in requested if other != path)}
        for path in roots:
            if path not in self.targets:
                raise ValueError(f"Unknown entity: {path}")
            if self.targets[path][1] not in (None, "."):
                raise ValueError("模板内部实体请在模板中删除；可直接删除整个实例")
        promoted = {}  # old path -> new path of children moving up a level
        for root in roots:
            if self.targets[root][1] is None:
                for child in self.targets[root][0].get('children', []):
                    old = root + '/' + child['name']
                    if old not in requested:
                        promoted[old] = dirname(root) + '/' + child['name']
        removed = {path for path in self.entities if any(path == root or path.startswith(root + '/') for root in roots)
                   and not any(path == old or path.startswith(old + '/') for old in promoted)}
        for old, new in promoted.items():
            if new in self.entities and new not in removed or list(promoted.values()).count(new) > 1:
                raise ValueError(f"{old} 上移后与已有实体 {new} 重名；请先删除或改名其中之一")
        keys, abstract, dropped = self._removal_filter(removed)
        def prune(nodes):
            for node in list(nodes):
                if any(self.targets[path][0] is node for path in roots):
                    nodes.remove(node)
                else:
                    prune(node.get('children', []))
        # Mounted sensors depend on a physical parent. Require an explicit choice
        # instead of silently deleting them or changing their coordinate system.
        for path, node in self.entities.items():
            mount = node.get('mount')
            if path not in removed and mount:
                ref = mount['entity']
                ref = normpath(join(path, ref)) if ref.startswith('.') else ref
                if ref in removed:
                    raise ValueError(f"{path} 挂载在被删除的实体上；请先解除挂载（网页编辑器删除时会自动保留其位置）")
        retargeted = {}
        for path, node in self.entities.items():
            # Ray targets that go away are dropped; a sensor left without any needs new targets.
            meshes = node.get('meshes')
            if path in removed or not meshes:
                continue
            kept = [ref for ref in meshes if (normpath(join(path, ref)) if ref.startswith('.') else ref) not in removed]
            if len(kept) != len(meshes):
                if self.targets[path][1] is not None:
                    raise ValueError(f"{path} 由模板定义，以被删除的实体为射线目标；请在模板中修改")
                retargeted[path] = kept
        for path, node in self.entities.items():
            region = (node.get('reset') or {}).get('region')
            if path not in removed and region in removed:
                raise ValueError(f"{path} 在 {region} 内随机放置；先修改其复位方式，再删除该区域")
        for joint in self.data.get('connections', []):
            if (joint.get('parent') or {}).get('entity') in removed and joint['child']['entity'] not in removed:
                raise ValueError(f"{joint['child']['entity']} 固定在被删除的实体上；请先解除连接"
                                 "（网页编辑器删除时会自动保留其位置）")
        _, layers, key = policy_section(self.data)
        for role, component in layers.items():
            if component.get(key) and all(dropped(term) for term in component[key]):
                raise ValueError(f"删除后 {role} 没有任何观察输入；请先在观察页为 {role} 选择其他输入，再删除")
        def container(path):
            parent = dirname(path)
            return self.data['entities'] if parent == self.data.get('name', 'world') else self.targets[parent][0]['children']
        for root in roots:
            raw = self.targets[root][0]
            moving = [child for child in raw.get('children', []) if root + '/' + child['name'] in promoted]
            if not moving:
                continue
            frame = Pose(**_pose(raw['pose'])) if raw.get('pose') else Pose()
            siblings = container(root)
            at = next(index for index, node in enumerate(siblings) if node is raw)
            for offset, child in enumerate(moving, 1):
                # Composition poses are relative to the logical parent: fold the removed parent in.
                world = frame.compose(Pose(**_pose(child['pose'])) if child.get('pose') else Pose())
                if child.get('pose') or raw.get('pose'):
                    child['pose'] = {'position': list(world.position), 'orientation': list(world.orientation)}
                siblings.insert(at + offset, child)
            # Internal joints between surviving children become project connections.
            for joint in raw.get('connections', []):
                ends = {end: normpath(join(root, joint[end]['entity'])) for end in ('child', 'parent') if joint.get(end)}
                if ends and all(path in promoted or any(path.startswith(old + '/') for old in promoted) for path in ends.values()):
                    moved = deepcopy(_plain(joint))
                    moved['name'] = root.split('/', 1)[1].replace('/', '__') + '__' + joint['name']
                    for end, path in ends.items():
                        moved[end]['entity'] = path
                    self.data.setdefault('connections', []).append(moved)
        for path, kept in retargeted.items():
            if kept:
                self.targets[path][0]['meshes'] = kept
            else:
                self.targets[path][0].pop('meshes', None)
        prune(self.data['entities'])
        shown = self.data.get('visualization', {}).get('abstract')
        if shown is not None:
            self.data['visualization']['abstract'] = [name for name in shown if name not in abstract]
        self.data['connections'] = [j for j in self.data.get('connections', [])
                                    if not any((j.get(end) or {}).get('entity') in removed for end in ('parent', 'child'))]
        authoring = self.data.get('authoring', {})
        for name, group in list(authoring.get('groups', {}).items()):
            group['entities'] = [path for path in group['entities'] if path not in removed]
            if not group['entities']:
                del authoring['groups'][name]
        for path, value in list(authoring.get('parents', {}).items()):
            if path in removed or value['entity'] in removed:
                del authoring['parents'][path]
        for key in ('labels', 'selectable'):
            for path in list(authoring.get(key, {})):
                if path in removed:
                    del authoring[key][path]
        for profile, value in list(self.data.get('observations', {}).items()):
            terms = value.get('terms', []) if isinstance(value, dict) else value
            kept = [term for term in terms if not dropped(term)]
            if kept == terms:
                continue
            if not kept:
                del self.data['observations'][profile]
            elif isinstance(value, dict):
                value['terms'] = kept
                for term in set(terms) - set(kept):
                    value.get('overrides', {}).pop(term, None)
            else:
                self.data['observations'][profile] = kept
        control, layers, key = policy_section(self.data)
        control = control or {}
        for component in layers.values():
            component[key] = [term for term in component.get(key, []) if not dropped(term)]
            if isinstance(component.get('history'), dict):
                component['history'] = {k: v for k, v in component['history'].items() if not dropped(k)}
                if not component['history']:
                    component.pop('history')
        for name, binding in list(control.get('output', {}).items()):
            if binding['asset_name'] in keys:
                del control['output'][name]
        for name in [name for name in control.get('modes', {}) if name in keys]:
            del control['modes'][name]
        if 'modes' in control and not control['modes']:
            del control['modes']
        for name, action in list(self.data.get('actions', {}).items()):
            if action.get('asset_name') in keys:
                del self.data['actions'][name]
        materials = self.data.get('world', {}).get('contact_materials', {})
        for entity in list(materials):
            if entity.replace('.', '__') in keys:
                del materials[entity]
        if promoted:
            world = self.data.get('name', 'world') + '/'
            names = [(old, new, old.removeprefix(world), new.removeprefix(world)) for old, new in promoted.items()]

            def rename(text):
                for old, new, old_relative, new_relative in names:
                    old_key, new_key = old_relative.replace('/', '__'), new_relative.replace('/', '__')
                    old_dot, new_dot = old_relative.replace('/', '.'), new_relative.replace('/', '.')
                    if text == old or text.startswith(old + '/'):
                        return new + text[len(old):]
                    if text.startswith(old_dot + '.'):
                        return new_dot + text[len(old_dot):]
                    if text == old_key or text.startswith(old_key + '__'):
                        return new_key + text[len(old_key):]
                    if text.startswith('entity__' + old_key + '__'):
                        return 'entity__' + new_key + text[len('entity__' + old_key):]
                return text
            _rename_in_place(self.data, rename)
        self._index()

    def rename_entity(self, path, name):
        """Rename the entity at ``path`` to ``name`` and every reference to it: paths (mounts, parents, regions,
        authoring), observation names (``name.joint_position``), control modes and native keys. A plain string
        equal to the old name is left alone unless it is a mapping key: a factory may share it (franka_panda).
        Policy and task code read ``obs.<name>`` and must follow by hand."""
        from posixpath import dirname
        if path not in self.entities:
            raise ValueError(f'场景中没有 {path}')
        if not (name.isascii() and name.isidentifier()) or name.startswith('_'):
            raise ValueError(f'实体名须是 Python 标识符（字母、数字、下划线，不以下划线开头）：{name!r}')
        raw, inner = self.targets[path]
        if inner is not None:
            raise ValueError(f'{path} 由模板定义，请在模板中改名')
        new = dirname(path) + '/' + name
        if new == path:
            return
        if new in self.entities:
            raise ValueError(f'已有实体 {new}')
        world = self.data.get('name', 'world') + '/'
        old_relative, new_relative = path.removeprefix(world), new.removeprefix(world)
        old_dot, new_dot = old_relative.replace('/', '.'), new_relative.replace('/', '.')
        old_key, new_key = old_relative.replace('/', '__'), new_relative.replace('/', '__')

        def value(text):
            if text == path or text.startswith(path + '/'):
                return new + text[len(path):]
            if text.startswith(old_dot + '.'):
                return new_dot + text[len(old_dot):]
            if text.startswith(old_key + '__'):
                return new_key + text[len(old_key):]
            if text.startswith('entity__' + old_key + '__'):
                return 'entity__' + new_key + text[len('entity__' + old_key):]
            return text

        def walk(item):
            if isinstance(item, dict):
                for index, key in enumerate(list(item)):
                    child = walk(item[key])
                    renamed = (new_key if key == old_key else value(key)) if isinstance(key, str) else key
                    if renamed != key:
                        del item[key]
                        if hasattr(item, 'insert'):
                            item.insert(index, renamed, child)
                        else:
                            item[renamed] = child
                    elif child is not item[key]:
                        item[key] = child
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    renamed = walk(child)
                    if renamed is not child:
                        item[index] = renamed
            elif isinstance(item, str):
                renamed = value(item)
                return item if renamed == item else renamed
            return item
        raw['name'] = name
        labels = self.data.get('authoring', {}).get('labels', {})
        labels.pop(path, None)  # it is shown by its new name
        walk(self.data)
        self._index()

    def set_layout_parent(self, path, value):
        """Editor-only parenting; simulator consumes the baked entity poses."""
        parents = self.data.setdefault('authoring', {}).setdefault('parents', {})
        if value is None:
            parents.pop(path, None)
            return
        if not isinstance(value, dict) or value.keys() - {'entity', 'body'}:
            raise ValueError('Layout parent requires entity and optional body')
        for ref in (path, value.get('entity')):
            if ref not in self.entities:
                raise ValueError('布局父子关系请选择场景中的实体')
        if self.entities[path].get('mount') or any(j['child']['entity'] == path for j in self.data.get('connections', [])):
            raise ValueError('请先解除物理安装，再设置布局父对象')
        if any(path in g['entities'] for g in self.data.get('authoring', {}).get('groups', {}).values()):
            raise ValueError('请先解散布局组合')
        proposed = {**parents, path: deepcopy(value)}
        for start in proposed:
            seen = set()
            current = start
            while current in proposed:
                if current in seen:
                    raise ValueError('父子关系不能形成循环')
                seen.add(current)
                current = proposed[current]['entity']
        parents[path] = deepcopy(value)

    def set_layout_group(self, name, value):
        _check_name(name)
        groups = self.data.setdefault("authoring", {}).setdefault("groups", {})
        if value is None:
            groups.pop(name, None)
        else:
            if not isinstance(value, dict) or value.keys() - {"entities", "pose"}:
                raise ValueError("Layout groups require entities and an optional pose")
            members = value.get("entities", [])
            if not members or len(members) != len(set(members)):
                raise ValueError("Choose unique entity roots for the layout group")
            if 'pose' in value:
                _pose(value['pose'])
            fixed = {joint['child']['entity'] for joint in self.data.get('connections', [])}
            for path in members:
                if path not in self.entities or path.rsplit('/', 1)[0] != self.data.get('name', 'world'):
                    raise ValueError("Layout groups currently accept top-level entities only")
                if self.entities[path].get('mount'):
                    raise ValueError("Group the mounted sensor's owner instead")
                if path in fixed:
                    raise ValueError('Group the physical attachment parent, not its fixed child')
            occupied = {path for key, group in groups.items() if key != name for path in group['entities']}
            if occupied.intersection(members):
                raise ValueError("Entity already belongs to another layout group")
            groups[name] = deepcopy(value)
        self._index()

    def add_layer(self, name, *, model, period, inputs):
        """Split a one-layer policy: a new layer ``name`` before it, which it then reads (scene.yaml only; the
        code of the new layer is policy.py's)."""
        policy = self.data.get('policy')
        if policy is None:
            raise ValueError('只有 policy: 写法的项目能增加一层（control: 是旧写法）')
        layers = policy['layers']
        if len(layers) != 1:
            raise ValueError('策略已经有两层')
        if not isinstance(name, str) or not name.isidentifier() or not name.isascii() or name.startswith('_') \
                or name in layers or name in ('simulation', 'input') \
                or any(path.split('/', 1)[-1].replace('/', '__') == name for path in self.entities):
            raise ValueError(f'层名 {name!r} 须为英文标识符，且不与实体或已有的层重名')
        if not isinstance(inputs, list) or not inputs:
            raise ValueError('新的一层至少读一项输入')
        existing = next(iter(layers))
        layers.insert(0, name, {'model': model, 'period': period, 'inputs': list(inputs)})
        if name not in layers[existing]['inputs']:
            layers[existing]['inputs'].append(name)
        policy['train'] = policy.get('train', existing)

    def remove_layer(self, name):
        """Merge a two-layer policy back into the other layer (scene.yaml only; policy.py keeps the code)."""
        policy = self.data.get('policy')
        if policy is None or name not in policy.get('layers', {}) or len(policy['layers']) != 2:
            raise ValueError('只能从两层的策略中去掉一层')
        layers = policy['layers']
        remaining = next(key for key in layers if key != name)
        layers[remaining]['inputs'] = [token for token in layers[remaining]['inputs']
                                       if token.partition('.')[0] != name]
        if not layers[remaining]['inputs']:
            raise ValueError(f'去掉 {name} 后 {remaining} 没有输入；先给 {remaining} 选择其他输入')
        del layers[name]
        policy.pop('train', None)

    def set_observations(self, profile, terms):
        if terms and set(terms).intersection(self.data.get('variables', {})):
            raise ValueError('Select basic information; compute model features in policy.py')
        control, layers, key = policy_section(self.data)
        if control is not None:
            role = (control.get('train') or next(iter(layers))) if profile == 'states' else profile
            if role not in layers:
                raise ValueError(f'Select the inputs of one of {list(layers)}')
            if not isinstance(terms, list) or not terms or len(set(terms)) != len(terms):
                raise ValueError('Choose a nonempty ordered list of unique observations')
            component = layers[role]
            from .layers import layer_tokens
            terms = layer_tokens(terms, layers) if key == 'inputs' else list(terms)
            component[key] = list(terms)
            if isinstance(component.get('history'), dict):
                # A deselected observation takes its window with it; kept ones keep their length.
                component['history'] = {k: v for k, v in component['history'].items() if k in terms}
                if not component['history']:
                    component.pop('history')
            from .entity_information import signal_profile
            if any(signal_profile(term).startswith('camera__') for term in terms):
                self.data.setdefault('application', {})['enable_cameras'] = True
            return
        if terms is None and profile.startswith('camera__'):
            self.data.setdefault('observations', {}).pop(profile, None)
            return
        if not isinstance(terms, list) or not terms or len(set(terms)) != len(terms):
            raise ValueError("Choose a nonempty ordered list of unique observation terms")
        observations = self.data.setdefault("observations", {})
        from .entity_information import signal_profile
        if (self.path.parent / 'task.py').is_file():
            observations[profile] = list(terms)
            if any(signal_profile(name).startswith('camera__') for name in terms):
                self.data.setdefault('application', {})['enable_cameras'] = True
            return
        if any(signal_profile(name).startswith('camera__') for name in terms):
            if len(terms) != 1 or profile != signal_profile(terms[0]):
                raise ValueError('Camera images require their independent camera observation group')
            observations[profile] = {'terms': list(terms), 'enable_corruption': False,
                                     'concatenate_terms': True, 'history_length': 0}
            self.data.setdefault('application', {})['enable_cameras'] = True
            return
        current = observations.get(profile, [])
        if isinstance(current, dict):
            current = deepcopy(current)
            current["terms"] = list(terms)
            if "overrides" in current:
                for name in list(current["overrides"]):
                    if name not in terms:
                        del current["overrides"][name]
            observations[profile] = current
        else:
            observations[profile] = list(terms)

    def set_action(self, name, settings):
        if 'control' in self.data:
            if settings is None:
                self.data['control']['output'].pop(name, None)
                return
            if not isinstance(settings, dict) or not settings.get('type') or not settings.get('asset_name'):
                raise ValueError('Basic control needs type and asset_name')
            _check_name(name)
            from .control import require_listed_joints, validate_outputs
            require_listed_joints({name: settings})
            validate_outputs({**self.data['control']['output'], name: settings})
            self.data['control']['output'][name] = deepcopy(settings)
            return
        if settings is None:
            self.data.get("actions", {}).pop(name, None)
            return
        _check_name(name)
        if not isinstance(settings, dict) or not settings.get("type") or not settings.get("asset_name"):
            raise ValueError("An action needs a type and an asset_name")
        actions = self.data.setdefault("actions", {})
        previous = actions.get(name, {})
        actions[name] = _merge(deepcopy(previous) if previous.get("type") == settings["type"] else {}, settings)

    def set_reward(self, name, settings):
        """Change an existing task reward without replacing its native parameters."""
        _check_name(name)
        if not isinstance(settings, dict) or settings.keys() - {"enabled", "weight", "inherit_weight"}:
            raise ValueError("Reward edits support enabled, weight and inherit_weight")
        if not isinstance(settings.get("enabled", True), bool):
            raise ValueError("Reward enabled must be a boolean")
        if "weight" in settings and (not isinstance(settings["weight"], (int, float)) or
                                      isinstance(settings["weight"], bool) or not isfinite(settings["weight"])):
            raise ValueError("Reward weight must be a finite number")
        from .catalog import describe_scene
        catalog = describe_scene(_plain(self.data), directory=self.path.parent)
        candidates = {item["id"] for item in catalog["rewards"] if "enabled" in item}
        if name not in candidates:
            raise ValueError(f"Unknown task reward: {name}. Add custom reward terms in YAML first.")
        task = self.data.setdefault("task", {})
        rewards = task.get("rewards", {})
        if rewards is None:
            rewards = {key: None for key in candidates}
        else:
            rewards = deepcopy(rewards)
        if not settings.get("enabled", True):
            rewards[name] = None
        else:
            current = deepcopy(rewards.get(name) or {})
            if settings.get("inherit_weight"):
                current.pop("weight", None)
            elif "weight" in settings:
                current["weight"] = settings["weight"]
            rewards[name] = current
        task["rewards"] = rewards

    def set_visualization(self, names):
        """Abstract entities drawn in the simulator viewer (``visualization.abstract``)."""
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError("Shown abstract entities must be a unique list of names")
        abstract = {node["name"] for node in self.data["entities"] if node.get("kind") == "abstract"}
        if set(names) - abstract:
            raise ValueError(f"Only abstract entities can be shown: {sorted(set(names) - abstract)}")
        if names:
            self.data.setdefault("visualization", {})["abstract"] = list(names)
        else:
            self.data.get("visualization", {}).pop("abstract", None)
            if "visualization" in self.data and not self.data["visualization"]:
                del self.data["visualization"]

    def add_entity(self, name, factory=None, *, parameters=None, pose=None, kind="entity", shape=None, size=None,
                   mount=None, meshes=None, scope=None):
        _check_name(name)
        if any(node["name"] == name for node in self.data["entities"]):
            raise ValueError(f"Entity already exists: {name}")
        if kind == "abstract":
            # Spatial content only: no factory, model or physics.
            if factory is not None or parameters:
                raise ValueError("Abstract entities have no factory or parameters")
            from .abstract import check_geometry
            node = {"name": name, "kind": "abstract", "shape": shape}
            size = list(check_geometry(name, shape, size or ()))
            if size:
                node["size"] = size
            if pose is not None:
                node["pose"] = _pose(pose)
            self.data["entities"].append(node)
            self._index()
            return self.data.get("name", "world") + "/" + name
        if shape is not None or size is not None:
            raise ValueError("shape and size belong to abstract entities")
        node = {"name": name, "factory": factory, "kind": kind}
        if scope is not None:
            if scope not in ("environment", "global"):
                raise ValueError(f"Entity {name}: scope must be environment or global")
            node["scope"] = scope
        if factory == 'camera':
            node['camera_only'] = True
        if parameters:
            node["parameters"] = deepcopy(parameters)
        if pose is not None:
            node["pose"] = _pose(pose)
        if mount:  # A sensor that only exists on a body is added already mounted.
            node["mount"] = {"entity": mount["entity"], "body": mount["body"]}
        if meshes:
            node["meshes"] = list(meshes)
        self.data["entities"].append(node)
        self._index()
        return self.data.get("name", "world") + "/" + name

    def _validate_parent_graph(self):
        from posixpath import normpath, join
        parents = {}
        physical_children = set()
        for path, node in self.entities.items():
            mount = node.get('mount')
            if mount:
                physical_children.add(path)
                ref = mount['entity']
                parents[path] = normpath(join(path, ref)) if ref.startswith('.') else ref
            elif path.rsplit('/', 1)[0] in self.entities:
                parents[path] = path.rsplit('/', 1)[0]
        for joint in self.data.get('connections', []):
            physical_children.add(joint['child']['entity'])
            if joint.get('parent'):
                parents[joint['child']['entity']] = joint['parent']['entity']
        for child, value in self.data.get('authoring', {}).get('parents', {}).items():
            if child not in self.entities or value['entity'] not in self.entities:
                raise ValueError('布局父子关系引用了不存在的实体')
            if child in physical_children:
                raise ValueError('布局子实体不能同时有物理安装')
            parents[child] = value['entity']
        for start in parents:
            current, seen = start, set()
            while current in parents:
                if current in seen:
                    raise ValueError('Cyclic 父子关系：布局或物理安装存在循环')
                seen.add(current)
                current = parents[current]

    def snapshot(self):
        self._validate_parent_graph()
        from .catalog import describe_scene
        from .visuals import load_visuals
        visuals = load_visuals(self.path, self) if self.data == self.original else {}
        catalog = describe_scene(_plain(self.data), directory=self.path.parent)
        asset_previews = {entry["id"]: entry.get("preview") for entry in catalog.get("assets", [])}
        entity_metadata = {entry["path"]: entry for entry in catalog["entities"]}
        entities = []
        for path, node in self.entities.items():
            raw, relative = self.targets[path]
            mount = node.get("mount")
            visual = visuals.get(path, {})
            display_pose = _pose(node.get("pose", visual.get("default_pose", {})))
            meta = entity_metadata.get(path, {})
            if node.get("kind") == "abstract":
                from .abstract import LABELS
                meta = {"category": "abstract", "category_label": "抽象体 · " + LABELS.get(node.get("shape"), "?"),
                        "native_key": node["name"]}
            entities.append({"path": path, "parent": path.rsplit("/", 1)[0],
                             "name": node["name"], "factory": node.get("factory"),
                             "label": self.data.get('authoring', {}).get('labels', {}).get(path, node['name']),
                             "selectable": self.data.get('authoring', {}).get('selectable', {}).get(
                                 path, meta.get("category") != "terrain"),
                             "kind": node.get("kind", "entity"), "pose": _pose(node.get("pose", {})),
                             "category": meta.get("category", "group"),
                             "category_label": meta.get("category_label", "组合"),
                             "native_key": meta.get("native_key"),
                             "initial_display": visual.get("initial_state", {}),
                             "exported": bool(visual),  # a current native record, with or without meshes
                             "initial_state": _plain(node.get("initial_state", {})),
                             "reset": _plain(node.get("reset", {})),
                             "camera_only": node.get("camera_only", False),
                             "kinematic": bool(node.get("kinematic", False)), "scope": meta.get("scope"),
                             "motion": _plain(node.get("motion")), "visible": node.get("visible", True) is not False,
                             "gravity_compensation": node.get("gravity_compensation", False) is True,
                             "display_pose": display_pose,
                             "pose_authored": "pose" in node, "parameters": node.get("parameters", {}),
                             "tags": node.get("tags", []), "mount": mount, "meshes": list(node.get("meshes", [])),
                             "shape": node.get("shape"), "size": node.get("size", []),
                             "connections": node.get("connections", []),
                             "instance_component": relative, "preview": (visual if visual.get("path") else
                                 _preview(node, self.path.parent, asset_previews.get(node.get("factory")))),
                             "transform_frame": "body-local" if mount else "parent-local"})
        return {"scene": str(self.path), "name": self.data.get("name", "world"),
                "revision": self.revision, "entities": entities, "catalog": catalog,
                "observations": _plain({role: component.get("observations", []) for role, component in self._control().get("components", {}).items()}
                                       if self._control() else self.data.get("observations", {})),
                "control": _plain(self._control()),
                "policy": _plain(self.data.get("policy")),
                "actions": _plain(self.data.get("actions", {})),
                "task": _plain(self.data.get("task", {})),
                "variables": _plain(self.data.get("variables", {})),
                "task_source": str(self.path.parent / "task.py") if (self.path.parent / "task.py").is_file() else None,
                "visualization": _plain(self.data.get("visualization", {})),
                "connections": _plain(self.data.get("connections", [])),
                "layout_groups": _plain(self.data.get("authoring", {}).get("groups", {})),
                "layout_parents": _plain(self.data.get("authoring", {}).get("parents", {})),
                "interface": self._interface()}

    def _control(self):
        """The internal control mapping (layers.control_from_policy for a ``policy:``), {} without a policy."""
        if 'policy' in self.data:
            from .layers import control_from_policy
            return control_from_policy(_plain(self.data['policy']), self.path.parent)
        return self.data.get('control', {})

    def _interface(self):
        from .interfaces import load_interface
        return load_interface(self.path) if self.data == self.original else None

    def save(self, *, expected=None):
        self.check_revision(expected)
        if self.data == self.original:
            return False
        if (self.path.parent / 'task.py').is_file():
            from .object_schema import generate
            generate(self.path, settings=_plain(self.data), write=False)
        buffer = StringIO()
        self.yaml.dump(self.data, buffer)
        content = buffer.getvalue()
        # Validate structure before replacing the source; preserve unknown keys.
        if not isinstance(self.yaml.load(content), dict):
            raise ValueError("Edited scene is not a mapping")
        descriptor, filename = tempfile.mkstemp(prefix=".scene-edit-", suffix=".yaml", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(filename, self.path.stat().st_mode)
            self.check_revision(expected)
            os.replace(filename, self.path)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
        self.files[str(self.path)] = sha256(self.path.read_bytes()).hexdigest()
        self.original = deepcopy(self.data)
        if (self.path.parent / 'task.py').is_file():
            from .object_schema import generate
            generate(self.path)
        return True


def _preview(node, directory, metadata=None):
    """Use explicitly referenced local geometry, otherwise label a schematic proxy."""
    parameters = node.get("parameters", {})
    if metadata:
        metadata = {"path": metadata} if isinstance(metadata, str) else deepcopy(metadata)
        path = metadata.get("path")
        if path and "://" not in path:
            candidate = (directory / path).expanduser().resolve()
            if candidate.is_file():
                return {**metadata, "path": str(candidate)}
    for field in ("usd_path", "mesh_path", "asset_path"):
        path = parameters.get(field)
        if isinstance(path, str) and "://" not in path:
            candidate = (directory / path).expanduser().resolve()
            if candidate.is_file() and candidate.suffix.lower() in (".usd", ".usda", ".usdc", ".obj", ".stl", ".glb", ".gltf"):
                return {"path": str(candidate), "exact": True}
    root = Path(__file__).resolve().parents[2]
    if node.get("factory") == "franka_panda":
        candidate = root / "assets/franka_panda/source/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
        if candidate.is_file():
            return {"path": str(candidate), "exact": False,
                    "note": "Local Panda visual reference; native factory joint defaults and physics remain authoritative"}
    return {"path": None, "exact": False,
            "note": "Schematic proxy only: dimensions, body frames and generated terrain are not physical previews"}


def apply_edits(path, payload):
    """Apply explicit edits from a CLI/Blender snapshot; fail on external changes."""
    allowed = {"layers", "control_train", "control_modes", "components", "revision", "entities", "visualization", "observations", "actions", "rewards", "task", "variables", "add_entities", "connections", "layout_groups", "remove_entities", "layout_parents", "labels", "selectable", "rename"}
    if not isinstance(payload, dict) or payload.keys() - allowed:
        raise ValueError("Unknown edit payload fields; expected revision, entities, visualization, observations, actions, components, rewards, add_entities, connections or layout_groups")
    document = SceneDocument(path)
    if "revision" not in payload:
        raise ValueError("Edits require the revision returned by inspect")
    document.check_revision(payload["revision"])
    for entity, label in payload.get('labels', {}).items():
        if entity not in document.entities or not isinstance(label, str) or not label.strip():
            raise ValueError('实体显示名称不能为空，且必须引用场景中的实体')
        document.data.setdefault('authoring', {}).setdefault('labels', {})[entity] = label.strip()
    # Editor-only: whether a click in the 3D view selects the entity (terrain is locked unless set).
    for entity, selectable in payload.get('selectable', {}).items():
        if entity not in document.entities or not isinstance(selectable, bool):
            raise ValueError('selectable maps scene entities to true or false')
        document.data.setdefault('authoring', {}).setdefault('selectable', {})[entity] = selectable
    if (document.path.parent / 'task.py').is_file() and set(payload) & {'variables', 'task'}:
        raise ValueError('Edit the Python task in task.py; Task formulas no longer belong in scene.yaml')
    for field in ('variables', 'task'):
        if field in payload:
            document.data[field] = deepcopy(payload[field])
    if 'variables' in payload or 'task' in payload:
        from .legacy.task import validate_evaluation, evaluation_config
        validate_evaluation(evaluation_config(document.data))
    for change in payload.get("entities", []):
        document.set_entity(change["path"], change["changes"], frame=change.get("frame", "local"))
    if 'layers' in payload:  # {'add': {name, model, period, inputs}} or {'remove': name}
        change = payload['layers']
        if 'add' in change:
            document.add_layer(**change['add'])
        if 'remove' in change:
            document.remove_layer(change['remove'])
    if 'control_train' in payload:
        control, layers, _ = policy_section(document.data)
        role = payload['control_train']
        if role not in layers:
            raise ValueError(f'Select one of the layers {list(layers)}')
        control['train'] = role
        document.data.get('world', {}).pop('decimation', None)
    if 'control_modes' in payload:
        from .control import CONTROL_MODES
        control, _, _ = policy_section(document.data)
        if control is None or 'output' in control:
            raise ValueError('control.modes applies to automatic robot control; this scene lists control.output')
        modes = control.setdefault('modes', {})
        for robot, mode in payload['control_modes'].items():
            if mode not in (*CONTROL_MODES, 'none'):
                raise ValueError(f'{robot}: control mode must be one of {", ".join((*CONTROL_MODES, "none"))}')
            if mode == 'joint_position':  # the default
                modes.pop(robot, None)
            else:
                modes[robot] = mode
        if not modes:
            del control['modes']
    for role, fields in payload.get('components', {}).items():
        _, layers, key = policy_section(document.data)
        component = layers.get(role)
        editable = {'period'} if key == 'inputs' else {'period', 'dimension'}  # a policy layer's sizes come from its model
        if component is None or not isinstance(fields, dict) or fields.keys() - editable:
            raise ValueError('组件只能修改 ' + ' 与 '.join(sorted(editable)))
        from .control import period_steps
        dt = float(document.data.get('simulation', {}).get('dt', .005))
        if 'period' in fields:
            period_steps(role, fields['period'], dt)
        dimension = fields.get('dimension', component.get('dimension', 'auto'))
        if dimension == 'auto' and role == 'planner' or dimension != 'auto' and (type(dimension) is not int or dimension <= 0):
            raise ValueError(f'{role} 输出维度须为正整数' + ('' if role == 'planner' else '或 auto'))
        component.update(fields)
        document.data.get('world', {}).pop('decimation', None)
    for profile, terms in payload.get("observations", {}).items():
        document.set_observations(profile, terms)
    for name, action in payload.get("actions", {}).items():
        document.set_action(name, action)
    for name, reward in payload.get("rewards", {}).items():
        document.set_reward(name, reward)
    for entity in payload.get("add_entities", []):
        document.add_entity(**entity)
    for name, value in payload.get("connections", {}).items():
        document.set_connection(name, value)
    for name, value in payload.get("layout_groups", {}).items():
        document.set_layout_group(name, value)
    for child, value in payload.get("layout_parents", {}).items():
        document.set_layout_parent(child, value)
    if payload.get("remove_entities"):
        document.remove_entities(payload["remove_entities"])
    if "visualization" in payload:
        value = payload["visualization"]
        if not isinstance(value, dict) or value.keys() - {"abstract"}:
            raise ValueError("visualization edits support abstract")
        document.set_visualization(value.get("abstract", []))
    # Last, as every other edit names entities by their current paths: {path: new name}.
    for entity, name in (payload.get("rename") or {}).items():
        document.rename_entity(entity, name)
    # Resolve all names/references before writing, so a bad edit leaves the
    # original YAML intact. The simulator still validates native tensor shapes.
    document.snapshot()
    from .catalog import _project_package
    from .project import project_registry, _check_factory
    from .native_terms import validate_task_settings
    from .visuals import geometry_backend
    factories, terms, tasks = project_registry(_project_package(document.path.parent),
                                               backend=geometry_backend(document.path))
    for entity_path, node in document.entities.items():
        if node.get("factory"):
            _check_factory(node["factory"], node.get("parameters", {}), factories, entity_path)
    from .assets import image_outputs
    from .entity_information import check_image_sources
    from .project import _selected_sources
    check_image_sources(_selected_sources(_plain(document.data)), {
        path.split('/', 1)[1].replace('/', '__'): image_outputs(factories[node['factory']], node.get('parameters', {}))
        for path, node in document.entities.items() if node.get('factory')})
    validate_task_settings(_plain(document.data), term_registry=terms, task_presets=tasks, require_training=False)
    changed = document.save(expected=payload["revision"])
    return {"changed": changed, "snapshot": document.snapshot()}


def simulation_settings(scene):
    """``(backend, choices, current)`` of the run-time choices of a project's train.yaml: the backend's
    ``project.SIMULATION_CHOICES`` and the values ``env.train.init_params`` names (else each default)."""
    from omegaconf import OmegaConf
    from .assets import backend_module
    from .visuals import geometry_backend
    backend = geometry_backend(scene)
    choices = backend_module(backend, 'project').SIMULATION_CHOICES
    recipe = Path(scene).resolve().parent / 'train.yaml'
    parameters = {}
    if recipe.is_file():
        parameters = OmegaConf.select(OmegaConf.load(recipe), 'env.train.init_params', default=None) or {}
    current = {name: str(parameters.get(name) or next(iter(options))) for name, options in choices.items()}
    return backend, choices, current


def write_simulation_settings(scene, values):
    """Write ``values`` (e.g. ``{'physics': 'ovphysx'}``) into train.yaml ``env.train.init_params``; evaluation
    follows training (``env.eval: ${env.train}``). Only those lines change: an existing key's value is
    replaced in place, a new key is added after the block's last key at the same indentation."""
    _, choices, _ = simulation_settings(scene)
    for name, value in values.items():
        if name not in choices or value not in choices[name]:
            raise ValueError(f'{name} must be one of {list(choices.get(name, ()))}, got {value!r}')
    recipe = Path(scene).resolve().parent / 'train.yaml'
    if not recipe.is_file():
        raise ValueError(f'{recipe} is missing; the run-time choices live in the project\'s train.yaml')
    text = recipe.read_text()
    node = YAML().load(text)
    for key in ('env', 'train', 'init_params'):
        node = node.get(key) if hasattr(node, 'get') else None
        if not hasattr(node, 'lc'):
            raise ValueError(f'{recipe}: add an env.train.init_params block (with backend) before choosing '
                             'the physics engine or renderer')
    lines = text.splitlines(keepends=True)
    for name, value in values.items():
        if name in node:
            row, column = node.lc.value(name)
            line = lines[row]
            # keep what follows the old scalar (spacing, comment, newline)
            rest = re.match(r'("[^"]*"|\'[^\']*\'|[^\s#]+)?(.*)$', line[column:], re.S).group(2)
            lines[row] = line[:column] + str(value) + rest
        else:
            last = list(node)[-1]
            row, column = node.lc.key(last)
            # after the last key and anything nested under it
            after = row + 1
            while after < len(lines) and (not lines[after].strip() or len(lines[after]) - len(lines[after].lstrip()) > column):
                after += 1
            lines.insert(after, ' ' * column + f'{name}: {value}\n')
            node[name] = value
            node.lc.data[name] = [after, column, after, column + len(name) + 2]
    temporary = recipe.with_name(f'.{recipe.name}.save')
    temporary.write_text(''.join(lines))
    os.replace(temporary, recipe)
