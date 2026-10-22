"""Check task expressions using signal semantics, without reading scene data."""
from dataclasses import replace
from ..information import Information


OPAQUE = {"rotation", "pose", "region", "image", "point_cloud", "grid", "record", "category"}


def infer_evaluation(config, sources):
    from .task import task_nodes, reward_entries
    nodes, histories, declarations = task_nodes(config)
    cache, expressions, unresolved = {}, {}, set()

    def fail(key, message):
        raise ValueError(f"Task {key}: {message}")

    def same(a, b, attribute, key):
        left, right = getattr(a, attribute), getattr(b, attribute)
        if attribute == "unit":
            left = product_unit(left, "1") or left
            right = product_unit(right, "1") or right
        if left is not None and right is not None and left != right:
            fail(key, f"incompatible {attribute}: {left!r} vs {right!r}; select matching information or explicitly convert")

    def numeric(info, key):
        if info.kind in OPAQUE or info.kind == "boolean":
            fail(key, f"{info.kind} is not an ordinary number/vector; extract a field or use its semantic operation")

    def boolean(info, key, scalar=False):
        if info.kind != "unknown" and info.kind != "boolean":
            fail(key, f"expected boolean condition, got {info.kind}")
        if scalar and info.shape not in (None, ()):
            fail(key, "expected one condition per environment; use all_elements / any_elements")

    def shape_pair(a, b, key, broadcast=True):
        if a.shape is None or b.shape is None: return None
        if broadcast and (a.shape == () or b.shape == ()): return b.shape if a.shape == () else a.shape
        if len(a.shape) != len(b.shape) or any(x is not None and y is not None and x != y for x, y in zip(a.shape, b.shape)):
            fail(key, f"incompatible shapes {a.shape} vs {b.shape}")
        return tuple(x if x is not None else y for x, y in zip(a.shape, b.shape))

    def node(name):
        if name not in cache:
            inferred = expr(nodes[name], "node." + name)
            cache[name] = declarations.get(name, inferred)
        return cache[name]

    def expr(e, key):
        result = compute(e, key)
        expressions[key] = result
        return result

    def compute(e, key):
        if isinstance(e, str): e = {'expr': e}
        if 'expr' in e:
            import ast
            from .expressions import dependencies, parse, DOMAIN_FUNCTIONS, domain_expression, domain_key
            refs, signal_paths, _, _ = dependencies(e['expr'], nodes)
            for ref in refs: node(ref)
            for path in signal_paths:
                if sources.get(path, Information()).kind == 'unknown': unresolved.add(path)
            from .expressions import path as expression_path
            tree = parse(e['expr'])
            root_info = None
            for call in ast.walk(tree):
                if isinstance(call, ast.Call) and expression_path(call.func) in DOMAIN_FUNCTIONS:
                    info = expr(domain_expression(call), domain_key(key, call))
                    if call is tree: root_info = info
            if root_info is not None: return root_info
            # A direct alias retains metadata. General Torch expressions are
            # checked at runtime unless their declaration supplies information.
            text = e['expr'].strip()
            if text in nodes: return node(text)
            if text in sources: return sources[text]
            return Information()
        if "signal" in e:
            path = e["signal"]
            info = sources.get(path, Information())
            if info.kind == "unknown": unresolved.add(path)
            return info
        if "ref" in e: return node(e["ref"])
        if "value" in e:
            def shape(v):
                if not isinstance(v, list): return ()
                sub = shape(v[0])
                if any(shape(x) != sub for x in v): fail(key, "ragged constant array")
                return (len(v), *sub)
            def is_bool(v): return all(is_bool(x) for x in v) if isinstance(v, list) else isinstance(v, bool)
            def leaves(v):
                return [y for x in v for y in leaves(x)] if isinstance(v, list) else [v]
            flat = leaves(e["value"])
            if any(isinstance(v, str) for v in flat):
                if not all(isinstance(v, str) for v in flat): fail(key, "cannot mix labels and numbers in a constant")
                return Information("category", shape(e["value"]), "int", labels=tuple(dict.fromkeys(flat)))
            return Information("boolean" if is_bool(e["value"]) else "number", shape(e["value"]),
                               "bool" if is_bool(e["value"]) else "float")
        op = e["op"]
        args = {k: expr(v, key + "." + k) for k, v in e.items() if isinstance(v, dict)}
        inputs = [expr(v, key + f".inputs.{i}") for i, v in enumerate(e.get("inputs", []))]
        x = args.get("input", Information())
        a, b = args.get("a", Information()), args.get("b", Information())
        if op == "field":
            if x.kind == "unknown": return Information()
            if e["name"] not in x.fields: fail(key, f"unknown field {e['name']!r}; available: {list(x.fields)}")
            return x.fields[e["name"]].info
        if op == "valid": return Information("boolean", (), "bool")
        if op in ("not", "hold", "ever", "within", "rising", "all_elements", "any_elements", "count"):
            boolean(x, key, scalar=op in ("hold", "within", "rising"))
            reduced = op in ("all_elements", "any_elements", "count")
            if reduced and x.shape == (): fail(key, "expected a collection of booleans")
            shape = x.shape[:-1] if reduced and x.shape is not None else x.shape
            return Information("number" if op == "count" else "boolean", shape, "int" if op == "count" else "bool",
                               "1" if op == "count" else None)
        if op in ("all", "any", "at_least", "sequence"):
            for info in inputs: boolean(info, key, scalar=True)
            return Information("boolean", (), "bool")
        if op == "rotation_distance":
            for info in (a, b):
                if info.kind not in ("unknown", "rotation"): fail(key, "rotation_distance requires declared rotations")
            same(a, b, "frame", key)
            return Information("number", (), "float", "rad")
        if op in ("rotate", "transform_point"):
            q = args["quaternion"]
            if q.kind not in ("rotation", "unknown", "number"): fail(key, "expected rotation quaternion")
            if q.shape is not None and q.shape != (4,): fail(key, "quaternion must have four components")
            numeric(x, key)
            if x.shape is not None and x.shape != (3,): fail(key, "rotation input must have three components")
            source = q.frame if e.get("inverse") else q.target_frame
            destination = q.target_frame if e.get("inverse") else q.frame
            if x.frame and source and x.frame != source: fail(key, f"rotation expects frame {source}, got {x.frame}")
            if op == "transform_point":
                p = args["position"]
                numeric(p, key)
                if p.shape is not None and p.shape != (3,): fail(key, "translation must have three components")
                same(x, p, "unit", key)
                if destination and p.frame and destination != p.frame: fail(key, "translation and rotated point have different frames")
                return Information("position", (3,), "float", p.unit or x.unit, p.frame or destination)
            return replace(x, frame=destination, kind="vector" if x.kind == "position" else x.kind)
        if op == "inside_region":
            region = args["region"]
            if x.kind not in ("unknown", "position", "number"): fail(key, "region containment expects a position")
            if region.kind not in ("unknown", "region"): fail(key, "expected a region")
            same(x, region, "frame", key)
            same(x, region, "unit", key)
            if x.shape is not None and x.shape != (3,): fail(key, "region point must have three components")
            return Information("boolean", (), "bool")
        if op == "select":
            if x.shape == (): fail(key, "cannot select a component of a scalar")
            indices = e["indices"]
            selected = (indices,) if isinstance(indices, int) else tuple(indices)
            for part in x.fields.values():
                if selected == part.indices: return part.info
            if x.kind in OPAQUE: fail(key, f"cannot split {x.kind} arbitrarily; use field")
            if x.shape and x.shape[-1] is not None and any(not -x.shape[-1] <= i < x.shape[-1] for i in selected):
                fail(key, "component index out of range")
            shape = None if x.shape is None else (x.shape[:-1] if isinstance(indices, int) else (*x.shape[:-1], len(indices)))
            return Information("number", shape, x.dtype, x.unit, x.frame)
        if op in ("equal", "not_equal", "in_set"):
            if op == "in_set":
                if x.kind in OPAQUE - {"category"}: fail(key, "membership expects numeric/boolean/category values")
                for choice in e["choices"]:
                    if isinstance(choice, str) and choice not in x.labels:
                        fail(key, f"unknown category {choice!r}; expected {x.labels}")
                return Information("boolean", x.shape, "bool")
            if a.kind == "category" or b.kind == "category":
                for side, other in (("a", b), ("b", a)):
                    if "value" in e[side] and args[side].kind == "category" and other.kind == "category":
                        labels = other.labels
                        if "value" in e["a"] and "value" in e["b"]:
                            labels = tuple(dict.fromkeys((*a.labels, *b.labels)))
                        if any(label not in labels for label in args[side].labels): fail(key, f"unknown category label; expected {labels}")
                        expressions[key + "." + side] = replace(args[side], labels=labels)
                if "value" not in e["a"] and "value" not in e["b"] and a.labels != b.labels:
                    fail(key, "category codebooks differ")
            same(a, b, "unit", key)
            same(a, b, "frame", key)
            if a.kind in OPAQUE - {"category"} or b.kind in OPAQUE - {"category"}: fail(key, "use semantic comparison for structured information")
            return Information("boolean", shape_pair(a, b, key), "bool")
        if op in ("between", "inside_box"):
            numeric(x, key)
            for bound in (args["lower"], args["upper"]):
                numeric(bound, key)
                same(x, bound, "unit", key)
                same(x, bound, "frame", key)
                shape_pair(x, bound, key)
            if op == "inside_box" and x.shape == (): fail(key, "inside_box expects vector coordinates")
            return Information("boolean", x.shape[:-1] if op == "inside_box" and x.shape is not None else x.shape, "bool")
        if "a" in args:
            numeric(a, key)
            numeric(b, key)
            geometry = op in ("distance", "angle", "dot")
            shape = shape_pair(a, b, key, broadcast=not geometry)
            same(a, b, "frame", key)
            if op not in ("multiply", "divide", "dot", "angle"): same(a, b, "unit", key)
            if op in ("less", "less_equal", "greater", "greater_equal", "near"):
                return Information("boolean", shape, "bool")
            if geometry:
                if shape == (): fail(key, f"{op} requires a vector; scalar differences use subtract/abs")
                if (a.kind == "position" or b.kind == "position") and op != "distance":
                    fail(key, "positions are points; subtract positions first to obtain a direction vector")
                if op == "distance" and {a.kind, b.kind} == {"position", "vector"}:
                    fail(key, "cannot measure distance between an absolute position and a displacement")
                return Information("number", shape[:-1] if shape is not None else None, "float",
                                   "rad" if op == "angle" else (product_unit(a.unit, b.unit) if op == "dot" else a.unit or b.unit))
            if op == "add" and a.kind == b.kind == "position": fail(key, "adding two positions is ambiguous; add a displacement vector")
            if op == "subtract" and a.kind == "vector" and b.kind == "position": fail(key, "cannot subtract an absolute position from a displacement")
            kind = "vector" if op == "subtract" and a.kind == b.kind == "position" else (a.kind if a.kind != "unknown" else b.kind)
            if op == "add" and "position" in (a.kind, b.kind): kind = "position"
            if op in ("multiply", "divide"):
                if "position" in (a.kind, b.kind): fail(key, "scale a displacement, not an absolute position")
                au = "1" if "value" in e["a"] and a.unit is None else a.unit
                bu = "1" if "value" in e["b"] and b.unit is None else b.unit
                unit = product_unit(au, bu, divide=op == "divide")
            else: unit = a.unit or b.unit
            return Information(kind, shape, "float", unit, a.frame or b.frame)
        if op == "changed":
            if x.kind in OPAQUE - {"category"}: fail(key, "structured change needs a semantic error measurement first")
            return Information("boolean", (), "bool")
        numeric(x, key)
        if op in ("norm", "sum", "mean", "min", "max"):
            if x.kind == "position": fail(key, "use distance to a reference point, not a reduction of an absolute position")
            if x.shape == (): fail(key, f"{op} requires a feature/collection axis")
            return Information("number", x.shape[:-1] if x.shape is not None else None, "float", x.unit)
        if op == "integral": return replace(x, unit=product_unit(x.unit, "s"))
        if op == "delta" and x.kind == "position": return replace(x, kind="vector")
        if x.kind == "position" and op in ("abs", "accumulate"):
            fail(key, "use a displacement or distance measurement instead of accumulating absolute positions")
        return x

    for name in nodes: node(name)
    for name in config.get("conditions", {}): boolean(cache[name], name)
    for key in ("success", "failure"):
        if key in config: boolean(expr(config[key], key), key, scalar=True)
    for name, item in reward_entries(config).items():
        info = expr(item["input"], "rewards." + name)
        if info.shape not in (None, ()): fail(name, "reward must be scalar per environment")
        if info.kind not in ("unknown", "number", "boolean"): fail(name, "reward must be numeric or boolean")
    for name, item in config.get("metrics", {}).items(): expr(item, "metrics." + name)
    return {"nodes": cache, "expressions": expressions, "untyped_signals": sorted(unresolved),
            "untyped_variables": sorted(name for name, info in cache.items() if info.kind == 'unknown')}


def product_unit(a, b, divide=False):
    """Small symbolic SI-unit algebra, without automatic numeric conversion."""
    import re
    if a is None or b is None:
        return None
    def powers(unit):
        result, sign = {}, 1
        for token in re.split(r"([*/])", unit):
            if token in ("*", "/"):
                sign = -1 if token == "/" else 1
            elif token and token != "1":
                match = re.fullmatch(r"([A-Za-z]+)(?:\^(-?\d+))?", token)
                if not match: return None
                name, power = match.group(1), int(match.group(2) or 1) * sign
                result[name] = result.get(name, 0) + power
        return result
    left, right = powers(a), powers(b)
    if left is None or right is None: return None
    for name, power in right.items(): left[name] = left.get(name, 0) + power * (-1 if divide else 1)
    parts = [name if power == 1 else f"{name}^{power}" for name, power in sorted(left.items()) if power]
    return "*".join(parts) or "1"
