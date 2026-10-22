"""Semantic information metadata; shapes exclude the simulation batch axis.

Unknown metadata is explicit. Three numbers alone never imply a position.
This module imports neither a simulator nor a numerical framework.
"""
from dataclasses import asdict, dataclass, field


KINDS = {
    "unknown": "未声明", "number": "数值", "boolean": "真假", "category": "类别",
    "position": "位置", "vector": "空间向量", "rotation": "旋转", "pose": "位姿",
    "region": "区域", "image": "图像", "point_cloud": "点云", "grid": "空间网格",
    "record": "组合信息",
}


@dataclass(frozen=True)
class Information:
    kind: str = "unknown"
    shape: tuple | None = None
    dtype: str | None = None
    unit: str | None = None
    frame: str | None = None
    target_frame: str | None = None
    encoding: str | None = None
    axes: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    fields: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown information kind: {self.kind}")
        if self.shape is not None and any(x is not None and (type(x) is not int or x < 1) for x in self.shape):
            raise ValueError("Information shape dimensions must be positive or unknown (None)")
        if self.axes and self.shape is not None and len(self.axes) != len(self.shape):
            raise ValueError("Information axes must match shape rank")
        if self.kind == "rotation" and self.encoding not in ("wxyz", "xyzw"):
            raise ValueError("Rotation metadata must specify wxyz or xyzw encoding")

    def describe(self):
        return {**asdict(self), "label": KINDS[self.kind], "operations": operations_for(self)}


@dataclass(frozen=True)
class Field:
    indices: tuple[int, ...]
    info: Information


def pose_information(frame, target_frame=None, encoding="wxyz"):
    return Information("pose", (7,), "float", frame=frame, target_frame=target_frame,
                       encoding="xyz+" + encoding, fields={
        "position": Field((0, 1, 2), Information("position", (3,), "float", "m", frame)),
        "orientation": Field((3, 4, 5, 6), Information("rotation", (4,), "float", "1", frame,
                                                     target_frame, encoding)),
    })


def operations_for(info):
    """UI suggestions; operand-pair checking remains the task compiler's job."""
    common = ["valid"]
    if info.kind == "unknown": return common
    if info.fields: common += ["field"]
    if info.kind == "boolean":
        if info.shape == ():
            return common + ["not", "all", "any", "at_least", "hold", "ever", "sequence", "within", "rising", "changed"]
        return common + ["not", "ever", "all_elements", "any_elements", "count", "changed"]
    if info.kind == "category": return common + ["equal", "not_equal", "in_set", "changed"]
    if info.kind == "position": return common + ["distance", "subtract", "inside_box", "inside_region", "transform_point"]
    if info.kind == "rotation": return common + ["rotation_distance", "rotate", "transform_point"]
    if info.kind == "pose": return common
    if info.kind == "region": return common + ["inside_region"]
    if info.kind in ("image", "point_cloud", "grid", "record"): return common
    numeric = common + ["add", "subtract", "multiply", "divide", "abs", "less", "less_equal", "greater", "greater_equal",
                     "equal", "not_equal", "near", "between", "changed", "delta", "accumulate", "integral",
                     "running_mean", "running_min", "running_max"]
    if info.shape != ():
        numeric += ["select", "sum", "mean", "min", "max", "norm", "distance", "dot", "angle"]
    return numeric
