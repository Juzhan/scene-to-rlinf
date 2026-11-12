"""Isaac Lab 3 USD spawning with colliders on the geometry Newton and OvPhysX read, and per-face
materials Newton's renderers draw.

Spawner configs name it (``func = SPAWN_FROM_USD``) in place of Isaac Lab 3's own USD spawner."""

NATIVE = 'isaaclab.sim.spawners.from_files.from_files:spawn_from_usd'
SPAWN_FROM_USD = f'{__name__}:spawn_from_usd'


def gprim_colliders(stage, root_path):
    """Move collision APIs authored on Xforms below ``root_path`` onto the geometry under them.

    Isaac Sim's URDF importer applies CollisionAPI/MeshCollisionAPI to the Xform that holds each
    collision mesh (piper, piper_x, M20). Kit PhysX collides that Xform's geometry, but UsdPhysics
    defines colliders on Gprims only: Newton and OvPhysX skip them and such robots get no collision.
    Instanceable ancestors of those Xforms are made editable on the spawned prim."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root = Sdf.Path(root_path)
    found = [prim.GetPath() for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
             if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim)]
    for path in found:
        for prefix in path.GetPrefixes():
            prim = stage.GetPrimAtPath(prefix)
            if prefix.HasPrefix(root) and prim.IsInstanceable():
                prim.SetInstanceable(False)
    for path in found:
        prim = stage.GetPrimAtPath(path)
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        approximation = (UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
                         if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None)
        geometry = [child for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Gprim)]
        if not geometry or any(child.IsInstanceProxy() or child.HasAPI(UsdPhysics.CollisionAPI) for child in geometry):
            raise ValueError(f'{path}: collision Xform must hold editable geometry without its own collider')
        for child in geometry:
            collider = UsdPhysics.CollisionAPI.Apply(child)
            if enabled.HasAuthoredValue():
                collider.CreateCollisionEnabledAttr(enabled.Get())
            if approximation is not None and child.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(approximation.Get())
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)


_PHYSICS_NAMESPACES = ('physics:', 'physx', 'newton:', 'mjc:')


def _copy_properties(source, target):
    """Authored non-physics attributes (with primvar metadata) and material bindings of ``source``."""
    from pxr import UsdShade
    for attribute in source.GetAuthoredAttributes():
        name = attribute.GetName()
        if name.startswith(_PHYSICS_NAMESPACES) or not attribute.HasAuthoredValue():
            continue
        copy = target.CreateAttribute(name, attribute.GetTypeName(), attribute.IsCustom())
        copy.Set(attribute.Get())
        for key in ('interpolation', 'elementSize'):
            if attribute.HasMetadata(key):
                copy.SetMetadata(key, attribute.GetMetadata(key))
    bindings = [relation for relation in source.GetRelationships()
                if relation.GetName().startswith('material:binding') and relation.GetTargets()]
    if bindings:
        UsdShade.MaterialBindingAPI.Apply(target)
    for relation in bindings:
        copy = target.CreateRelationship(relation.GetName(), relation.IsCustom())
        copy.SetTargets(relation.GetTargets())
        if relation.HasMetadata('bindMaterialAs'):
            copy.SetMetadata('bindMaterialAs', relation.GetMetadata('bindMaterialAs'))


def material_subset_visuals(stage, root_path):
    """Give drawn collider meshes below ``root_path`` with per-face materials a Newton-only visual copy.

    SimReady parts are one mesh that is both collider and visual, with materials bound to
    GeomSubsets. Newton splits material subsets only on visual-only meshes and draws a visible
    collider with a single material. Each such collider gets ``render`` purpose (still a collider,
    still in RTX final renders, hidden from Newton's viewport semantics) and a physics-free sibling
    copy with ``proxy`` purpose (drawn by Newton, which splits its subsets; not in RTX final
    renders). Only the in-memory stage changes; instanceable ancestors are made editable."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    def drawn(prim):
        image = UsdGeom.Imageable(prim)
        return (image.ComputeVisibility() != UsdGeom.Tokens.invisible
                and image.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy))

    def bound(prim):
        return any(relation.GetName().startswith('material:binding') and relation.GetTargets()
                   for relation in prim.GetRelationships())

    root = Sdf.Path(root_path)
    found = [prim.GetPath() for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
             if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI) and drawn(prim)
             and any(bound(subset.GetPrim()) for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)))]
    for path in found:
        for prefix in path.GetPrefixes():
            prim = stage.GetPrimAtPath(prefix)
            if prefix.HasPrefix(root) and prim.IsInstanceable():
                prim.SetInstanceable(False)
    for path in found:
        prim = stage.GetPrimAtPath(path)
        name, index = f'{path.name}_visual', 1
        while stage.GetPrimAtPath(path.GetParentPath().AppendChild(name)):
            name, index = f'{path.name}_visual{index}', index + 1
        visual = stage.DefinePrim(path.GetParentPath().AppendChild(name), 'Mesh')
        _copy_properties(prim, visual)
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
            _copy_properties(subset.GetPrim(), stage.DefinePrim(visual.GetPath().AppendChild(subset.GetPrim().GetName()),
                                                                'GeomSubset'))
        UsdGeom.Imageable(visual).CreatePurposeAttr(UsdGeom.Tokens.proxy)
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.render)


def _spawn_from_usd(prim_path, cfg, *args, **kwargs):
    from isaaclab.sim.spawners.from_files import from_files
    prim = from_files.spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
    gprim_colliders(prim.GetStage(), prim_path)
    material_subset_visuals(prim.GetStage(), prim_path)
    return prim


def spawn_from_usd(prim_path, cfg, *args, **kwargs):
    """Isaac Lab 3's ``spawn_from_usd`` with :func:`gprim_colliders` applied to env_0 before cloning."""
    from isaaclab.sim.utils import clone
    return clone(_spawn_from_usd)(prim_path, cfg, *args, **kwargs)
