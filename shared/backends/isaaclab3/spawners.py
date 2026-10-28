"""Isaac Lab 3 USD spawning with colliders on the geometry Newton and OvPhysX read.

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


def _spawn_from_usd(prim_path, cfg, *args, **kwargs):
    from isaaclab.sim.spawners.from_files import from_files
    prim = from_files.spawn_from_usd.__wrapped__(prim_path, cfg, *args, **kwargs)
    gprim_colliders(prim.GetStage(), prim_path)
    return prim


def spawn_from_usd(prim_path, cfg, *args, **kwargs):
    """Isaac Lab 3's ``spawn_from_usd`` with :func:`gprim_colliders` applied to env_0 before cloning."""
    from isaaclab.sim.utils import clone
    return clone(_spawn_from_usd)(prim_path, cfg, *args, **kwargs)
