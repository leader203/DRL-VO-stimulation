"""Owned session door poses and a current PhysX grid; no source layer writes."""
import hashlib
import json
import math
from pathlib import Path


def open_doors(stage, report):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)
              and p.GetAttribute('interaction:role').Get() == 'door']
    rows = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for joint in joints:
            targets = joint.GetRelationship('physics:body1').GetTargets()
            if len(targets) != 1:
                raise ValueError('Door needs exactly one authored body1 hinge')
            leaf = stage.GetPrimAtPath(targets[0])
            if leaf.GetAttribute('physics:rigidBodyEnabled').Get():
                raise ValueError('Fixed-open preview requires the existing frozen door bodies')
            axis = joint.GetAttribute('physics:axis').Get()
            pivot = joint.GetAttribute('physics:localPos1').Get()
            rotation = joint.GetAttribute('physics:localRot1').Get()
            if axis != 'Z' or tuple(pivot) != (0.,0.,0.) or rotation != Gf.Quatf(1.):
                raise ValueError('Unexpected door hinge frame; retain original door')
            lower = joint.GetAttribute('physics:lowerLimit').Get()
            upper = joint.GetAttribute('physics:upperLimit').Get()
            angle = -90. if lower <= -90. <= upper else 90.
            if not lower <= angle <= upper:
                raise ValueError('Door has no admitted quarter-turn opening')
            transform = UsdGeom.Xformable(leaf)
            before = transform.GetLocalTransformation()
            turn = Gf.Matrix4d(1.).SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),angle))
            transform.MakeMatrixXform().Set(turn*before)
            rows.append(dict(path=str(leaf.GetPath()),joint=str(joint.GetPath()),
                angle_deg=angle,rigid_body_enabled=False,joint_enabled=False,
                colliders=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(leaf))))
    if len(rows) != 15:
        raise ValueError('Expected the audited 15 NewMetro door leaves')
    report['doors'] = dict(mode='all fixed open in session; physical colliders follow leaf',
                          count=len(rows),leaves=rows,source_layers_saved=False)
    return rows


def refresh_grid(native, query, output, report, pump, *, sfm_force_only=False):
    """Requery all historically known cells at the new static door snapshot."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    provider = native.provider
    original = native._labels.copy()
    grid = original.copy()
    cells = np.argwhere(original != 205)
    xy = native.pixel_to_world(cells)
    ground = provider.ground_height_m
    floors, statics = set(provider.floor_paths),set(provider.static_paths)
    counts = np.zeros(len(cells), dtype=np.int32)
    support = np.zeros(len(cells),dtype=np.bool_)
    door_hits = {}
    def is_dynamic(path):
        return path.startswith('/World/envs/') or path.startswith('/World/NavigationRobot/')
    for i, ((row,col), (x,y)) in enumerate(zip(cells,xy)):
        hits = []
        def overlap(hit):
            path = str(hit.collision)
            if is_dynamic(path):
                return True
            if path not in floors and path not in statics:
                raise ValueError('Unexpected static collider in door grid: '+path)
            if path in statics:
                hits.append(path)
                if '/Motion/Leaf_' in path:
                    leaf = path.split('/Collision/')[0]
                    door_hits[leaf]=door_hits.get(leaf,0)+1
            return True
        query.overlap_box((.05,.05,.825),(float(x),float(y),ground+.975),(0.,0.,0.,1.),overlap,False)
        counts[i]=len(hits)
        if hits:
            grid[row,col]=0
        else:
            def ray(hit):
                if str(hit.collision) in floors and hit.normal[2]>=.9 and abs(hit.position[2]-ground)<.08:
                    support[i]=True
                return True
            query.raycast_all((float(x),float(y),ground+.3),(0.,0.,-1.),.6,ray,False)
            grid[row,col]=254 if support[i] else 205
        if i%500==0:
            pump()
    # The existing conservative clearance formula is preserved exactly.
    blocked = grid != 254
    free = distance_transform_edt(np.pad(~blocked,1,constant_values=False))[1:-1,1:-1]*.1
    occupied = distance_transform_edt(np.pad(blocked,1,constant_values=True))[1:-1,1:-1]*.1
    half = math.sqrt(2)*.1/2
    native._labels=grid.copy()
    native._walkable=~blocked & (free-2*half>=native.clearance_m)
    native._connector_walkable=~blocked & (free-2*half>=native.start_recovery_clearance)
    native._signed_grid=np.where(blocked,-occupied,free-half).astype(np.float32)
    native._sdf_tensors.clear()
    native._route_segment_cache.clear()
    if not sfm_force_only:
        if not native.is_walkable(native._origins).all() or not native.is_walkable(native._routes).all():
            raise ValueError('Open doors interfere with the existing patrol or spawns')
        for actor,route in enumerate(native._routes):
            for vertex in range(len(route)):
                path=native.astar_path(route[vertex],route[(vertex+1)%len(route)])
                if path is None:
                    raise ValueError('Open-door patrol has an unreachable segment')
                native._route_segment_cache[actor,vertex]=path
    for value in (native._labels,native._walkable,native._connector_walkable,native._signed_grid):
        value.setflags(write=False)
    provider._map=grid.copy()
    path=Path(output)/'open_doors_native_grid.npz'
    np.savez_compressed(path,labels=grid,original_labels=original,pixels=cells,
                        world_xy=xy,static_overlap_count=counts,floor_supported=support)
    report['open_doors_map'] = dict(status='CURRENT_PHYSX_STATIC_DOOR_GRID_QUERIED',
        known_cells_queried=len(cells),unknown_cells_promoted=0,
        changed_cells=int((grid!=original).sum()),
        blocked_to_free_cells=int(((original==0)&(grid==254)).sum()),
        free_to_blocked_cells=int(((original==254)&(grid==0)).sum()),
        all_patrol_points_walkable='NOT_APPLICABLE_SFM_FORCE_ONLY' if sfm_force_only else True,
        all_patrol_segments_reachable='NOT_EXECUTED_SFM_FORCE_ONLY' if sfm_force_only else True,
        floor_support_tested_for_every_current_free_cell=True,
        door_leaf_cell_hit_counts=door_hits,artifact=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Owned current-pose preview grid; baseline native-map certificate remains historical')
    (Path(output)/'OPEN_DOORS.json').write_text(json.dumps(
        dict(doors=report['doors'],map=report['open_doors_map']),indent=2)+'\n')
