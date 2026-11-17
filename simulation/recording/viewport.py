# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Remove session-owned USD camera specifications after recording."""


def _remove_recording_camera_specs(stage, camera_path, authored_paths) -> None:
    from pxr import Sdf

    layers = stage.GetLayerStack()
    with Sdf.ChangeBlock():
        for layer in layers:
            existing = authored_paths.get(layer.identifier)
            if existing is None:
                # A newly attached user layer may already contain this path.
                continue
            # Kit can author camera properties in the current root edit target,
            # creating an over outside the session layer that defined the camera.
            if camera_path not in existing and layer.GetPrimAtPath(camera_path) is not None:
                _remove_prim_spec(layer, camera_path)
            for path in reversed(camera_path.GetPrefixes()[:-1]):
                spec = layer.GetPrimAtPath(path)
                if (
                    path not in existing
                    and spec is not None
                    and spec.specifier == Sdf.SpecifierOver
                    and not spec.nameChildren
                    and not spec.properties
                    and set(spec.ListInfoKeys()) <= {"specifier"}
                ):
                    _remove_prim_spec(layer, path)


def _remove_prim_spec(layer, path) -> None:
    from pxr import Sdf

    edit = Sdf.BatchNamespaceEdit()
    edit.Add(path, Sdf.Path.emptyPath)
    if not layer.Apply(edit):
        raise RuntimeError(f"Cannot remove temporary recording camera spec: {path} in {layer.identifier}")
