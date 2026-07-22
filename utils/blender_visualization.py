"""
Convert a co-layout 2D grid optimization result into a metric 3D scene and
render it in Blender.

This is the single entry point for the whole 3D visualization pipeline. The
file plays two roles depending on how it is executed:

  1. Driver (plain Python): reads a session's memory.json and
     optimization/result.json (see output/sessions/<session_id>/), retrieves
     a matching Imaginarium asset for every furniture item (see
     asset_library/asset_retriever.py), and writes a metric "layout JSON"
     (grid cells -> meters, wall/door/window openings, per-object asset +
     transform). It then either prints the Blender command to run, or (with
     --auto-render) invokes Blender itself as a subprocess -- re-running this
     same file inside Blender for step 2.

  2. Builder (running inside Blender, via
     `blender -b -P utils/blender_visualization.py -- --layout <path> ...`):
     reads the layout JSON written by step 1 and uses Blender's Python API
     (bpy/bmesh) to build the scene -- a procedural floor, walls with
     door/window openings (door leaf + window glass), imported FBX furniture,
     lighting, and camera framing -- then optionally renders a still image
     and/or saves a .blend file.

Usage:
    # Export + render in one command:
    python utils/blender_visualization.py --session <session_id> --auto-render

    # Or just export the layout JSON and print the Blender command to run:
    python utils/blender_visualization.py --session <session_id>

Prerequisites (once):
    python -m asset_library.download_imaginarium
    python -m asset_library.asset_retriever
"""

import argparse
import glob
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import bmesh
    import bpy

    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

ROOT_PATH = Path(__file__).parent.parent.resolve()
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from constants import (  # noqa: E402
    DOOR_SIZE,
    EAST,
    F,
    NORTH,
    SOUTH,
    WEST,
    WINDOW_SIZE,
    get_optimization_dir,
    get_session_dir,
    get_visualization_dir,
)
from asset_library.paths import ASSET_INDEX_DEFAULT_PATH  # noqa: E402

_FACING_LETTER = {EAST: "E", WEST: "W", SOUTH: "S", NORTH: "N"}

# (sigma, mu) -> facing compass direction -> yaw (degrees), assuming the
# imported asset's canonical/un-rotated pose faces -Y (Imaginarium's
# convention). See optimization/coopt_model.py: "(sigma, mu) = (0,0)->East
# (0,1)->West (1,0)->South (1,1)->North".
_ORIENTATION_TO_YAW = {
    (0, 0): 90.0,  # East
    (0, 1): 270.0,  # West
    (1, 0): 180.0,  # South
    (1, 1): 0.0,  # North
}

DEFAULT_FLOOR_TEXTURE = "assets/floor_texture.jpg"
# Exterior (building-envelope) walls only ever have a room on one side, so
# they are pushed fully to the outside of the cell boundary (see
# _wall_edge_geometry) and can afford to be thicker. Interior partitions sit
# between two occupied rooms with no "outside" to push into, so they stay
# centered on the boundary and are kept thinner to minimize how much they eat
# into either room's (already packed, zero-wall-thickness) furniture layout.
EXTERIOR_WALL_THICKNESS = 0.2
INTERIOR_WALL_THICKNESS = 0.1
DOOR_HEIGHT = 2.1
WINDOW_SILL_HEIGHT = 0.9
WINDOW_HEIGHT = 1.2


# ============================================================================
# STEP 1 (driver / plain python): grid optimization result -> metric layout JSON
# ============================================================================


def _clean_query_text(text: str) -> str:
    """Turn e.g. 'nightstand1' into 'nightstand' for a cleaner semantic query."""
    text = text.replace("_", " ").replace("-", " ")
    text = "".join(ch for ch in text if not ch.isdigit())
    return " ".join(text.split()).strip() or text


def _door_window_span_cells(direction: int, facing: int, i: int, j: int, size: int):
    """Cells (i, j) covered by a door/window opening, matching co-layout's own
    utils/floorplan_visualization_cadstyle.py drawing logic: for an EAST/WEST
    facing opening the span runs along i (fixed j), for a SOUTH/NORTH facing
    opening it runs along j (fixed i); ``direction`` picks which way it extends.
    """
    if facing in (EAST, WEST):
        if direction == SOUTH:
            return [(i + s, j) for s in range(size)]
        return [(i - s, j) for s in range(size)]
    if direction == EAST:
        return [(i, j + s) for s in range(size)]
    return [(i, j - s) for s in range(size)]


_FACING_DELTA = {EAST: (0, 1), WEST: (0, -1), SOUTH: (1, 0), NORTH: (-1, 0)}


def _is_exterior_edge(i: int, j: int, facing: int, outdoor_array: list, width: int, length: int) -> bool:
    """True if the cell across this edge is outside the building envelope
    (off-grid, or itself an outdoor/obstacle cell) rather than another
    occupied room -- i.e. this edge is on the building's outer perimeter, not
    an interior partition. This mirrors utils/post_process.py's add_wall():
    its first loop (exterior walls) fires exactly when the neighbor is in
    ``outdoor_space_coordinates``; its second loop (interior partitions
    between rooms) fires when the neighbor belongs to a different room, which
    is never an obstacle cell.
    """
    di, dj = _FACING_DELTA[facing]
    ni, nj = i + di, j + dj
    if ni < 0 or ni >= width or nj < 0 or nj >= length:
        return True
    return outdoor_array[ni][nj] > 0.5


def _extract_wall_edges(result_data: dict) -> list:
    """Flatten wall_array + door_array + window_array into a list of
    {"i", "j", "facing", "type", "exterior"} cell-edge records for the
    Blender builder to extrude. Door/window spans override the plain "wall"
    type of the cells they cover.
    """
    wall_array = result_data["wall_array"]
    outdoor_array = result_data["outdoor_array"]
    width = len(wall_array[EAST])
    length = len(wall_array[EAST][0]) if width else 0

    edge_type = {}
    for facing in (EAST, WEST, SOUTH, NORTH):
        plane = wall_array[facing]
        for i in range(width):
            for j in range(length):
                if plane[i][j] > 0.5:
                    edge_type[(i, j, facing)] = "wall"

    door_size = round(DOOR_SIZE / F)
    for door in result_data.get("door_array", []):
        span = _door_window_span_cells(door["direction"], door["facing"], door["start_i"], door["start_j"], door_size)
        for i, j in span:
            edge_type[(i, j, door["facing"])] = "door"

    window_size = round(WINDOW_SIZE / F)
    for window in result_data.get("window_array", []):
        span = _door_window_span_cells(window["direction"], window["facing"], window["start_i"], window["start_j"], window_size)
        for i, j in span:
            edge_type[(i, j, window["facing"])] = "window"

    return [
        {
            "i": i,
            "j": j,
            "facing": _FACING_LETTER[facing],
            "type": edge_kind,
            "exterior": _is_exterior_edge(i, j, facing, outdoor_array, width, length),
        }
        for (i, j, facing), edge_kind in edge_type.items()
    ]


def _retrieve_best_asset(retriever, name: str, length_m: float, width_m: float, height_m: float, size_tolerance: float, min_score: float):
    """Try both [length,width,height] and [width,length,height] size orderings
    against the asset index, return whichever scores higher.

    Returns (best_asset_result, size_used_for_blender_dimensions, swapped).
    """
    description = _clean_query_text(name)
    candidates = [([length_m, width_m, height_m], False)]
    if abs(width_m - length_m) > 1e-6:
        candidates.append(([width_m, length_m, height_m], True))

    best_result = None
    best_size = candidates[0][0]
    best_swapped = False
    best_score = float("-inf")
    for size_constraint, swapped in candidates:
        results = retriever.retrieve(
            description=description,
            size_constraint=size_constraint,
            size_tolerance=size_tolerance,
            top_k=1,
            min_score=min_score,
        )
        if not results:
            continue
        result = results[0]
        if result["score"] > best_score:
            best_score = result["score"]
            best_result = result
            best_size = size_constraint
            best_swapped = swapped

    if best_result is None:
        raise RuntimeError(f"No asset could be retrieved for '{name}'")

    return best_result, best_size, best_swapped


def export_layout(
    session_id: str,
    result_path: Path = None,
    asset_index_path: Path = ASSET_INDEX_DEFAULT_PATH,
    embedding_model: str = None,
    size_tolerance: float = 0.5,
    min_score: float = 0.2,
    output_path: Path = None,
) -> Path:
    from asset_library.asset_retriever import DEFAULT_EMBEDDING_MODEL, AssetRetriever
    from utils.pre_process import extract_data

    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    memory_path = get_session_dir(session_id) / "memory.json"
    if not memory_path.exists():
        raise FileNotFoundError(f"memory.json not found for session '{session_id}': {memory_path}")
    with open(memory_path, "r", encoding="utf-8") as f:
        memory_data = json.load(f)
    building_params, rooms, furniture, _, _ = extract_data(memory_data)
    wall_height = float(building_params.get("floor_height") or 2.8)
    room_name_list = list(rooms.keys())

    result_path = Path(result_path) if result_path else get_optimization_dir(session_id) / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Optimization result not found for session '{session_id}': {result_path}. "
            "Run run_optimization.py --session <session_id> first, or pass --result explicitly."
        )
    print(f"[layout_export] Using optimization result: {result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    outdoor_array = result_data["outdoor_array"]
    grid_width = len(outdoor_array)
    grid_length = len(outdoor_array[0]) if grid_width else 0
    occupied_cells = [[i, j] for i in range(grid_width) for j in range(grid_length) if outdoor_array[i][j] < 0.5]

    wall_edges = _extract_wall_edges(result_data)

    print(f"[layout_export] Loading asset retriever (index: {asset_index_path}) ...")
    retriever = AssetRetriever(index_path=str(asset_index_path), embedding_model=embedding_model)

    sigma_array = result_data["furniture_orientation_sigma_array"]
    mu_array = result_data["furniture_orientation_mu_array"]
    furniture_info = result_data["furniture_info"]

    objects = []
    obj_id = 0
    for k, room_name in enumerate(room_name_list):
        if k >= len(furniture_info):
            continue
        room_furniture = furniture.get(room_name, [])
        for l, info in enumerate(furniture_info[k]):
            name = info["name"]
            sigma = 1 if sigma_array[k][l] > 0.5 else 0
            mu = 1 if mu_array[k][l] > 0.5 else 0

            if l < len(room_furniture) and room_furniture[l]["name"] == name:
                width_m = room_furniture[l]["width"]
                length_m = room_furniture[l]["length"]
                height_m = room_furniture[l]["height"] or info["height"]
            else:
                # Fallback: memory.json furniture ordering didn't line up with the
                # optimization result (shouldn't normally happen). Approximate the
                # original (unrotated) footprint from the world-space rectangle
                # instead. sigma=1 -> world x(j)=length, world y(i)=width;
                # sigma=0 -> world x(j)=width, world y(i)=length (see
                # optimization/coopt_model.py's sigma/parallel_size/vertical_size
                # convention, mirrored in utils/floorplan_visualization_cadstyle.py).
                print(
                    f"[layout_export] WARNING: furniture name mismatch at room='{room_name}' "
                    f"index={l}: expected '{room_furniture[l]['name'] if l < len(room_furniture) else '<missing>'}', "
                    f"got '{name}' from the optimization result. Falling back to the result's "
                    f"own (post wall-snap) footprint for retrieval size matching."
                )
                if sigma:
                    width_m, length_m = info["i_size"] * F, info["j_size"] * F
                else:
                    width_m, length_m = info["j_size"] * F, info["i_size"] * F
                height_m = info["height"]

            base_yaw = _ORIENTATION_TO_YAW[(sigma, mu)]
            best_result, size_used, swapped = _retrieve_best_asset(
                retriever,
                name=name,
                length_m=length_m,
                width_m=width_m,
                height_m=height_m,
                size_tolerance=size_tolerance,
                min_score=min_score,
            )
            # Swapping which target dimension maps to the asset's local x/y axis
            # (done when it fits the retrieved asset's real proportions better)
            # must be compensated with an extra quarter turn to keep the
            # resulting world-space footprint correct.
            final_yaw = (base_yaw + 90.0) % 360.0 if swapped else base_yaw

            center = [info["j_center"] * F, info["i_center"] * F, height_m / 2.0]

            objects.append(
                {
                    "id": obj_id,
                    "name": name,
                    "room_name": room_name,
                    "size": size_used,
                    "center": center,
                    "rotation": [0.0, 0.0, final_yaw],
                    "jid": best_result["jid"],
                    "asset_short_desc": best_result.get("short_desc", ""),
                    "asset_score": round(float(best_result["score"]), 4),
                    "swapped_width_length": swapped,
                }
            )
            obj_id += 1

    layout = {
        "session_id": session_id,
        "grid_scale": F,
        "grid_width": grid_width,
        "grid_length": grid_length,
        "wall_height": wall_height,
        "occupied_cells": occupied_cells,
        "wall_edges": wall_edges,
        "objects": objects,
    }

    output_path = Path(output_path) if output_path else get_visualization_dir(session_id) / "layout_3d.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    print(f"[layout_export] Exported {len(objects)} objects and {len(wall_edges)} wall/door/window segments to {output_path}")
    return output_path


# ============================================================================
# STEP 2 (builder / inside Blender): layout JSON -> Blender scene
# ============================================================================


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # bpy.data.collections never includes the scene's own root collection, so
    # this safely drops every room/architecture/lighting collection from a
    # previous build without touching it.
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)


# ---- Collection hierarchy (Architecture / one per room / Lighting_Camera) ----


def _get_or_create_collection(name: str, parent) -> "bpy.types.Collection":
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if parent.children.get(collection.name) is None:
        parent.children.link(collection)
    return collection


def _find_layer_collection(layer_collection, target_collection):
    if layer_collection.collection == target_collection:
        return layer_collection
    for child in layer_collection.children:
        found = _find_layer_collection(child, target_collection)
        if found is not None:
            return found
    return None


def _set_active_collection(collection):
    """Make ``collection`` active so that subsequent bpy.ops-based creation
    (primitives, FBX import, lights, ...) lands inside it instead of the
    scene's root collection.
    """
    layer_collection = _find_layer_collection(bpy.context.view_layer.layer_collection, collection)
    if layer_collection is not None:
        bpy.context.view_layer.active_layer_collection = layer_collection


def render_image_file(output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image: {output_path}")
    return output_path


def save_blend_file(output_path, pack_assets: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pack_assets:
        try:
            bpy.ops.file.pack_all()
        except RuntimeError as exc:
            print(f"WARNING: failed to pack some external textures, saving .blend anyway: {exc}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    return output_path


# ---- Floor ----


def _floor_material(texture_path, texture_scale: float):
    """Image-textured floor material. ``texture_path`` defaults to
    DEFAULT_FLOOR_TEXTURE (assets/floor_texture.jpg, bundled in the repo); if
    that (or a caller-provided override) can't be found, fall back to a
    flat solid color rather than a procedural pattern.
    """
    mat = bpy.data.materials.new(name="Floor_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    resolved_texture_path = None
    if texture_path:
        candidate = Path(texture_path)
        candidate = candidate if candidate.is_absolute() else ROOT_PATH / texture_path
        if candidate.exists():
            resolved_texture_path = candidate
        else:
            print(f"WARNING: floor texture not found: {candidate}")

    if resolved_texture_path is None:
        print("WARNING: no floor texture available; using a flat fallback color")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.72, 0.60, 0.42, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.6
        return mat

    tex_image = nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(str(resolved_texture_path))
    tex_image.image.colorspace_settings.name = "sRGB"
    tex_image.location = (-500, 300)
    tex_image.extension = "REPEAT"

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 300)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 300)

    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.55
    print(f"Floor texture loaded: {resolved_texture_path}")
    return mat


def create_floor_from_grid(occupied_cells: list, grid_scale: float = 1.0, texture_path=None, texture_scale: float = 0.5, height: float = 0.0):
    """Build the floor as one quad per occupied (i, j) grid cell.

    co-layout has no boundary polygon, so instead of tracing a contour we
    simply tile a ``grid_scale`` x ``grid_scale`` quad over every occupied
    cell. This trivially handles concave / multi-room floor plans at the cost
    of a few extra coincident vertices at shared cell edges (harmless for a
    flat floor).
    """
    mesh = bpy.data.meshes.new("Floor_Mesh")
    obj = bpy.data.objects.new("Floor", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    half = grid_scale / 2.0

    for i, j in occupied_cells:
        cx, cy = j * grid_scale, i * grid_scale
        verts = [
            bm.verts.new((cx - half, cy - half, height)),
            bm.verts.new((cx + half, cy - half, height)),
            bm.verts.new((cx + half, cy + half, height)),
            bm.verts.new((cx - half, cy + half, height)),
        ]
        face = bm.faces.new(verts)
        for loop in face.loops:
            x, y, _ = loop.vert.co
            loop[uv_layer].uv = (x * texture_scale, y * texture_scale)

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    obj.data.materials.append(_floor_material(texture_path, texture_scale))
    return obj


# ---- Walls, doors, windows ----


def _wall_material():
    mat = bpy.data.materials.new(name="Wall_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.92, 0.90, 0.86, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.85
    return mat


def _door_material():
    mat = bpy.data.materials.new(name="Door_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.32, 0.19, 0.10, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
    return mat


def _glass_material():
    mat = bpy.data.materials.new(name="Window_Glass_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.75, 0.88, 0.92, 1.0)
        transmission_input = bsdf.inputs.get("Transmission Weight") or bsdf.inputs.get("Transmission")
        if transmission_input:
            transmission_input.default_value = 0.95
        bsdf.inputs["Roughness"].default_value = 0.05
        ior_input = bsdf.inputs.get("IOR")
        if ior_input:
            ior_input.default_value = 1.45
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    return mat


def _wall_edge_geometry(i: int, j: int, facing: str, grid_scale: float, thickness: float, exterior: bool):
    """Center (x, y) and long-axis ('x' or 'y') of the wall segment at grid
    cell-edge (i, j, facing). Walls only ever run along the world X or Y axis
    (co-layout's floor plans are Manhattan/grid-aligned), so a plain
    axis-aligned box always suffices -- no rotation is ever needed.

    Exterior walls are pushed fully to the outside of the cell boundary (their
    inner face sits exactly on it), so they never eat into the interior space
    the 2D solver already packed furniture into assuming zero-thickness
    walls. Interior partitions have no "outside" to push into -- both sides
    are occupied rooms -- so they stay centered on the boundary instead.
    """
    cx, cy = j * grid_scale, i * grid_scale
    half = grid_scale / 2.0
    offset = half + thickness / 2.0 if exterior else half
    if facing == "E":
        return cx + offset, cy, "y"
    if facing == "W":
        return cx - offset, cy, "y"
    if facing == "S":
        return cx, cy + offset, "x"
    return cx, cy - offset, "x"  # "N"


def _add_box(name: str, cx: float, cy: float, z_min: float, z_max: float, size_x: float, size_y: float):
    height = z_max - z_min
    if height <= 1e-4:
        return None
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, (z_min + z_max) / 2.0))
    box = bpy.context.active_object
    box.name = name
    box.dimensions = (size_x, size_y, height)
    return box


def _join_objects(objects: list, name: str):
    objects = [o for o in objects if o is not None]
    if not objects:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    return joined


def create_architecture(wall_edges: list, grid_scale: float, wall_height: float):
    """Procedurally build walls with door/window openings from ``wall_edges``
    (see ``_extract_wall_edges``): plain "wall" segments are a solid
    floor-to-ceiling box; "door" segments get a lintel above door height and
    a closed door-leaf panel filling the opening below it; "window" segments
    get a sill, a lintel, and a glass pane in between.
    """
    door_height = min(DOOR_HEIGHT, max(wall_height - 0.05, 0.1))
    window_bottom = min(WINDOW_SILL_HEIGHT, max(wall_height - 0.3, 0.0))
    window_top = min(window_bottom + WINDOW_HEIGHT, max(wall_height - 0.05, window_bottom))

    wall_parts = []
    glass_parts = []
    door_parts = []
    for idx, edge in enumerate(wall_edges):
        exterior = edge.get("exterior", True)
        thickness = EXTERIOR_WALL_THICKNESS if exterior else INTERIOR_WALL_THICKNESS
        cx, cy, axis = _wall_edge_geometry(edge["i"], edge["j"], edge["facing"], grid_scale, thickness, exterior)
        size_x = thickness if axis == "y" else grid_scale
        size_y = grid_scale if axis == "y" else thickness
        edge_type = edge["type"]

        if edge_type == "wall":
            wall_parts.append(_add_box(f"WallSeg_{idx:04d}", cx, cy, 0.0, wall_height, size_x, size_y))
        elif edge_type == "door":
            wall_parts.append(_add_box(f"DoorLintel_{idx:04d}", cx, cy, door_height, wall_height, size_x, size_y))
            leaf_thickness = thickness * 0.4
            leaf_size_x = leaf_thickness if axis == "y" else grid_scale * 0.92
            leaf_size_y = grid_scale * 0.92 if axis == "y" else leaf_thickness
            door_parts.append(_add_box(f"DoorLeaf_{idx:04d}", cx, cy, 0.02, door_height - 0.03, leaf_size_x, leaf_size_y))
        elif edge_type == "window":
            wall_parts.append(_add_box(f"WindowSill_{idx:04d}", cx, cy, 0.0, window_bottom, size_x, size_y))
            wall_parts.append(_add_box(f"WindowLintel_{idx:04d}", cx, cy, window_top, wall_height, size_x, size_y))
            glass_size_x = thickness * 0.3 if axis == "y" else grid_scale * 0.96
            glass_size_y = grid_scale * 0.96 if axis == "y" else thickness * 0.3
            glass_parts.append(_add_box(f"WindowGlass_{idx:04d}", cx, cy, window_bottom, window_top, glass_size_x, glass_size_y))

    walls_obj = _join_objects(wall_parts, "Walls")
    if walls_obj is not None:
        walls_obj.data.materials.append(_wall_material())
    glass_obj = _join_objects(glass_parts, "WindowGlass")
    if glass_obj is not None:
        glass_obj.data.materials.append(_glass_material())
    doors_obj = _join_objects(door_parts, "DoorLeaves")
    if doors_obj is not None:
        doors_obj.data.materials.append(_door_material())
    return walls_obj, glass_obj, doors_obj


# ---- Furniture asset import ----


def _new_imported_objects(before_object_names: set) -> list:
    return [obj for obj in bpy.data.objects if obj.name not in before_object_names]


def _join_imported_mesh_objects(imported_objects, imported_mesh_objects, asset_path: Path):
    if not imported_mesh_objects:
        print(f"WARNING: no MESH object found after import: {asset_path}")
        for obj in imported_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        return None

    imported_object_names = [obj.name for obj in imported_objects]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_mesh_objects[0]
    if len(imported_mesh_objects) > 1:
        bpy.ops.object.join()
    asset = bpy.context.active_object

    # FBX imports often bring in Armature/Empty root nodes alongside the mesh;
    # join() only removes the joined *mesh* objects, so clean up the rest by name.
    asset_name = asset.name if asset is not None else None
    for obj_name in imported_object_names:
        if obj_name == asset_name:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
    return asset


def _normalize_imported_asset_object(asset):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = asset
    asset.select_set(True)
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    asset.select_set(False)


def _apply_asset_transform(asset, name: str, center: list, size: list, rotation: list, jid: str):
    for key in list(asset.keys()):
        del asset[key]
    if asset.animation_data is not None:
        asset.animation_data_clear()
    asset.parent = None
    for constraint in list(asset.constraints):
        asset.constraints.remove(constraint)

    asset.name = name
    if getattr(asset, "data", None) is not None:
        asset.data.name = f"{name}_mesh"
    asset.scale = (1.0, 1.0, 1.0)
    asset.rotation_euler = (0.0, 0.0, 0.0)
    asset.location = center
    asset.dimensions = size
    # object.dimensions is a write-through property backed by a cached
    # evaluated bounding box: the scale it computes (visible via asset.scale)
    # is correct immediately, but re-reading .dimensions right away can still
    # report the pre-update value until the depsgraph is refreshed. Force
    # that refresh now (while still unrotated, so the check below compares
    # like-for-like local axes) so it reports the true, final size.
    bpy.context.view_layer.update()
    actual = tuple(round(v, 3) for v in asset.dimensions)
    target = tuple(round(v, 3) for v in size)
    if any(abs(a - t) > 0.01 for a, t in zip(actual, target)):
        print(f"WARNING: '{name}' dimensions after scaling {actual} deviate from target size {target}")

    asset.rotation_euler = (math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))
    asset["jid"] = jid
    return asset


def import_imaginarium_asset(assets_dir: Path, jid: str, name: str, center: list, size: list, rotation: list):
    """Import a single Imaginarium FBX asset and place it in the scene."""
    asset_path = assets_dir / jid / f"{jid}.fbx"
    if not asset_path.exists():
        print(f"WARNING: Imaginarium asset not found: {asset_path}")
        return None

    before_object_names = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(asset_path))

    imported_objects = _new_imported_objects(before_object_names)
    imported_mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    asset = _join_imported_mesh_objects(imported_objects, imported_mesh_objects, asset_path)
    if asset is None:
        return None

    _normalize_imported_asset_object(asset)
    return _apply_asset_transform(asset, name, center, size, rotation, jid)


# ---- World axes (optional debugging aid) ----


def create_world_axes(length: float = 1.0, origin: tuple = (0.0, 0.0, 0.0)):
    """Create RGB axis indicators (X=red, Y=green, Z=blue) for orientation debugging."""
    shaft_radius = 0.015
    head_length = length * 0.12
    head_radius = shaft_radius * 3.0
    shaft_length = length - head_length

    axis_specs = {
        "X": ((1.0, 0.1, 0.1, 1.0), (0.0, math.radians(90.0), 0.0)),
        "Y": ((0.1, 0.85, 0.1, 1.0), (math.radians(-90.0), 0.0, 0.0)),
        "Z": ((0.1, 0.3, 1.0, 1.0), (0.0, 0.0, 0.0)),
    }

    created_names = []
    for axis_name, (color, rot_euler) in axis_specs.items():
        bpy.ops.mesh.primitive_cylinder_add(radius=shaft_radius, depth=shaft_length, location=(0.0, 0.0, shaft_length / 2.0))
        shaft = bpy.context.active_object
        bpy.ops.mesh.primitive_cone_add(radius1=head_radius, radius2=0.0, depth=head_length, location=(0.0, 0.0, shaft_length + head_length / 2.0))
        head = bpy.context.active_object

        bpy.ops.object.select_all(action="DESELECT")
        shaft.select_set(True)
        head.select_set(True)
        bpy.context.view_layer.objects.active = shaft
        bpy.ops.object.join()
        axis_obj = bpy.context.active_object
        axis_obj.name = f"WorldAxis_{axis_name}"
        axis_obj.rotation_euler = rot_euler

        mat = bpy.data.materials.new(name=f"WorldAxis_{axis_name}_mat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = 4.0
        output = nodes.new(type="ShaderNodeOutputMaterial")
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        axis_obj.data.materials.clear()
        axis_obj.data.materials.append(mat)
        created_names.append(axis_obj.name)

    ox, oy, oz = origin
    if (ox, oy, oz) != (0.0, 0.0, 0.0):
        for name in created_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.location.x += ox
                obj.location.y += oy
                obj.location.z += oz
        bpy.context.view_layer.update()

    print(f"[Axes] Created world axes (length={length}m, origin={origin}): {created_names}")
    return created_names


# ---- Lighting + camera ----


def get_scene_bounds():
    """Compute the combined bounding box of all visible mesh objects."""
    from mathutils import Vector

    min_co = Vector((float("inf"),) * 3)
    max_co = Vector((float("-inf"),) * 3)
    found = False

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            min_co = Vector(map(min, zip(min_co, world_co)))
            max_co = Vector(map(max, zip(max_co, world_co)))
            found = True

    if not found:
        return Vector((-3, -3, 0)), Vector((3, 3, 2.8))

    return min_co, max_co


def remove_existing_setup_objects():
    """Remove all camera and light objects (and their data blocks) from the scene."""
    to_delete = [obj for obj in bpy.data.objects if obj.type in ("CAMERA", "LIGHT")]
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

    for cam_data in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam_data)
    for light_data in list(bpy.data.lights):
        bpy.data.lights.remove(light_data)


def get_floor_center(default_min, default_max):
    """Prefer the "Floor" mesh's bounding-box center as the camera look-at target."""
    from mathutils import Vector

    floor = bpy.data.objects.get("Floor")
    if floor is not None and floor.type == "MESH" and floor.visible_get():
        corners = [floor.matrix_world @ Vector(corner) for corner in floor.bound_box]
        min_co = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
        max_co = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
        return (min_co + max_co) / 2
    return Vector(((default_min.x + default_max.x) / 2, (default_min.y + default_max.y) / 2, default_min.z))


def add_area_light(name, location, rotation_euler, size, energy, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = rotation_euler
    light = obj.data
    light.name = name + "_data"
    light.size = size
    light.energy = energy
    light.color = color
    light.use_shadow = True
    light.shadow_soft_size = size * 0.5
    return obj


def add_spot_light(name, location, rotation_euler, energy, spot_size_deg, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="SPOT", location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = rotation_euler
    light = obj.data
    light.name = name + "_data"
    light.energy = energy
    light.spot_size = math.radians(spot_size_deg)
    light.spot_blend = 0.4
    light.color = color
    light.use_shadow = True
    return obj


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v):
    length = math.sqrt(_dot(v, v))
    if length == 0:
        raise ValueError(f"Cannot normalize a zero vector: {v}")
    return tuple(x / length for x in v)


def _camera_basis(azimuth_deg, elevation_deg):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    view_from_target = _normalize(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
    )
    forward = tuple(-x for x in view_from_target)
    world_up = (0.0, 0.0, 1.0)
    right = _cross(forward, world_up)
    if _dot(right, right) < 1e-12:
        right = (1.0, 0.0, 0.0)
    else:
        right = _normalize(right)
    up = _normalize(_cross(right, forward))
    return view_from_target, right, up


def _bounds_corners(bounds_min, bounds_max, target):
    return [
        (x - target[0], y - target[1], z - target[2])
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
    ]


def _required_camera_distance(corners, azimuth_deg, elevation_deg, fov_h, fov_v):
    view_from_target, right, up = _camera_basis(azimuth_deg, elevation_deg)
    tan_h = math.tan(fov_h / 2)
    tan_v = math.tan(fov_v / 2)

    required_distance = 0.0
    for corner in corners:
        along_camera = _dot(corner, view_from_target)
        required_distance = max(
            required_distance,
            along_camera + abs(_dot(corner, right)) / tan_h,
            along_camera + abs(_dot(corner, up)) / tan_v,
        )
    return required_distance


def calculate_camera_distance_for_bounds(bounds_min, bounds_max, target, azimuth_deg, elevation_deg, fov_h, fov_v, framing_margin=1.0):
    """Distance needed so the camera frustum just contains the real scene bounds."""
    corners = _bounds_corners(bounds_min, bounds_max, target)
    required_distance = _required_camera_distance(corners=corners, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, fov_h=fov_h, fov_v=fov_v)
    bounds_size = (
        bounds_max[0] - bounds_min[0],
        bounds_max[1] - bounds_min[1],
        bounds_max[2] - bounds_min[2],
    )
    return max(required_distance * framing_margin, max(bounds_size) * 0.5)


def setup_camera(center, room_w, room_d, room_h, azimuth_deg=270.0, elevation_deg=24.0, focal_length_mm=35.0, target_center=None, bounds_min=None, bounds_max=None):
    """Place a top-down-ish camera outside the scene bounds, framing the whole scene."""
    from mathutils import Vector

    scene = bpy.context.scene
    render = scene.render
    aspect = render.resolution_x / render.resolution_y if render.resolution_y else 1.0
    sensor_w = 36.0
    fov_h = 2 * math.atan(sensor_w / (2 * focal_length_mm))
    fov_v = 2 * math.atan(sensor_w / aspect / (2 * focal_length_mm))

    target = Vector(target_center) if target_center is not None else center
    if bounds_min is not None and bounds_max is not None:
        dist = calculate_camera_distance_for_bounds(
            bounds_min=tuple(bounds_min),
            bounds_max=tuple(bounds_max),
            target=tuple(target),
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            fov_h=fov_h,
            fov_v=fov_v,
        )
    else:
        required_distance = _required_camera_distance(
            corners=[(x, y, z) for x in (-room_w / 2, room_w / 2) for y in (-room_d / 2, room_d / 2) for z in (-room_h / 2, room_h / 2)],
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            fov_h=fov_h,
            fov_v=fov_v,
        )
        dist = max(required_distance, max(room_w, room_d, room_h) * 0.5)
    view_from_target, _, _ = _camera_basis(azimuth_deg, elevation_deg)

    cam_x = target.x + dist * view_from_target[0]
    cam_y = target.y + dist * view_from_target[1]
    cam_z = target.z + dist * view_from_target[2]

    cam_data = bpy.data.cameras.new("Camera_data")
    cam_data.lens = focal_length_mm
    cam_data.clip_start = 0.1
    cam_data.clip_end = max(room_w, room_d, room_h) * 20

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (cam_x, cam_y, cam_z)

    direction = target - Vector((cam_x, cam_y, cam_z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    scene.camera = cam_obj

    print(f"[Camera] Placed at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})")
    print(f"[Camera]   azimuth={azimuth_deg}deg elevation={elevation_deg}deg lens={focal_length_mm}mm distance={dist:.2f}m")
    print(f"[Camera]   look-at target: ({target.x:.2f}, {target.y:.2f}, {target.z:.2f})")
    return cam_obj


def _configure_cycles_gpu(scene, preferred_backend="AUTO"):
    """Best-effort enable of a Cycles GPU device; falls back to CPU if unavailable."""
    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception as exc:
        print(f"[Render] WARNING: cannot access Cycles device preferences, using CPU: {exc}")
        scene.cycles.device = "CPU"
        return "CPU"

    requested_backend = (preferred_backend or "AUTO").upper()
    default_backends = ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL")
    backends = default_backends if requested_backend == "AUTO" else (requested_backend,)

    for backend in backends:
        try:
            cycles_prefs.compute_device_type = backend
        except Exception:
            continue

        try:
            if hasattr(cycles_prefs, "refresh_devices"):
                cycles_prefs.refresh_devices()
            elif hasattr(cycles_prefs, "get_devices"):
                cycles_prefs.get_devices()
        except Exception:
            pass

        enabled = []
        for device in getattr(cycles_prefs, "devices", []):
            device_type = str(getattr(device, "type", "")).upper()
            is_gpu = device_type not in {"", "NONE", "CPU"}
            try:
                device.use = is_gpu
            except Exception:
                pass
            if is_gpu:
                enabled.append(f"{device_type}:{getattr(device, 'name', 'unknown')}")

        if enabled:
            scene.cycles.device = "GPU"
            print(f"[Render] Cycles GPU backend: {backend}")
            print(f"[Render] Enabled devices: {', '.join(enabled)}")
            return backend

    scene.cycles.device = "CPU"
    print("[Render] WARNING: no usable GPU device found, using CPU")
    return "CPU"


def _set_render_engine(scene, render_engine: str):
    """Switch render engine, tolerating Blender-version differences in engine IDs."""
    name = (render_engine or "EEVEE").upper()
    candidates = ("CYCLES",) if name == "CYCLES" else ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")

    for engine_id in candidates:
        try:
            if scene.render.engine != engine_id:
                scene.render.engine = engine_id
                print(f"[Render] Switched engine to {engine_id}")
            return
        except (TypeError, ValueError):
            continue
    print(f"[Render] WARNING: engine {render_engine} not supported here, keeping {scene.render.engine}")


def setup_indoor_lighting_and_render(
    brightness=1.0,
    azimuth_deg=235.0,
    elevation_deg=55.0,
    focal_length_mm=35.0,
    cycles_samples=512,
    transparent_bg=True,
    render_engine="BLENDER_EEVEE_NEXT",
    resolution_x=1920,
    resolution_y=1920,
    world_color=(1.0, 1.0, 1.0, 1.0),
    world_strength=2.0,
    use_gpu=True,
    cycles_gpu_backend="AUTO",
    eevee_samples=64,
    use_raytracing=True,
    use_eevee_ao=True,
):
    """Apply an auto-framed camera, a simple indoor 3-light rig, and render settings."""
    remove_existing_setup_objects()

    min_co, max_co = get_scene_bounds()
    center = (min_co + max_co) / 2
    camera_target = get_floor_center(min_co, max_co)
    room_size = max_co - min_co
    room_w, room_d, room_h = room_size.x, room_size.y, room_size.z
    ceil_z = max_co.z + 0.3
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution_x)
    scene.render.resolution_y = int(resolution_y)
    scene.render.resolution_percentage = 100

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = world_color
    bg_node.inputs["Strength"].default_value = world_strength * brightness

    try:
        world.light_settings.use_ambient_occlusion = True
        world.light_settings.ao_factor = 0.6
        world.light_settings.distance = 1.0
    except AttributeError:
        pass

    print(f"[Lighting] Scene bounds: {min_co} -> {max_co}")
    print(f"[Lighting] Room size: W={room_w:.2f}m D={room_d:.2f}m H={room_h:.2f}m")
    print(f"[Camera] Floor-center look-at target: {camera_target}")

    main_size = min(room_w, room_d) * 0.7
    main_energy = 200 * brightness
    add_area_light(name="MainCeiling", location=(center.x, center.y, ceil_z), rotation_euler=(0, 0, 0), size=main_size, energy=main_energy, color=(1.0, 0.97, 0.88))

    fill_z = center.z + room_h * 0.5
    fill_dist = max(room_w, room_d) * 0.65
    fill_energy = main_energy * 0.35
    fill_size = min(room_w, room_d) * 0.5
    add_area_light(name="FillA", location=(center.x + fill_dist, center.y, fill_z), rotation_euler=(math.radians(-45), 0, math.radians(90)), size=fill_size, energy=fill_energy, color=(0.95, 0.98, 1.0))
    add_area_light(name="FillB", location=(center.x, center.y - fill_dist, fill_z), rotation_euler=(math.radians(-45), 0, 0), size=fill_size, energy=fill_energy * 0.7, color=(0.95, 0.98, 1.0))
    add_spot_light(name="Rim", location=(center.x - room_w * 0.5, center.y + room_d * 0.5, ceil_z + room_h * 0.4), rotation_euler=(math.radians(-120), 0, math.radians(-45)), energy=main_energy * 0.15, spot_size_deg=60, color=(1.0, 0.95, 0.85))

    setup_camera(
        center=center,
        room_w=room_w,
        room_d=room_d,
        room_h=room_h,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        focal_length_mm=focal_length_mm,
        target_center=camera_target,
        bounds_min=min_co,
        bounds_max=max_co,
    )

    _set_render_engine(scene, render_engine)
    if use_gpu and scene.render.engine == "CYCLES":
        _configure_cycles_gpu(scene, preferred_backend=cycles_gpu_backend)

    scene.render.film_transparent = transparent_bg
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.samples = cycles_samples
    except Exception:
        pass

    try:
        scene.eevee.taa_render_samples = int(eevee_samples)
    except Exception:
        pass

    try:
        scene.eevee.use_ssr = use_raytracing
        scene.eevee.use_ssr_refraction = use_raytracing
        scene.eevee.use_gtao = use_eevee_ao
        scene.eevee.gtao_distance = 0.5
    except Exception:
        pass
    try:
        scene.eevee.use_raytracing = use_raytracing
    except Exception:
        pass

    print("[Lighting] Done: 4 lights + world ambient + AO")
    print(f"[Render] engine={scene.render.engine} transparent_bg={transparent_bg}")


# ---- Scene build ----


def build_and_render(
    layout_path,
    assets_dir: Path,
    floor_texture=DEFAULT_FLOOR_TEXTURE,
    texture_scale: float = 0.5,
    show_axes: bool = False,
    axis_length: float = 1.0,
    render_image_path=None,
    render_resolution: tuple = (1920, 1920),
    render_engine: str = "BLENDER_EEVEE_NEXT",
    camera_azimuth: float = 235.0,
    camera_elevation: float = 55.0,
    save_blend_path=None,
):
    layout_path = Path(layout_path)
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_path}")
    with open(layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    clear_scene()
    bpy.context.scene.tool_settings.transform_pivot_point = "BOUNDING_BOX_CENTER"
    scene_root = bpy.context.scene.collection

    session_id = layout.get("session_id", layout_path.stem)
    print(f"\n{'=' * 60}\nBuilding scene: {session_id}\n{'=' * 60}")

    grid_scale = layout.get("grid_scale", 1.0)
    wall_height = layout.get("wall_height", 2.8)

    architecture_collection = _get_or_create_collection("Architecture", scene_root)
    _set_active_collection(architecture_collection)

    occupied_cells = layout.get("occupied_cells", [])
    create_floor_from_grid(occupied_cells, grid_scale=grid_scale, texture_path=floor_texture, texture_scale=texture_scale)
    print(f"Floor built from {len(occupied_cells)} grid cells")

    wall_edges = layout.get("wall_edges", [])
    create_architecture(wall_edges, grid_scale=grid_scale, wall_height=wall_height)
    print(f"Architecture built from {len(wall_edges)} wall/door/window segments (wall height {wall_height:.2f}m)")

    objects = layout.get("objects", [])
    print(f"Object count: {len(objects)}")
    print("-" * 60)
    success_count = 0
    room_collections = {}
    for obj_data in objects:
        obj_id = obj_data["id"]
        name = obj_data.get("name", f"object_{obj_id}")
        jid = obj_data.get("jid")
        if not jid:
            print(f"WARNING: object {obj_id} ('{name}') has no jid, skipping")
            continue

        room_name = obj_data.get("room_name") or "Unassigned"
        room_collection = room_collections.get(room_name)
        if room_collection is None:
            room_collection = _get_or_create_collection(room_name, scene_root)
            room_collections[room_name] = room_collection
        _set_active_collection(room_collection)

        obj_name = f"{obj_id:02d}_{name.replace(' ', '_')}"
        asset = import_imaginarium_asset(assets_dir, jid, obj_name, obj_data["center"], obj_data["size"], obj_data.get("rotation", [0, 0, 0]))
        if asset is None:
            print(f"WARNING: failed to import asset for object {obj_id} ('{name}', jid={jid})")
            continue

        asset["room_name"] = obj_data.get("room_name", "")
        asset["asset_score"] = obj_data.get("asset_score", 0.0)
        success_count += 1
    print("-" * 60)
    print(f"Successfully placed {success_count}/{len(objects)} objects")
    print(f"Room collections: {sorted(room_collections.keys())}")

    lighting_collection = _get_or_create_collection("Lighting_Camera", scene_root)
    _set_active_collection(lighting_collection)
    print("Applying automatic lighting, camera, and render settings...")
    setup_indoor_lighting_and_render(
        render_engine=render_engine,
        resolution_x=render_resolution[0],
        resolution_y=render_resolution[1],
        azimuth_deg=camera_azimuth,
        elevation_deg=camera_elevation,
    )

    if show_axes:
        debug_collection = _get_or_create_collection("Debug", scene_root)
        _set_active_collection(debug_collection)
        create_world_axes(length=axis_length, origin=(0.0, 0.0, wall_height + 0.2))

    _set_active_collection(scene_root)

    if save_blend_path:
        save_blend_file(save_blend_path)

    if render_image_path:
        render_image_file(render_image_path)

    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")


# ============================================================================
# CLI entry points
# ============================================================================

_COMMON_BLENDER_PATHS = [
    "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS, no version in the app bundle name
]


def _find_blender_binary():
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin

    which_result = shutil.which("blender")
    if which_result:
        return which_result

    for pattern in ["/Applications/Blender*.app/Contents/MacOS/Blender"] + _COMMON_BLENDER_PATHS:
        for candidate in glob.glob(pattern):
            if Path(candidate).exists():
                return candidate
    return None


def _build_blender_command(blender_bin, layout_path, render_image, render_resolution, floor_texture, texture_scale, show_axes, axis_length, render_engine, camera_azimuth, camera_elevation, save_blend):
    cmd = [blender_bin, "-b", "-P", str(Path(__file__).resolve()), "--", "--layout", str(layout_path)]
    if render_image:
        cmd += ["--render-image", render_image]
    cmd += ["--render-resolution", str(render_resolution[0]), str(render_resolution[1])]
    if floor_texture:
        cmd += ["--floor-texture", floor_texture]
    cmd += ["--texture-scale", str(texture_scale)]
    if show_axes:
        cmd += ["--show-axes", "--axis-length", str(axis_length)]
    cmd += ["--render-engine", render_engine]
    cmd += ["--camera-azimuth", str(camera_azimuth), "--camera-elevation", str(camera_elevation)]
    if save_blend:
        cmd += ["--save-blend", save_blend]
    return cmd


def _main_driver():
    parser = argparse.ArgumentParser(description="Export a co-layout optimization result to a 3D layout and render it in Blender.")
    parser.add_argument("--session", "-s", type=str, required=True, help="Session ID (same as run_optimization.py --session)")
    parser.add_argument("--result", type=str, default=None, help="Path to a specific result.json (default: output/sessions/<session>/optimization/result.json)")
    parser.add_argument("--output", type=str, default=None, help="Output layout JSON path (default: output/sessions/<session>/visualization/layout_3d.json)")
    parser.add_argument("--asset-index", type=str, default=str(ASSET_INDEX_DEFAULT_PATH), help="Path to the built asset index (see asset_library/asset_retriever.py)")
    parser.add_argument("--embedding-model", type=str, default=None, help="sentence-transformers model name; must match the one used to build the index")
    parser.add_argument("--size-tolerance", type=float, default=0.5, help="Relative size tolerance for asset retrieval scoring")
    parser.add_argument("--min-score", type=float, default=0.2, help="Minimum retrieval score before falling back to the best available match")

    parser.add_argument("--render-image", type=str, default=None, help="Where to render a still image (default: <layout>_render.png next to the layout JSON)")
    parser.add_argument("--render-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=[1920, 1920])
    parser.add_argument("--floor-texture", type=str, default=None, help=f"Floor texture image path override (default: the builder's own default, {DEFAULT_FLOOR_TEXTURE})")
    parser.add_argument("--texture-scale", type=float, default=0.5)
    parser.add_argument("--show-axes", action="store_true", help="Draw an RGB world-axis indicator above the scene (debugging aid)")
    parser.add_argument("--axis-length", type=float, default=1.0)
    parser.add_argument("--render-engine", type=str, default="BLENDER_EEVEE_NEXT", choices=["BLENDER_EEVEE_NEXT", "EEVEE", "CYCLES"])
    parser.add_argument("--camera-azimuth", type=float, default=235.0, help="Camera azimuth around the scene, in degrees")
    parser.add_argument("--camera-elevation", type=float, default=55.0, help="Camera elevation above the horizon, in degrees (higher = more top-down, needed to see over the walls)")
    parser.add_argument("--save-blend", type=str, default=None, help="Also save the built scene to this .blend path")

    parser.add_argument("--auto-render", action="store_true", help="Automatically invoke Blender as a subprocess instead of just printing the command")
    parser.add_argument("--blender-bin", type=str, default=None, help="Path to the Blender executable (default: auto-detect / $BLENDER_BIN)")
    args = parser.parse_args()

    layout_path = export_layout(
        session_id=args.session,
        result_path=Path(args.result) if args.result else None,
        asset_index_path=Path(args.asset_index),
        embedding_model=args.embedding_model,
        size_tolerance=args.size_tolerance,
        min_score=args.min_score,
        output_path=Path(args.output) if args.output else None,
    )

    render_image = args.render_image or str(layout_path.with_name("render.png"))

    blender_bin = args.blender_bin or _find_blender_binary()
    if not blender_bin:
        print("\n[blender_visualization] Could not auto-detect a Blender executable. Pass --blender-bin, or set the BLENDER_BIN environment variable.")
        blender_bin = "blender"

    cmd = _build_blender_command(
        blender_bin=blender_bin,
        layout_path=layout_path,
        render_image=render_image,
        render_resolution=tuple(args.render_resolution),
        floor_texture=args.floor_texture,
        texture_scale=args.texture_scale,
        show_axes=args.show_axes,
        axis_length=args.axis_length,
        render_engine=args.render_engine,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        save_blend=args.save_blend,
    )

    if args.auto_render:
        print(f"[blender_visualization] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)
        print(f"[blender_visualization] Done. Render: {render_image}")
    else:
        print("\n[blender_visualization] Layout exported. Rendering must run inside Blender's own process; run:\n")
        print(shlex.join(cmd))
        print("\n(or re-run this script with --auto-render to have it invoked automatically)\n")


def _main_builder():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    from asset_library.paths import IMAGINARIUM_ASSETS_DIR

    parser = argparse.ArgumentParser(description="Build and render a co-layout 3D layout JSON inside Blender.")
    parser.add_argument("--layout", "-l", type=str, required=True, help="Path to the *_layout_3d.json produced by the driver step")
    parser.add_argument("--assets-dir", type=str, default=str(IMAGINARIUM_ASSETS_DIR), help="Path to imaginarium_assets/ (default: asset_library/paths.py's IMAGINARIUM_ASSETS_DIR)")
    parser.add_argument("--floor-texture", "-f", type=str, default=DEFAULT_FLOOR_TEXTURE, help=f"Floor texture image path (default: {DEFAULT_FLOOR_TEXTURE})")
    parser.add_argument("--texture-scale", "-s", type=float, default=0.5, help="Floor texture tiling scale")
    parser.add_argument("--show-axes", action="store_true", help="Draw an RGB world-axis indicator above the scene (debugging aid)")
    parser.add_argument("--axis-length", type=float, default=1.0, help="World axis length in meters")
    parser.add_argument("--render-image", type=str, default=None, help="Render a still image to this path")
    parser.add_argument("--render-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=[1920, 1920], help="Render resolution")
    parser.add_argument("--render-engine", type=str, default="BLENDER_EEVEE_NEXT", choices=["BLENDER_EEVEE_NEXT", "EEVEE", "CYCLES"], help="Render engine")
    parser.add_argument("--camera-azimuth", type=float, default=235.0, help="Camera azimuth around the scene, in degrees")
    parser.add_argument("--camera-elevation", type=float, default=55.0, help="Camera elevation above the horizon, in degrees (higher = more top-down, needed to see over the walls)")
    parser.add_argument("--save-blend", type=str, default=None, help="Save the built scene to this .blend path")
    args = parser.parse_args(argv)

    build_and_render(
        layout_path=args.layout,
        assets_dir=Path(args.assets_dir),
        floor_texture=args.floor_texture,
        texture_scale=args.texture_scale,
        show_axes=args.show_axes,
        axis_length=args.axis_length,
        render_image_path=args.render_image,
        render_resolution=tuple(args.render_resolution),
        render_engine=args.render_engine,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        save_blend_path=args.save_blend,
    )


def main():
    if _IN_BLENDER:
        _main_builder()
    else:
        _main_driver()


if __name__ == "__main__":
    main()
