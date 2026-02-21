import os

from gurobipy import GRB, quicksum

from constants import PATH_OF_OPTIMIZATION_LOGS, PATH_OF_OPTIMIZATION_MODELS
from utils.find_boundary import find_boundary_coordinates


def configure_model_log(model, model_name: str) -> None:
    """Configure a clean log file for a Gurobi model."""
    log_file_path = os.path.join(PATH_OF_OPTIMIZATION_LOGS, f"{model_name}.log")
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
        except PermissionError:
            print(
                f"WARNING: Unable to delete log file {log_file_path}. "
                "The file may be in use by another process. "
                "The script will attempt to continue execution."
            )
    model.setParam("LogFile", log_file_path)


def run_staged_optimization(
    model,
    model_name: str,
    objective_function,
    stage_num: int,
    callback=None,
) -> None:
    """Run one-stage or two-stage optimization with shared settings."""
    model.write(os.path.join(PATH_OF_OPTIMIZATION_MODELS, f"{model_name}.lp"))

    if stage_num == 1:
        model.setParam("MIPFocus", 2)
        model.setObjective(objective_function, GRB.MINIMIZE)
        model.optimize(callback) if callback else model.optimize()
        return

    if stage_num == 2:
        model.setParam("MIPFocus", 1)
        model.setObjective(0)  # find a feasible solution
        model.optimize()
        if model.status == GRB.OPTIMAL:
            model.update()
            model.setParam("MIPFocus", 1)
            model.setObjective(objective_function, GRB.MINIMIZE)  # optimize the objective function
            model.optimize(callback) if callback else model.optimize()
        return

    raise ValueError(f"Unsupported stage_num: {stage_num}")


def add_room_corner_constraints(
    model,
    x,
    room_constraints: dict,
    room_name_list: list,
    north_west: tuple,
    north_east: tuple,
    south_west: tuple,
    south_east: tuple,
) -> None:
    """Add room-at-corner constraints for available room names."""
    corner_rooms = room_constraints["corner_rooms"]
    corner_specs = [
        ("North West", north_west, "nw"),
        ("North East", north_east, "ne"),
        ("South West", south_west, "sw"),
        ("South East", south_east, "se"),
    ]
    for room_label, corner_coordinate, corner_code in corner_specs:
        room_index = (
            room_name_list.index(corner_rooms[room_label])
            if corner_rooms[room_label] in room_name_list
            else None
        )
        if room_index is not None:
            model.addConstr(
                x[room_index, corner_coordinate[0], corner_coordinate[1]] == 1,
                name=f"room_at_corner_{corner_code}_{room_index}",
            )


def add_room_bounding_box_constraints(
    model,
    x,
    room_num: int,
    valid_coordinates: list,
    big_m: int,
    rect_min_i,
    rect_len_i,
    rect_min_j,
    rect_len_j,
) -> None:
    """Link room occupancy variables with room bounding-box variables."""
    for k in range(room_num):
        for i, j in valid_coordinates:
            model.addConstr(
                rect_min_i[k] <= i + big_m * (1 - x[k, i, j]),
                name=f"bbox_min_i_{k}_{i}_{j}",
            )
            model.addConstr(
                rect_min_i[k] + rect_len_i[k] - 1 >= i - big_m * (1 - x[k, i, j]),
                name=f"bbox_max_i_{k}_{i}_{j}",
            )
            model.addConstr(
                rect_min_j[k] <= j + big_m * (1 - x[k, i, j]),
                name=f"bbox_min_j_{k}_{i}_{j}",
            )
            model.addConstr(
                rect_min_j[k] + rect_len_j[k] - 1 >= j - big_m * (1 - x[k, i, j]),
                name=f"bbox_max_j_{k}_{i}_{j}",
            )


def build_layout_context(
    width: int,
    length: int,
    rooms: dict,
    outdoor_space_coordinates: list,
    print_big_m: bool = False,
) -> dict:
    """Build common pre-process context shared by coarse and fine models."""
    room_num = len(rooms)
    room_name_list = list(rooms.keys())
    room_area_list = [int(rooms[room]["area"]) for room in room_name_list]

    total_area = width * length
    total_room_area = sum(room_area_list)
    outdoor_space_area = len(outdoor_space_coordinates)
    usable_area = total_area - outdoor_space_area

    if total_room_area < 0.9 * usable_area or total_room_area > 0.99 * usable_area:
        print("\n Automatically adjust room area")
        target_total_room_area = 0.9 * usable_area
        room_area_list = [
            int(area * target_total_room_area / total_room_area)
            for area in room_area_list
        ]

    valid_coordinates = [
        (i, j)
        for i in range(width)
        for j in range(length)
        if (i, j) not in outdoor_space_coordinates
    ]
    valid_coordinates_set = set(valid_coordinates)

    big_m = width * length
    if print_big_m:
        print("\n" + "-" * 25)
        print(f"Set Big M = {big_m}")
        print("-" * 25)

    open_room_indices = []
    common_room_indices = []
    for room_index, room_name in enumerate(room_name_list):
        if rooms[room_name]["is_open"]:
            open_room_indices.append(room_index)
        else:
            common_room_indices.append(room_index)

    boundary = find_boundary_coordinates(width, length, outdoor_space_coordinates)
    boundary_min_i = min(p[0] for p in boundary)
    candidates = [p for p in boundary if p[0] == boundary_min_i]
    boundary_min_j = min(p[1] for p in candidates)
    north_west = (boundary_min_i, boundary_min_j)

    boundary_max_j = max(p[1] for p in candidates)
    north_east = (boundary_min_i, boundary_max_j)

    boundary_max_i = max(p[0] for p in boundary)
    candidates = [p for p in boundary if p[0] == boundary_max_i]
    boundary_min_j = min(p[1] for p in candidates)
    south_west = (boundary_max_i, boundary_min_j)

    boundary_max_j = max(p[1] for p in candidates)
    south_east = (boundary_max_i, boundary_max_j)

    privacy_order = []
    for room_index, room_name in enumerate(room_name_list):
        privacy_level = rooms[room_name].get("privacy_level", 1)
        privacy_order.append((privacy_level, room_index))
    privacy_order.sort(reverse=True)
    privacy_order = [index for _, index in privacy_order]

    return {
        "BigM": big_m,
        "room_num": room_num,
        "room_name_list": room_name_list,
        "room_area_list": room_area_list,
        "valid_coordinates": valid_coordinates,
        "valid_coordinates_set": valid_coordinates_set,
        "open_room_indices": open_room_indices,
        "common_room_indices": common_room_indices,
        "boundary": boundary,
        "north_west": north_west,
        "north_east": north_east,
        "south_west": south_west,
        "south_east": south_east,
        "privacy_order": privacy_order,
    }


def add_room_adjacency_constraints(
    model,
    x,
    room_constraints: dict,
    room_name_list: list,
    valid_coordinates: list,
) -> None:
    """Add pairwise room adjacency constraints."""
    adjacent_rooms = room_constraints["room_adjacency"]
    adjacent_pairs = []
    for room1, room2 in adjacent_rooms:
        try:
            index1 = room_name_list.index(room1)
            index2 = room_name_list.index(room2)
            adjacent_pairs.append((index1, index2))
        except ValueError:
            print(
                f"Warning: Room name not found in list during adjacency setup: {room1} or {room2}"
            )
            continue

    for k, l in adjacent_pairs:
        room_adj = model.addVars(
            valid_coordinates,
            vtype=GRB.BINARY,
            name=f"room_adj_{k}_{l}",
        )
        model.addConstr(
            quicksum(room_adj[i, j] for i, j in valid_coordinates) >= 1,
            name=f"room_adj_at_least_one_{k}_{l}",
        )
        for i, j in valid_coordinates:
            neighbors = (
                x[l, i + 1, j]
                + x[l, i - 1, j]
                + x[l, i, j + 1]
                + x[l, i, j - 1]
            )
            model.addConstr(
                neighbors >= room_adj[i, j],
                name=f"room_adj_neighbor_link_{k}_{l}_{i}_{j}",
            )
            model.addConstr(
                room_adj[i, j] <= x[k, i, j],
                name=f"room_adj_room_link_{k}_{l}_{i}_{j}",
            )
