import os
import time
from dataclasses import dataclass

from gurobipy import GRB, quicksum

from utils.find_boundary import find_boundary_coordinates


class ConnectivityOptimizationError(RuntimeError):
    """Raised when connectivity cuts cannot produce a usable incumbent."""


def exact_bounded_integer_product(
    model,
    left,
    right,
    *,
    left_upper_bound: int,
    right_upper_bound: int,
    name: str,
):
    """Return an exact linear expression for ``left * right``.

    Both integer variables must have domain ``1..upper_bound``.  The smaller
    domain is represented by one-hot selectors, and each selector gates the
    other variable with an exact binary-times-bounded-variable formulation.
    """
    if left_upper_bound < 1 or right_upper_bound < 1:
        raise ValueError("Integer-product upper bounds must be positive")

    if left_upper_bound <= right_upper_bound:
        selected = left
        selected_upper_bound = left_upper_bound
        other = right
        other_upper_bound = right_upper_bound
    else:
        selected = right
        selected_upper_bound = right_upper_bound
        other = left
        other_upper_bound = left_upper_bound

    values = range(1, selected_upper_bound + 1)
    selectors = model.addVars(
        values,
        vtype=GRB.BINARY,
        name=f"{name}_select",
    )
    gated_other = model.addVars(
        values,
        lb=0,
        ub=other_upper_bound,
        name=f"{name}_gated",
    )
    model.addConstr(
        quicksum(selectors[value] for value in values) == 1,
        name=f"{name}_select_one",
    )
    model.addConstr(
        selected
        == quicksum(
            value * selectors[value]
            for value in values
        ),
        name=f"{name}_selected_value",
    )
    for value in values:
        model.addConstr(
            gated_other[value]
            <= other_upper_bound * selectors[value],
            name=f"{name}_gate_off_{value}",
        )
        model.addConstr(
            gated_other[value] <= other,
            name=f"{name}_gate_upper_{value}",
        )
        model.addConstr(
            gated_other[value]
            >= other
            - other_upper_bound * (1 - selectors[value]),
            name=f"{name}_gate_lower_{value}",
        )

    return quicksum(
        value * gated_other[value]
        for value in values
    )


def add_room_rectangularity_constraints(
    model,
    x,
    *,
    room_num: int,
    valid_coordinates,
    rect_len_i,
    rect_len_j,
    width: int,
    length: int,
):
    """Create exact linear room bounding-box area slacks."""
    rect_diff_slack = model.addVars(
        room_num,
        vtype=GRB.INTEGER,
        lb=0,
        name="rect_diff_slack",
    )
    for room_index in range(room_num):
        bounding_box_area = exact_bounded_integer_product(
            model,
            rect_len_i[room_index],
            rect_len_j[room_index],
            left_upper_bound=width,
            right_upper_bound=length,
            name=f"bbox_area_{room_index}",
        )
        model.addConstr(
            bounding_box_area
            - quicksum(
                x[room_index, i, j]
                for i, j in valid_coordinates
            )
            <= rect_diff_slack[room_index],
            name=f"rect_slack_def_{room_index}",
        )
    return rect_diff_slack


@dataclass(frozen=True)
class ConnectivityFlowArtifacts:
    variables: tuple
    constraints: tuple


def add_fixed_root_connectivity(
    model,
    active_by_coordinate,
    valid_coordinates,
    *,
    root_coordinate,
    name: str,
):
    """Connect active four-neighbor cells to one required active root."""
    coordinates = tuple(sorted(valid_coordinates))
    if not coordinates:
        raise ValueError("Connectivity requires at least one valid coordinate")
    if root_coordinate not in coordinates:
        raise ValueError("Connectivity root must be a valid coordinate")

    coordinate_set = set(coordinates)
    directed_edges = tuple(
        (i, j, neighbor_i, neighbor_j)
        for i, j in coordinates
        for neighbor_i, neighbor_j in (
            (i - 1, j),
            (i + 1, j),
            (i, j - 1),
            (i, j + 1),
        )
        if (neighbor_i, neighbor_j) in coordinate_set
    )
    capacity = len(coordinates)
    flow = model.addVars(
        directed_edges,
        lb=0,
        ub=capacity,
        vtype=GRB.CONTINUOUS,
        name=f"{name}_flow",
    )

    constraints = [
        model.addConstr(
            active_by_coordinate[root_coordinate] == 1,
            name=f"{name}_root_active",
        )
    ]
    total_active = quicksum(
        active_by_coordinate[coordinate]
        for coordinate in coordinates
    )

    for i, j in coordinates:
        active = active_by_coordinate[i, j]
        incoming = quicksum(
            flow[neighbor_i, neighbor_j, i, j]
            for neighbor_i, neighbor_j in (
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
            )
            if (neighbor_i, neighbor_j) in coordinate_set
        )
        outgoing = quicksum(
            flow[i, j, neighbor_i, neighbor_j]
            for neighbor_i, neighbor_j in (
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
            )
            if (neighbor_i, neighbor_j) in coordinate_set
        )
        balance = (
            outgoing - incoming == total_active - 1
            if (i, j) == root_coordinate
            else incoming - outgoing == active
        )
        constraints.append(
            model.addConstr(
                balance,
                name=f"{name}_balance_{i}_{j}",
            )
        )

    for i, j, neighbor_i, neighbor_j in directed_edges:
        constraints.append(
            model.addConstr(
                flow[i, j, neighbor_i, neighbor_j]
                <= capacity * active_by_coordinate[i, j],
                name=(
                    f"{name}_flow_from_active_"
                    f"{i}_{j}_{neighbor_i}_{neighbor_j}"
                ),
            )
        )
        constraints.append(
            model.addConstr(
                flow[i, j, neighbor_i, neighbor_j]
                <= capacity
                * active_by_coordinate[neighbor_i, neighbor_j],
                name=(
                    f"{name}_flow_to_active_"
                    f"{i}_{j}_{neighbor_i}_{neighbor_j}"
                ),
            )
        )

    return ConnectivityFlowArtifacts(
        variables=tuple(flow.values()),
        constraints=tuple(constraints),
    )


@dataclass(frozen=True)
class ConnectivityCutStats:
    rounds: int
    cuts_added: int
    elapsed_seconds: float


def run_connectivity_cut_loop(
    model,
    separator,
    callback=None,
    time_limit=None,
) -> ConnectivityCutStats:
    """Add violated connectivity cuts and reoptimize within one time budget."""
    started = time.monotonic()
    rounds = 0
    cuts_added = 0
    original_time_limit = model.Params.TimeLimit
    total_time_limit = original_time_limit if time_limit is None else time_limit
    accepted_statuses = {
        GRB.OPTIMAL,
        GRB.TIME_LIMIT,
        GRB.SUBOPTIMAL,
        GRB.INTERRUPTED,
    }

    try:
        while model.SolCount > 0:
            if model.Status not in accepted_statuses:
                raise ConnectivityOptimizationError(
                    f"Unsupported optimization status: {model.Status}"
                )

            cut_count = separator(rounds)
            if cut_count == 0:
                return ConnectivityCutStats(
                    rounds=rounds,
                    cuts_added=cuts_added,
                    elapsed_seconds=time.monotonic() - started,
                )

            cuts_added += cut_count
            remaining = total_time_limit - (time.monotonic() - started)
            if remaining <= 0:
                raise ConnectivityOptimizationError(
                    "Connectivity cut generation exceeded its time limit"
                )

            for variable in model.getVars():
                variable.Start = variable.X
            model.Params.TimeLimit = remaining
            (
                model.optimize(callback)
                if callback is not None
                else model.optimize()
            )
            rounds += 1

        raise ConnectivityOptimizationError(
            "No incumbent after connectivity cuts"
        )
    finally:
        model.Params.TimeLimit = original_time_limit


def configure_model_log(model, model_name: str, output_dir) -> None:
    """Configure a clean log file for a Gurobi model, written into ``output_dir``
    (typically the model's session-scoped optimization directory)."""
    log_file_path = os.path.join(output_dir, f"{model_name}.log")
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
    output_dir,
    callback=None,
) -> None:
    """Run one-stage or two-stage optimization with shared settings.

    ``output_dir`` is the model's session-scoped optimization directory,
    where the exported .lp file is written.
    """
    model.write(os.path.join(output_dir, f"{model_name}.lp"))

    def optimize_with_callback():
        if callback is None:
            model.optimize()
            return
        reset_callback = getattr(callback, "reset", None)
        if reset_callback is not None:
            reset_callback()
        model.optimize(callback)

    if stage_num == 1:
        model.setParam("MIPFocus", 2)
        model.setObjective(objective_function, GRB.MINIMIZE)
        optimize_with_callback()
        return

    if stage_num == 2:
        model.setParam("MIPFocus", 1)
        model.setObjective(0)  # find a feasible solution
        optimize_with_callback()
        if model.status == GRB.OPTIMAL:
            model.update()
            model.setParam("MIPFocus", 1)
            model.setObjective(objective_function, GRB.MINIMIZE)  # optimize the objective function
            optimize_with_callback()
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
    big_m_i: int,
    big_m_j: int,
    rect_min_i,
    rect_len_i,
    rect_min_j,
    rect_len_j,
) -> None:
    """Link room occupancy variables with room bounding-box variables."""
    for k in range(room_num):
        for i, j in valid_coordinates:
            model.addConstr(
                rect_min_i[k] <= i + big_m_i * (1 - x[k, i, j]),
                name=f"bbox_min_i_{k}_{i}_{j}",
            )
            model.addConstr(
                rect_min_i[k] + rect_len_i[k] - 1
                >= i - big_m_i * (1 - x[k, i, j]),
                name=f"bbox_max_i_{k}_{i}_{j}",
            )
            model.addConstr(
                rect_min_j[k] <= j + big_m_j * (1 - x[k, i, j]),
                name=f"bbox_min_j_{k}_{i}_{j}",
            )
            model.addConstr(
                rect_min_j[k] + rect_len_j[k] - 1
                >= j - big_m_j * (1 - x[k, i, j]),
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

    # Keep the legacy area bound for expressions (notably furniture-distance
    # relations) whose exact affine range depends on several furniture
    # dimensions and offsets.  Use the smaller, provably safe bounds below
    # for coordinate, binary-implication, and flow constraints.
    big_m = width * length
    big_m_i = max(1, width)
    big_m_j = max(1, length)
    big_m_binary = 1
    big_m_flow = max(1, len(valid_coordinates))
    if print_big_m:
        print("\n" + "-" * 25)
        print(
            "Set Big M bounds: "
            f"legacy={big_m}, i={big_m_i}, j={big_m_j}, "
            f"binary={big_m_binary}, flow={big_m_flow}"
        )
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
        "BigM_i": big_m_i,
        "BigM_j": big_m_j,
        "BigM_binary": big_m_binary,
        "BigM_flow": big_m_flow,
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
