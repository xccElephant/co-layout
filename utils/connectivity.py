from collections import deque
from dataclasses import dataclass

import numpy as np


_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class SeparatorCut:
    component: frozenset[tuple[int, int]]
    boundary: frozenset[tuple[int, int]]
    source: tuple[int, int]
    target: tuple[int, int]


def connected_components(
    active_coordinates,
    valid_coordinates,
) -> list[set[tuple[int, int]]]:
    remaining = set(active_coordinates) & set(valid_coordinates)
    components = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        component = {start}

        while queue:
            i, j = queue.popleft()
            for delta_i, delta_j in _NEIGHBOR_OFFSETS:
                neighbor = (i + delta_i, j + delta_j)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    components.sort(key=lambda component: (-len(component), min(component)))
    return components


def separator_cuts(
    components,
    valid_coordinates,
) -> list[SeparatorCut]:
    ordered_components = sorted(
        (set(component) for component in components),
        key=lambda component: (-len(component), min(component)),
    )
    if not ordered_components:
        return []

    valid = set(valid_coordinates)
    target = min(ordered_components[0])
    cuts = []

    for component in ordered_components[1:]:
        boundary = set()
        for i, j in component:
            for delta_i, delta_j in _NEIGHBOR_OFFSETS:
                neighbor = (i + delta_i, j + delta_j)
                if neighbor in valid and neighbor not in component:
                    boundary.add(neighbor)

        cuts.append(
            SeparatorCut(
                component=frozenset(component),
                boundary=frozenset(boundary),
                source=min(component),
                target=target,
            )
        )

    return cuts


class ConnectivityValidationError(AssertionError):
    """Raised when exported layout arrays violate connectivity requirements."""


def validate_layout_connectivity(
    room_grids,
    furniture_grids,
    room_names=None,
    closed_room_indices=None,
):
    """Validate room and closed-room free-space connectivity from arrays only."""
    room_grids = np.asarray(room_grids)
    if room_grids.ndim != 3:
        raise ConnectivityValidationError(
            "x_array must be three-dimensional, "
            f"got shape {room_grids.shape}"
        )

    room_count, width, length = room_grids.shape
    if room_names is None:
        room_names = [str(index) for index in range(room_count)]
    if len(room_names) != room_count:
        raise ConnectivityValidationError(
            "room_names length does not match x_array room dimension"
        )
    if len(furniture_grids) != room_count:
        raise ConnectivityValidationError(
            "furniture_array length does not match x_array room dimension"
        )

    valid_coordinates = {
        (i, j) for i in range(width) for j in range(length)
    }
    closed_room_indices = set(closed_room_indices or [])
    report = {
        "room_components": {},
        "free_space_components": {},
    }

    for room_index, room_name in enumerate(room_names):
        room_coordinates = {
            (int(i), int(j))
            for i, j in np.argwhere(room_grids[room_index] > 0.5)
        }
        room_component_count = len(
            connected_components(room_coordinates, valid_coordinates)
        )
        report["room_components"][room_name] = room_component_count
        if room_component_count != 1:
            raise ConnectivityValidationError(
                f"{room_name} room occupancy has "
                f"{room_component_count} components"
            )

        if room_index not in closed_room_indices:
            continue

        room_furniture = np.asarray(furniture_grids[room_index])
        if room_furniture.size == 0:
            occupied_grid = np.zeros((width, length))
        else:
            if room_furniture.ndim != 3:
                raise ConnectivityValidationError(
                    f"{room_name} furniture grid must be three-dimensional, "
                    f"got shape {room_furniture.shape}"
                )
            occupied_grid = room_furniture.sum(axis=0)
            if occupied_grid.shape != (width, length):
                raise ConnectivityValidationError(
                    f"{room_name} furniture grid shape "
                    f"{occupied_grid.shape} does not match room grid "
                    f"{(width, length)}"
                )
        free_coordinates = {
            (int(i), int(j))
            for i, j in np.argwhere(
                (room_grids[room_index] > 0.5)
                & (occupied_grid <= 0.5)
            )
        }
        free_component_count = len(
            connected_components(free_coordinates, valid_coordinates)
        )
        report["free_space_components"][room_name] = free_component_count
        if free_component_count != 1:
            raise ConnectivityValidationError(
                f"{room_name} free space has "
                f"{free_component_count} components"
            )

    return report


def validate_result_connectivity(
    result,
    room_names,
    closed_room_names,
):
    """Validate connectivity using only serialized result arrays."""
    room_name_to_index = {
        room_name: index for index, room_name in enumerate(room_names)
    }
    missing_closed_rooms = set(closed_room_names) - set(room_name_to_index)
    if missing_closed_rooms:
        raise ConnectivityValidationError(
            "closed room names not found in room_names: "
            f"{sorted(missing_closed_rooms)}"
        )
    return validate_layout_connectivity(
        result["x_array"],
        result["furniture_array"],
        room_names=room_names,
        closed_room_indices=[
            room_name_to_index[room_name]
            for room_name in closed_room_names
        ],
    )
