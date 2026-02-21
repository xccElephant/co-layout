import random
import numpy as np
from constants import *


def add_door(floorplan_model):
    x_array = floorplan_model.x_array
    passage_array = floorplan_model.passage_array
    furniture_array = floorplan_model.furniture_array
    furniture_num_list = floorplan_model.furniture_num_list
    open_room_indices = floorplan_model.open_room_indices
    entrance = floorplan_model.entrance
    num, width, length = x_array.shape
    door_size = round(DOOR_SIZE / F)
    door_array = []

    for k in range(num):
        if k in open_room_indices:
            continue
        first_candidate_doors = []  # Door adjacent to open room
        second_candidate_doors = []  # Door adjacent to passage
        for i, j in floorplan_model.valid_coordinates:
            open_room_idx_n = None
            open_room_idx_s = None
            open_room_idx_e = None
            open_room_idx_w = None
            count = np.zeros((4, 4))
            for ko in open_room_indices:
                if (
                    i - 1 >= 0
                    and x_array[ko, i - 1, j]
                    - sum(
                        furniture_array[ko][l, i - 1, j]
                        for l in range(furniture_num_list[ko])
                    )
                    > 0.5
                ):
                    open_room_idx_n = ko
                if (
                    i + 1 < width
                    and x_array[ko, i + 1, j]
                    - sum(
                        furniture_array[ko][l, i + 1, j]
                        for l in range(furniture_num_list[ko])
                    )
                    > 0.5
                ):
                    open_room_idx_s = ko
                if (
                    j - 1 >= 0
                    and x_array[ko, i, j - 1]
                    - sum(
                        furniture_array[ko][l, i, j - 1]
                        for l in range(furniture_num_list[ko])
                    )
                    > 0.5
                ):
                    open_room_idx_w = ko
                if (
                    j + 1 < length
                    and x_array[ko, i, j + 1]
                    - sum(
                        furniture_array[ko][l, i, j + 1]
                        for l in range(furniture_num_list[ko])
                    )
                    > 0.5
                ):
                    open_room_idx_e = ko
            for ds in range(door_size):
                # East
                if (
                    j + ds < length
                    and x_array[k, i, j + ds]
                    - sum(
                        furniture_array[k][l, i, j + ds]
                        for l in range(furniture_num_list[k])
                    )
                    > 0.5
                ):
                    if i - 1 >= 0 and passage_array[i - 1, j + ds] > 0.5:
                        count[0][0] += 1
                    if i + 1 < width and passage_array[i + 1, j + ds] > 0.5:
                        count[1][0] += 1
                    if (
                        open_room_idx_n is not None
                        and i - 1 >= 0
                        and x_array[open_room_idx_n, i - 1, j + ds]
                        - sum(
                            furniture_array[open_room_idx_n][l, i - 1, j + ds]
                            for l in range(furniture_num_list[open_room_idx_n])
                        )
                        > 0.5
                    ):
                        count[2][0] += 1
                    if (
                        open_room_idx_s is not None
                        and i + 1 < width
                        and x_array[open_room_idx_s, i + 1, j + ds]
                        - sum(
                            furniture_array[open_room_idx_s][l, i + 1, j + ds]
                            for l in range(furniture_num_list[open_room_idx_s])
                        )
                        > 0.5
                    ):
                        count[3][0] += 1
                # West
                if (
                    j - ds >= 0
                    and x_array[k, i, j - ds]
                    - sum(
                        furniture_array[k][l, i, j - ds]
                        for l in range(furniture_num_list[k])
                    )
                    > 0.5
                ):
                    if i - 1 >= 0 and passage_array[i - 1, j - ds] > 0.5:
                        count[0][1] += 1
                    if i + 1 < width and passage_array[i + 1, j - ds] > 0.5:
                        count[1][1] += 1
                    if (
                        open_room_idx_n is not None
                        and i - 1 >= 0
                        and x_array[open_room_idx_n, i - 1, j - ds]
                        - sum(
                            furniture_array[open_room_idx_n][l, i - 1, j - ds]
                            for l in range(furniture_num_list[open_room_idx_n])
                        )
                        > 0.5
                    ):
                        count[2][1] += 1
                    if (
                        open_room_idx_s is not None
                        and i + 1 < width
                        and x_array[open_room_idx_s, i + 1, j - ds]
                        - sum(
                            furniture_array[open_room_idx_s][l, i + 1, j - ds]
                            for l in range(furniture_num_list[open_room_idx_s])
                        )
                        > 0.5
                    ):
                        count[3][1] += 1
                # South
                if (
                    i + ds < width
                    and x_array[k, i + ds, j]
                    - sum(
                        furniture_array[k][l, i + ds, j]
                        for l in range(furniture_num_list[k])
                    )
                    > 0.5
                ):
                    if j - 1 >= 0 and passage_array[i + ds, j - 1] > 0.5:
                        count[0][2] += 1
                    if j + 1 < length and passage_array[i + ds, j + 1] > 0.5:
                        count[1][2] += 1
                    if (
                        open_room_idx_w is not None
                        and j - 1 >= 0
                        and x_array[open_room_idx_w, i + ds, j - 1]
                        - sum(
                            furniture_array[open_room_idx_w][l, i + ds, j - 1]
                            for l in range(furniture_num_list[open_room_idx_w])
                        )
                        > 0.5
                    ):
                        count[2][2] += 1
                    if (
                        open_room_idx_e is not None
                        and j + 1 < length
                        and x_array[open_room_idx_e, i + ds, j + 1]
                        - sum(
                            furniture_array[open_room_idx_e][l, i + ds, j + 1]
                            for l in range(furniture_num_list[open_room_idx_e])
                        )
                        > 0.5
                    ):
                        count[3][2] += 1
                # North
                if (
                    i - ds >= 0
                    and x_array[k, i - ds, j]
                    - sum(
                        furniture_array[k][l, i - ds, j]
                        for l in range(furniture_num_list[k])
                    )
                    > 0.5
                ):
                    if j - 1 >= 0 and passage_array[i - ds, j - 1] > 0.5:
                        count[0][3] += 1
                    if j + 1 < length and passage_array[i - ds, j + 1] > 0.5:
                        count[1][3] += 1
                    if (
                        open_room_idx_w is not None
                        and j - 1 >= 0
                        and x_array[open_room_idx_w, i - ds, j - 1]
                        - sum(
                            furniture_array[open_room_idx_w][l, i - ds, j - 1]
                            for l in range(furniture_num_list[open_room_idx_w])
                        )
                        > 0.5
                    ):
                        count[2][3] += 1
                    if (
                        open_room_idx_e is not None
                        and j + 1 < length
                        and x_array[open_room_idx_e, i - ds, j + 1]
                        - sum(
                            furniture_array[open_room_idx_e][l, i - ds, j + 1]
                            for l in range(furniture_num_list[open_room_idx_e])
                        )
                        > 0.5
                    ):
                        count[3][3] += 1

            rows, cols = np.where(count == door_size)
            for row, col in zip(rows, cols):
                door_dirc = col
                if col <= 1:
                    if row == 0:
                        second_candidate_doors.append((door_dirc, NORTH, i, j))
                    elif row == 1:
                        second_candidate_doors.append((door_dirc, SOUTH, i, j))
                    elif row == 2:
                        first_candidate_doors.append((door_dirc, NORTH, i, j))
                    elif row == 3:
                        first_candidate_doors.append((door_dirc, SOUTH, i, j))
                else:
                    if row == 0:
                        second_candidate_doors.append((door_dirc, WEST, i, j))
                    elif row == 1:
                        second_candidate_doors.append((door_dirc, EAST, i, j))
                    elif row == 2:
                        first_candidate_doors.append((door_dirc, WEST, i, j))
                    elif row == 3:
                        first_candidate_doors.append((door_dirc, EAST, i, j))

        if first_candidate_doors:
            door_direction, door_face, door_i, door_j = random.choice(
                first_candidate_doors
            )
            door_array.append((door_direction, door_face, door_i, door_j))
        elif second_candidate_doors:
            door_direction, door_face, door_i, door_j = random.choice(
                second_candidate_doors
            )
            door_array.append((door_direction, door_face, door_i, door_j))

    # Add entrance door
    if entrance[0] <= 1:
        door_array.append(
            (SOUTH, entrance[0], entrance[1] - door_size // 2, entrance[2])
        )
    else:
        door_array.append(
            (EAST, entrance[0], entrance[1], entrance[2] - door_size // 2)
        )

    return door_array


def add_wall(floorplan_model):
    x_array = floorplan_model.x_array
    num, width, length = x_array.shape
    wall_array = np.zeros((4, width, length), dtype=np.int8)
    open_room_indices = floorplan_model.open_room_indices
    obstacles = set(floorplan_model.outdoor_space_coordinates)

    for i in range(width):
        for j in range(length):
            if (i, j) not in obstacles:
                # East
                if j == length - 1 or (i, j + 1) in obstacles:
                    wall_array[EAST, i, j] = 1
                # West
                if j == 0 or (i, j - 1) in obstacles:
                    wall_array[WEST, i, j] = 1
                # South
                if i == width - 1 or (i + 1, j) in obstacles:
                    wall_array[SOUTH, i, j] = 1
                # North
                if i == 0 or (i - 1, j) in obstacles:
                    wall_array[NORTH, i, j] = 1

    for k in range(num):
        if k in open_room_indices:
            continue
        for i in range(width):
            for j in range(length):
                if x_array[k, i, j] < 0.5:
                    continue

                # Be careful to avoid double counting
                # East Neighbor (i, j+1)
                if j < length - 1 and x_array[k, i, j + 1] < 0.5:
                    wall_array[EAST, i, j] = (
                        1 if wall_array[WEST, i, j + 1] < 0.5 else 0
                    )
                # West Neighbor (i, j-1)
                if j > 0 and x_array[k, i, j - 1] < 0.5:
                    wall_array[WEST, i, j] = (
                        1 if wall_array[EAST, i, j - 1] < 0.5 else 0
                    )
                # South Neighbor (i+1, j)
                if i < width - 1 and x_array[k, i + 1, j] < 0.5:
                    wall_array[SOUTH, i, j] = (
                        1 if wall_array[NORTH, i + 1, j] < 0.5 else 0
                    )
                # North Neighbor (i-1, j)
                if i > 0 and x_array[k, i - 1, j] < 0.5:
                    wall_array[NORTH, i, j] = (
                        1 if wall_array[SOUTH, i - 1, j] < 0.5 else 0
                    )

    return wall_array


def add_window(floorplan_model):
    x_array = floorplan_model.x_array
    passage_array = floorplan_model.passage_array
    furniture_array = floorplan_model.furniture_array
    furniture_num_list = floorplan_model.furniture_num_list
    boundary = floorplan_model.boundary
    valid_coordinates = floorplan_model.valid_coordinates
    num, _, _ = x_array.shape
    window_size = round(WINDOW_SIZE / F)
    window_array = []

    for k in range(num):
        candidate_windows = []
        for i, j in boundary:
            count = [0, 0, 0, 0]
            if (
                x_array[k, i, j]
                - sum(furniture_array[k][l, i, j] for l in range(furniture_num_list[k]))
                < 0.5
            ):
                continue
            for ws in range(window_size):
                # East
                if (i, j + ws) in boundary and x_array[k, i, j + ws] - sum(
                    furniture_array[k][l, i, j + ws]
                    for l in range(furniture_num_list[k])
                ) > 0.5:
                    count[0] += 1
                # West
                if (i, j - ws) in boundary and x_array[k, i, j - ws] - sum(
                    furniture_array[k][l, i, j - ws]
                    for l in range(furniture_num_list[k])
                ) > 0.5:
                    count[1] += 1
                # South
                if (i + ws, j) in boundary and x_array[k, i + ws, j] - sum(
                    furniture_array[k][l, i + ws, j]
                    for l in range(furniture_num_list[k])
                ) > 0.5:
                    count[2] += 1
                # North
                if (i - ws, j) in boundary and x_array[k, i - ws, j] - sum(
                    furniture_array[k][l, i - ws, j]
                    for l in range(furniture_num_list[k])
                ) > 0.5:
                    count[3] += 1
            rows = [index for index in range(4) if count[index] == window_size]
            for idx in rows:
                window_face = None
                if idx <= 1:
                    if (i + 1, j) not in valid_coordinates:
                        window_face = SOUTH
                    if (i - 1, j) not in valid_coordinates:
                        window_face = NORTH
                else:
                    if (i, j + 1) not in valid_coordinates:
                        window_face = EAST
                    if (i, j - 1) not in valid_coordinates:
                        window_face = WEST
                if window_face is not None and (i, j) != (
                    floorplan_model.entrance[1],
                    floorplan_model.entrance[2],
                ):
                    candidate_windows.append((idx, window_face, i, j))

        if candidate_windows and len(candidate_windows) <= 6:
            sample_window1 = random.choice(candidate_windows)
            window_array.append(sample_window1)
        if (
            len(candidate_windows) > 6
            and floorplan_model.room_area_list[k] >= 12 / F / F
        ):
            sample_window1 = random.choice(candidate_windows)
            window_array.append(sample_window1)
            for win in candidate_windows:
                if (
                    win[0] != sample_window1[0]
                    or abs(win[2] - sample_window1[2]) > 2
                    or abs(win[3] - sample_window1[3]) > 2
                ):
                    window_array.append(win)
                    break

        # if candidate_windows:
        #     # Assuming there's only one window.
        #     window_dirc, window_face, window_i, window_j = random.choice(candidate_windows)
        #     window_array.append((window_dirc, window_face, window_i, window_j))
    # Add passage windows
    candidate_windows = []
    for i, j in boundary:
        count = [0, 0, 0, 0]
        if passage_array[i, j] < 0.5:
            continue
        for ws in range(window_size):
            # East
            if (i, j + ws) in boundary and passage_array[i, j + ws] > 0.5:
                count[0] += 1
            # West
            if (i, j - ws) in boundary and passage_array[i, j - ws] > 0.5:
                count[1] += 1
            # South
            if (i + ws, j) in boundary and passage_array[i + ws, j] > 0.5:
                count[2] += 1
            # North
            if (i - ws, j) in boundary and passage_array[i - ws, j] > 0.5:
                count[3] += 1
        rows = [index for index in range(4) if count[index] == window_size]
        for idx in rows:
            window_face = None
            if idx <= 1:
                if (i + 1, j) not in valid_coordinates:
                    window_face = SOUTH
                if (i - 1, j) not in valid_coordinates:
                    window_face = NORTH
            else:
                if (i, j + 1) not in valid_coordinates:
                    window_face = EAST
                if (i, j - 1) not in valid_coordinates:
                    window_face = WEST
            if window_face is not None and (abs(i - floorplan_model.entrance[1]) >= window_size or abs(j - floorplan_model.entrance[2]) >= window_size):
                candidate_windows.append((idx, window_face, i, j))
    if candidate_windows and len(candidate_windows) <= 6:
        sample_window1 = random.choice(candidate_windows)
        window_array.append(sample_window1)
    if len(candidate_windows) > 6:
        sample_window1 = random.choice(candidate_windows)
        window_array.append(sample_window1)
        for win in candidate_windows:
            if win[0] != sample_window1[0] or abs(win[2] - sample_window1[2]) > 2 or abs(win[3] - sample_window1[3]) > 2:
                window_array.append(win)
                break
    return window_array


def rearrange(floorplan_model, room, furniture):
    x_array = floorplan_model.x_array
    furniture_array = floorplan_model.furniture_array
    furniture_parallel_size = floorplan_model.furniture_parallel_size
    furniture_vertical_size = floorplan_model.furniture_vertical_size
    furniture_num_list = floorplan_model.furniture_num_list
    _, width, length = x_array.shape
    door_size = round(DOOR_SIZE / F)
    door_array = floorplan_model.door_array
    p = furniture_parallel_size[room][furniture]
    v = furniture_vertical_size[room][furniture]

    room_region = x_array[room] > 0.5
    door_region = np.zeros((width, length))
    door = None
    for dirc, facing, di, dj in door_array:
        if x_array[room, di, dj] > 0.5:
            door = (dirc, facing, di, dj)
            break
    if door is not None:
        i_dir = 1
        j_dir = 1
        dirc, facing, di, dj = door
        if dirc == 1 or facing == 0:
            j_dir = -1
        if dirc == 3 or facing == 2:
            i_dir = -1
        for si in range(door_size):
            for sj in range(door_size):
                door_region[di + i_dir * si, dj + j_dir * sj] = 1
    door_region = door_region > 0.5
    mask = np.arange(furniture_num_list[room]) != furniture
    other_sum = np.sum(furniture_array[room][mask], axis=0)
    furniture_region = other_sum > 0.5

    valid_region = room_region & ~furniture_region & ~door_region

    rows, cols = np.where(valid_region)
    candidates = []
    for i, j in zip(rows, cols):
        if (
            i + v < width
            and j + p < length
            and np.all(valid_region[i : i + v, j : j + p])
        ):
            candidates.append((0, i, j))
        if (
            i + p < width
            and j + v < length
            and np.all(valid_region[i : i + p, j : j + v])
        ):
            candidates.append((1, i, j))

    if candidates:
        sigma, pos_i, pos_j = random.choice(candidates)
        floorplan_model.furniture_orientation_sigma_array[room][furniture] = sigma
        floorplan_model.f_rect_min_i = pos_i
        floorplan_model.f_rect_min_j = pos_j
        floorplan_model.furniture_array[room][furniture, :, :] = 0
        if sigma == 0:
            floorplan_model.furniture_array[room][
                furniture, pos_i : pos_i + v, pos_j : pos_j + p
            ] = 1
        else:
            floorplan_model.furniture_array[room][
                furniture, pos_i : pos_i + p, pos_j : pos_j + v
            ] = 1
