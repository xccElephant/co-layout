import datetime
import itertools
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from constants import *

cad_background_color = "white"
cad_wall_color = "black"


def visualize_floorplan_cadstyle(
    x_array,
    passage_array,
    outdoor_space_array=None,
    wall_array=None,
    save_path="floorplan_cad_visualization.svg",
):
    num_rooms, width, length = x_array.shape
    room_matrix = np.full((width, length), -1)
    cad_hatch_color = "black"

    for i in range(width):
        for j in range(length):
            if passage_array[i, j] > 0.5:
                room_matrix[i, j] = -2

    for i in range(width):
        for j in range(length):
            if room_matrix[i, j] == -1:
                for k in range(num_rooms):
                    if x_array[k, i, j] > 0.5:
                        room_matrix[i, j] = k
                        break

    fig, ax = plt.subplots(figsize=(length / 1.5, width / 1.5))
    fig.patch.set_facecolor(cad_background_color)
    ax.set_facecolor(cad_background_color)

    # Draw room background (uniform color)
    # -2 represents public passages, also using background color
    # -1 represents outdoor/undefined areas, also using background color
    # Other numbers represent different rooms, uniformly using background color
    cmap = ListedColormap([cad_background_color])
    ax.imshow(room_matrix, cmap=cmap, interpolation="nearest")

    ax.axis("off")

    # Cell borders
    for i in range(0, width):
        ax.plot(
            [-0.5, length - 0.5],  # x ranges from -0.5 to length - 0.5
            [i - 0.5, i - 0.5],  # The y coordinate is fixed to i - 0.5
            color="gray",
            linewidth=0.5,
            linestyle=(0, (3, 7)),
        )
    for j in range(0, length):
        ax.plot(
            [j - 0.5, j - 0.5],  # The x coordinate is fixed to j - 0.5
            [-0.5, width - 0.5],  # The y range from -0.5 to width - 0.5
            color="gray",
            linewidth=0.5,
            linestyle=(0, (3, 7)),
        )

    # Set plot limits to match the grid extent
    ax.set_xlim(-1, length)
    ax.set_ylim(width, -1)  # Inverted y-axis for image coordinates

    # visualize obstacle
    if outdoor_space_array is not None:
        for i in range(width):
            for j in range(length):
                if outdoor_space_array[i, j] > 0.5:
                    ax.fill_between(
                        [j - 0.5, j + 0.5],
                        i - 0.5,
                        i + 0.5,
                        facecolor=cad_background_color,
                        hatch="X",
                        edgecolor=cad_hatch_color,
                        linewidth=0.5,
                    )

    if wall_array is not None:
        _visualize_walls(wall_array, width, length, ax, cad_wall_color)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"SVG image saved to {os.path.abspath(save_path)}")


def visualize_floorplan_with_furniture_cadstyle(
    x_array,
    passage_array,
    furniture_array,
    furniture_orientation_sigma_array,
    furniture_orientation_mu_array,
    f_rect_min_i_array,
    f_rect_min_j_array,
    outdoor_space_array=None,
    wall_array=None,
    door_array=None,
    window_array=None,
    furniture_info=None,
    furniture_constraints=None,
    save_path="floorplan_furniture_visualization.svg",
):
    # Loading a saved solution
    num_rooms, width, length = x_array.shape
    num_furnitures = []
    for k in range(num_rooms):
        num_furnitures.append(furniture_array[k].shape[0])

    # Create a room allocation matrix (each grid stores the room number)
    room_matrix = np.full((width, length), -1)  # -1 means unassigned

    # CAD style color definitions
    cad_furniture_fill_color = "white"
    cad_furniture_edge_color = "black"
    cad_text_color = "black"
    cad_hatch_color = "black"

    room_name_list = list(furniture_info.keys())
    furniture_name_list = []
    for room_name in room_name_list:
        f_list = []
        for f in furniture_info[room_name]:
            f_list.append(f["name"])
        furniture_name_list.append(f_list)

    for i in range(width):
        for j in range(length):
            if passage_array[i, j] > 0.5:
                room_matrix[i, j] = -2

    for i in range(width):
        for j in range(length):
            if room_matrix[i, j] == -1:
                for k in range(num_rooms):
                    if x_array[k, i, j] > 0.5:
                        room_matrix[i, j] = k
                        break

    # Create visualization
    fig, ax = plt.subplots(
        figsize=(length / 1.5, width / 1.5)
    )  # Adjust image size based on floor plan
    fig.patch.set_facecolor(cad_background_color)
    ax.set_facecolor(cad_background_color)

    # Draw room backgrounds (uniform color)
    # -2 represents public passages, also using background color
    # -1 represents outdoor/undefined areas, also using background color
    # Other numbers represent different rooms, all using background color
    cmap = ListedColormap([cad_background_color])
    ax.imshow(room_matrix, cmap=cmap, interpolation="nearest")

    # Set axis
    ax.axis("off")

    # Cell boundaries
    for i in range(0, width):
        ax.plot(
            [-0.5, length - 0.5],  # x range from -0.5 to length - 0.5
            [i - 0.5, i - 0.5],  # y coordinate fixed at i - 0.5
            color="gray",
            linewidth=0.5,
            linestyle=(0, (3, 7)),
        )
    for j in range(0, length):
        ax.plot(
            [j - 0.5, j - 0.5],  # x coordinate fixed at j - 0.5
            [-0.5, width - 0.5],  # y range from -0.5 to width - 0.5
            color="gray",
            linewidth=0.5,
            linestyle=(0, (3, 7)),
        )

    # Set plot limits to match the grid extent
    ax.set_xlim(-1, length)
    ax.set_ylim(width, -1)  # Inverted y-axis for image coordinates

    # Visualize obstacle
    if outdoor_space_array is not None:
        for i in range(width):
            for j in range(length):
                if outdoor_space_array[i, j] > 0.5:
                    ax.fill_between(
                        [j - 0.5, j + 0.5],
                        i - 0.5,
                        i + 0.5,
                        facecolor=cad_background_color,
                        hatch="X",
                        edgecolor=cad_hatch_color,
                        linewidth=0.5,
                    )

    # Draw furnitures
    furniture_parallel_size = []
    furniture_vertical_size = []
    furniture_original_parallel_size = []
    furniture_original_vertical_size = []
    for k in range(num_rooms):
        p_list = []
        v_list = []
        p_original_list = []
        v_original_list = []
        room_name = room_name_list[k]
        for l in range(num_furnitures[k]):
            p_list.append(furniture_info[room_name][l]["width"])
            v_list.append(furniture_info[room_name][l]["length"])
            p_original_list.append(furniture_info[room_name][l]["original_width"])
            v_original_list.append(furniture_info[room_name][l]["original_length"])
        furniture_parallel_size.append(p_list)
        furniture_vertical_size.append(v_list)
        furniture_original_parallel_size.append(p_original_list)
        furniture_original_vertical_size.append(v_original_list)
    drawn_furniture_rects = []  # List to store rectangles of drawn furniture
    for k in range(num_rooms):
        furniture_rect_list = []
        room_name = room_name_list[k]
        for l in range(num_furnitures[k]):
            if furniture_orientation_sigma_array[k][l] > 0.5:
                # Parallel to j-axis
                i_range = furniture_parallel_size[k][l]
                j_range = furniture_vertical_size[k][l]
                i_size = furniture_original_parallel_size[k][l]
                j_size = furniture_original_vertical_size[k][l]
            else:
                # Parallel to i-axis
                i_range = furniture_vertical_size[k][l]
                j_range = furniture_parallel_size[k][l]
                i_size = furniture_original_vertical_size[k][l]
                j_size = furniture_original_parallel_size[k][l]

            i_center = f_rect_min_i_array[k][l] + (i_range - 1) / 2
            j_center = f_rect_min_j_array[k][l] + (j_range - 1) / 2

            north_boundary = False
            south_boundary = False
            west_boundary = False
            east_boundary = False
            for i in range(width):
                for j in range(length):
                    if furniture_array[k][l, i, j] > 0.5:
                        if i == 0 or x_array[k, i - 1, j] < 0.5:
                            north_boundary = True
                        if i == width - 1 or x_array[k, i + 1, j] < 0.5:
                            south_boundary = True
                        if j == 0 or x_array[k, i, j - 1] < 0.5:
                            west_boundary = True
                        if j == length - 1 or x_array[k, i, j + 1] < 0.5:
                            east_boundary = True

            if (
                furniture_info[room_name][l]["name"]
                in furniture_constraints[room_name]["boundary_items"]
            ):
                if (
                    furniture_orientation_sigma_array[k][l] < 0.5
                    and furniture_orientation_mu_array[k][l] < 0.5
                ):
                    if west_boundary:
                        j_center -= (j_range - j_size) / 2
                        west_boundary = False
                    elif north_boundary:
                        i_center -= (i_range - i_size) / 2
                        north_boundary = False
                    elif south_boundary:
                        i_center += (i_range - i_size) / 2
                        south_boundary = False
                elif (
                    furniture_orientation_sigma_array[k][l] < 0.5
                    and furniture_orientation_mu_array[k][l] > 0.5
                ):
                    if east_boundary:
                        j_center += (j_range - j_size) / 2
                        east_boundary = False
                    elif north_boundary:
                        i_center -= (i_range - i_size) / 2
                        north_boundary = False
                    elif south_boundary:
                        i_center += (i_range - i_size) / 2
                        south_boundary = False
                elif (
                    furniture_orientation_sigma_array[k][l] > 0.5
                    and furniture_orientation_mu_array[k][l] < 0.5
                ):
                    if north_boundary:
                        i_center -= (i_range - i_size) / 2
                        north_boundary = False
                    elif west_boundary:
                        j_center -= (j_range - j_size) / 2
                        west_boundary = False
                    elif east_boundary:
                        j_center += (j_range - j_size) / 2
                        east_boundary = False
                elif (
                    furniture_orientation_sigma_array[k][l] > 0.5
                    and furniture_orientation_mu_array[k][l] > 0.5
                ):
                    if south_boundary:
                        i_center += (i_range - i_size) / 2
                        south_boundary = False
                    elif west_boundary:
                        j_center -= (j_range - j_size) / 2
                        west_boundary = False
                    elif east_boundary:
                        j_center += (j_range - j_size) / 2
                        east_boundary = False
            if (
                j_size >= 1
                or i_size >= 1
                or (i_size > 0 and j_size / i_size >= 2)
                or (j_size > 0 and i_size / j_size >= 2)
            ):
                if north_boundary and not south_boundary:
                    i_center -= (i_range - i_size) / 2
                if south_boundary and not north_boundary:
                    i_center += (i_range - i_size) / 2
                if west_boundary and not east_boundary:
                    j_center -= (j_range - j_size) / 2
                if east_boundary and not west_boundary:
                    j_center += (j_range - j_size) / 2
            if north_boundary and south_boundary and i_size > i_range:
                i_size = i_range
            if west_boundary and east_boundary and j_size > j_range:
                j_size = j_range

            # Define the current furniture's rectangle
            current_j_min = j_center - j_size / 2
            current_i_min = i_center - i_size / 2
            current_j_max = j_center + j_size / 2
            current_i_max = i_center + i_size / 2
            furniture_rect_list.append(
                [current_j_min, current_i_min, current_j_max, current_i_max]
            )
        drawn_furniture_rects.append(furniture_rect_list)

    for k in range(num_rooms):
        for l1, l2 in itertools.combinations(range(num_furnitures[k]), 2):
            rect1 = drawn_furniture_rects[k][l1]
            rect2 = drawn_furniture_rects[k][l2]
            j1_min, i1_min, j1_max, i1_max = rect1
            j2_min, i2_min, j2_max, i2_max = rect2

            overlap_j_min = max(j1_min, j2_min)
            overlap_i_min = max(i1_min, i2_min)
            overlap_j_max = min(j1_max, j2_max)
            overlap_i_max = min(i1_max, i2_max)

            # Check for overlapping
            if (overlap_j_max > overlap_j_min) and (overlap_i_max > overlap_i_min):
                overlap_j = overlap_j_max - overlap_j_min
                overlap_i = overlap_i_max - overlap_i_min

                if overlap_j > overlap_i:
                    if i1_min < i2_min:
                        drawn_furniture_rects[k][l1][3] -= overlap_i / 2
                        drawn_furniture_rects[k][l2][1] += overlap_i / 2
                    else:
                        drawn_furniture_rects[k][l2][3] -= overlap_i / 2
                        drawn_furniture_rects[k][l1][1] += overlap_i / 2
                else:
                    if j1_min < j2_min:
                        drawn_furniture_rects[k][l1][2] -= overlap_j / 2
                        drawn_furniture_rects[k][l2][0] += overlap_j / 2
                    else:
                        drawn_furniture_rects[k][l2][2] -= overlap_j / 2
                        drawn_furniture_rects[k][l1][0] += overlap_j / 2
    # Draw furniture
    furniture_info_ = []
    for k in range(num_rooms):
        info_list = []
        room_name = room_name_list[k]
        for l in range(num_furnitures[k]):
            # Draw the furniture rectangle
            j_min, i_min, j_max, i_max = drawn_furniture_rects[k][l]
            j_size = j_max - j_min
            i_size = i_max - i_min
            j_center = (j_min + j_max) / 2
            i_center = (i_min + i_max) / 2
            info_list.append(
                {
                    "name": furniture_name_list[k][l],
                    "room_idx(k)": k,
                    "furniture_idx(l)": l,
                    "j_min": j_min,
                    "i_min": i_min,
                    "j_max": j_max,
                    "i_max": i_max,
                    "j_size": j_size,
                    "i_size": i_size,
                    "j_center": j_center,
                    "i_center": i_center,
                    "height": furniture_info[room_name][l]["height"],
                }
            )
            rect = mpatches.Rectangle(
                (j_center - j_size / 2, i_center - i_size / 2),
                j_size,
                i_size,
                facecolor=cad_furniture_fill_color,
                edgecolor=cad_furniture_edge_color,
                linewidth=1.5,
            )
            ax.add_patch(rect)
        furniture_info_.append(info_list)

    # Generate a unique filename using the current timestamp
    # save to json
    doors = []
    windows = []
    for d in door_array:
        dirc, facing, di, dj = d
        info = {}
        info["direction"] = int(dirc)
        info["facing"] = int(facing)
        info["start_i"] = int(di)
        info["start_j"] = int(dj)
        doors.append(info)
    for w in window_array:
        dirc, facing, wi, wj = w
        info = {}
        info["direction"] = int(dirc)
        info["facing"] = int(facing)
        info["start_i"] = int(wi)
        info["start_j"] = int(wj)
        windows.append(info)
    result = {
        "x_array": x_array.tolist(),
        "passage_array": passage_array.tolist(),
        "wall_array": wall_array.tolist(),
        "outdoor_array": outdoor_space_array.tolist(),
        "furniture_array": [f.tolist() for f in furniture_array],
        "furniture_orientation_sigma_array": [
            f.tolist() for f in furniture_orientation_sigma_array
        ],
        "furniture_orientation_mu_array": [
            f.tolist() for f in furniture_orientation_mu_array
        ],
        "f_rect_min_i_array": [f.tolist() for f in f_rect_min_i_array],
        "f_rect_min_j_array": [f.tolist() for f in f_rect_min_j_array],
        "door_array": doors,
        "window_array": windows,
        "furniture_info": furniture_info_,
    }
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = f"{timestamp}.json"
    results_path = os.path.join(PATH_OF_OPTIMIZATION_RESULTS, results_filename)

    with open(
        results_path,
        "w",
    ) as f:
        json.dump(result, f, indent=4)
        print(f"Results saved to {f.name}")

    if wall_array is not None:
        _visualize_walls(wall_array, width, length, ax, cad_wall_color)

    if door_array is not None:
        _visualize_doors(door_array, ax)

    if window_array is not None:
        _visualize_windows(window_array, ax)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"SVG image saved to {os.path.abspath(save_path)}")


def _visualize_walls(wall_array, width, length, ax, wall_color):
    wall_line_width = 5
    for d in range(4):
        for i in range(width):
            for j in range(length):
                if wall_array[d, i, j] > 0.5:
                    if d == 0:
                        ax.plot(
                            [j + 0.5, j + 0.5],
                            [i - 0.5, i + 0.5],
                            color=wall_color,
                            linewidth=wall_line_width,
                        )
                    elif d == 1:
                        if j == 0 or wall_array[0, i, j - 1] < 0.5:
                            ax.plot(
                                [j - 0.5, j - 0.5],
                                [i - 0.5, i + 0.5],
                                color=wall_color,
                                linewidth=wall_line_width,
                            )
                    elif d == 2:
                        ax.plot(
                            [j - 0.5, j + 0.5],
                            [i + 0.5, i + 0.5],
                            color=wall_color,
                            linewidth=wall_line_width,
                        )
                    elif d == 3:
                        if i == 0 or wall_array[2, i - 1, j] < 0.5:
                            ax.plot(
                                [j - 0.5, j + 0.5],
                                [i - 0.5, i - 0.5],
                                color=wall_color,
                                linewidth=wall_line_width,
                            )


def _visualize_doors(door_array, ax):
    door_radius = round(DOOR_SIZE / F) - 0.2
    door_linewidth = 1.5
    door_size = round(DOOR_SIZE / F)
    for door in door_array:
        door_dirc, door_face, door_i, door_j = door
        if door_face == EAST:
            center = (door_j + 0.5, door_i - 0.4)
            if door_dirc == SOUTH:
                ax.plot(
                    [door_j + 0.5, door_j + 0.5],
                    [door_i - 0.4, door_i + door_size - 0.6],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            else:
                ax.plot(
                    [door_j + 0.5, door_j + 0.5],
                    [door_i - door_size + 0.6, door_i + 0.4],
                    color=cad_background_color,
                    linewidth=4.5,
                )
                center = (door_j + 0.5, door_i - door_size + 0.6)

            arc = mpatches.Arc(
                center,
                door_radius * 2,
                door_radius * 2,
                angle=0,
                theta1=90,
                theta2=180,
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
            ax.add_patch(arc)
            ax.plot(
                [center[0], center[0] - door_radius],
                [center[1], center[1]],
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
        elif door_face == WEST:
            center = (door_j - 0.5, door_i + 0.4)
            if door_dirc == SOUTH:
                center = (door_j - 0.5, door_i + door_size - 0.6)
                ax.plot(
                    [door_j - 0.5, door_j - 0.5],
                    [door_i - 0.4, door_i + door_size - 0.6],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            else:
                ax.plot(
                    [door_j - 0.5, door_j - 0.5],
                    [door_i - door_size + 0.6, door_i + 0.4],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            arc = mpatches.Arc(
                center,
                door_radius * 2,
                door_radius * 2,
                angle=0,
                theta1=270,
                theta2=360,
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
            ax.add_patch(arc)
            ax.plot(
                [center[0], center[0] + door_radius],
                [center[1], center[1]],
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
        elif door_face == SOUTH:
            center = (door_j + 0.4, door_i + 0.5)
            if door_dirc == EAST:
                center = (door_j + door_size - 0.6, door_i + 0.5)
                ax.plot(
                    [door_j - 0.4, door_j + door_size - 0.6],
                    [door_i + 0.5, door_i + 0.5],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            else:
                ax.plot(
                    [door_j - door_size + 0.6, door_j + 0.4],
                    [door_i + 0.5, door_i + 0.5],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            arc = mpatches.Arc(
                center,
                door_radius * 2,
                door_radius * 2,
                angle=0,
                theta1=180,
                theta2=270,
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
            ax.add_patch(arc)
            ax.plot(
                [center[0], center[0]],
                [center[1], center[1] - door_radius],
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
        elif door_face == NORTH:
            center = (door_j + 0.4, door_i - 0.5)
            if door_dirc == EAST:
                center = (door_j + door_size - 0.6, door_i - 0.5)
                ax.plot(
                    [door_j - 0.4, door_j + door_size - 0.6],
                    [door_i - 0.5, door_i - 0.5],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            else:
                ax.plot(
                    [door_j - door_size + 0.6, door_j + 0.4],
                    [door_i - 0.5, door_i - 0.5],
                    color=cad_background_color,
                    linewidth=4.5,
                )
            arc = mpatches.Arc(
                center,
                door_radius * 2,
                door_radius * 2,
                angle=0,
                theta1=90,
                theta2=180,
                color=cad_wall_color,
                linewidth=door_linewidth,
            )
            ax.add_patch(arc)
            ax.plot(
                [center[0], center[0]],
                [center[1], center[1] + door_radius],
                color=cad_wall_color,
                linewidth=door_linewidth,
            )


def _visualize_windows(window_array, ax):
    window_size = round(WINDOW_SIZE / F)
    for window in window_array:
        window_dirc, window_face, window_i, window_j = window
        if window_face == EAST:
            if window_dirc == SOUTH:
                ax.plot(
                    [window_j + 0.5, window_j + 0.5],
                    [window_i - 0.4, window_i + window_size - 0.6],
                    color=cad_background_color,
                    linewidth=4,
                )
            else:
                ax.plot(
                    [window_j + 0.5, window_j + 0.5],
                    [window_i - window_size + 0.6, window_i + 0.4],
                    color=cad_background_color,
                    linewidth=4,
                )
        elif window_face == WEST:
            if window_dirc == SOUTH:
                ax.plot(
                    [window_j - 0.5, window_j - 0.5],
                    [window_i - 0.4, window_i + window_size - 0.6],
                    color=cad_background_color,
                    linewidth=4,
                )
            else:
                ax.plot(
                    [window_j - 0.5, window_j - 0.5],
                    [window_i - window_size + 0.6, window_i + 0.4],
                    color=cad_background_color,
                    linewidth=4,
                )
        elif window_face == SOUTH:
            if window_dirc == EAST:
                ax.plot(
                    [window_j - 0.4, window_j + window_size - 0.6],
                    [window_i + 0.5, window_i + 0.5],
                    color=cad_background_color,
                    linewidth=4,
                )
            else:
                ax.plot(
                    [window_j - window_size + 0.6, window_j + 0.4],
                    [window_i + 0.5, window_i + 0.5],
                    color=cad_background_color,
                    linewidth=4,
                )
        elif window_face == NORTH:
            if window_dirc == EAST:
                ax.plot(
                    [window_j - 0.4, window_j + window_size - 0.6],
                    [window_i - 0.5, window_i - 0.5],
                    color=cad_background_color,
                    linewidth=4,
                )
            else:
                ax.plot(
                    [window_j - window_size + 0.6, window_j + 0.4],
                    [window_i - 0.5, window_i - 0.5],
                    color=cad_background_color,
                    linewidth=4,
                )
