import numpy as np
import textwrap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from constants import *
import os

from utils.furniture_layout_geometry import (
    compute_furniture_draw_rects,
    warn_on_furniture_overlaps,
)


def visualize_floorplan(
    x_array,
    passage_array=None,
    outdoor_space_array=None,
    wall_array=None,
    room_name=None,
    save_path="floorplan_visualization.png",
):
    # Loading a saved solution
    num_rooms, width, length = x_array.shape

    # Create a room allocation matrix (each grid stores the room number)
    room_matrix = np.full((width, length), -1)  # -1 means unassigned

    # If passage_array is provided, mark the passage area first
    if passage_array is not None:
        for i in range(width):
            for j in range(length):
                if passage_array[i, j] > 0.5:
                    room_matrix[i, j] = -1  # Passage area marked as -1

    # Mark the room area
    for i in range(width):
        for j in range(length):
            for k in range(num_rooms):
                if x_array[k, i, j] > 0.5:  # Keep [k,i,j] order
                    room_matrix[i, j] = k
                    break

    # Creating a visualization
    fig, ax = plt.subplots(figsize=(14, 10))  # Further increase the image size
    unique_rooms = np.unique(room_matrix)
    num_unique_rooms = len(unique_rooms)

    harmonious_colors = [
        [0.55, 0.70, 0.85, 1.0],  # Steel Blue (deeper)
        [0.85, 0.70, 0.50, 1.0],  # Warm Sand (deeper)
        [0.55, 0.78, 0.55, 1.0],  # Sage Green (deeper)
        [0.85, 0.60, 0.60, 1.0],  # Dusty Rose (deeper)
        [0.70, 0.55, 0.85, 1.0],  # Muted Purple (deeper)
        [0.90, 0.75, 0.40, 1.0],  # Mustard/Gold (deeper)
        [0.45, 0.65, 0.80, 1.0],  # Ocean Blue (deeper)
        [0.80, 0.55, 0.55, 1.0],  # Coral (deeper)
    ]

    # If the number of rooms exceeds the predefined colors, use the color cycle of matplotlib to generate more colors
    if num_unique_rooms - 1 > len(harmonious_colors):
        # Generate additional colors using the hsv color space
        import colorsys
        extra_colors_needed = num_unique_rooms - 1 - len(harmonious_colors)
        for i in range(extra_colors_needed):
            h = i / extra_colors_needed
            r, g, b = colorsys.hsv_to_rgb(h, 0.5, 0.80)
            harmonious_colors.append([r, g, b, 1.0])

    # Assign colors to each room
    colors = []
    for i in range(num_unique_rooms):
        if unique_rooms[i] == -1:
            colors.append([1, 1, 1, 1])  # Unallocated areas are white
        else:
            if unique_rooms[i] != -1:
                colors.append(
                    harmonious_colors[int(unique_rooms[i]) % len(harmonious_colors)]
                )

    cmap = ListedColormap(colors)

    # Draw the matrix graph
    im = ax.imshow(room_matrix, cmap=cmap, interpolation="nearest")

    # Create legend item list
    legend_elements = []

    # Add color bar and ensure label alignment with color bar
    cbar = plt.colorbar(im, ticks=unique_rooms, ax=ax, fraction=0.046, pad=0.04)
    room_labels = ["Passage" if r == -1 else textwrap.fill(f"{room_name[r]}", width=30) for r in unique_rooms]

    # Adjust label position to ensure alignment with color bar
    cbar.ax.set_yticklabels(room_labels)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)

    ax.set_title("Floor Plan Visualization", fontsize=16, pad=20, fontweight='bold', color='#333333')
    ax.set_xlabel("Length", fontsize=12, labelpad=10)
    ax.set_ylabel("Width", fontsize=12, labelpad=10)
    
    # Grid lines (softer)
    for i in range(width + 1):
        ax.axhline(i - 0.5, color="#d0d0d0", linestyle="--", linewidth=0.8)
    for j in range(length + 1):
        ax.axvline(j - 0.5, color="#d0d0d0", linestyle="--", linewidth=0.8)

    # Visualize obstacle
    if outdoor_space_array is not None:
        for i in range(width):
            for j in range(length):
                if outdoor_space_array[i, j] > 0.5:
                    plt.fill_between(
                        [j - 0.5, j + 0.5],
                        i - 0.5,
                        i + 0.5,
                        color="#b0b0b0",
                        alpha=0.8,
                        hatch="////",
                        edgecolor="#888888",
                        linewidth=0.5
                    )
        # Visualize obstacle legend
        obstacle_patch = mpatches.Patch(
            facecolor="#b0b0b0", 
            alpha=0.8, 
            hatch="////", 
            edgecolor="#888888", 
            label="Obstacle"
        )
        legend_elements.append(obstacle_patch)

    # Visualize wall if provided
    if wall_array is not None:
        # Direction d: 0=East, 1=West, 2=South, 3=North
        _visualize_walls(wall_array, width, length)
        # Visualize wall legend
        wall_line = plt.Line2D([0], [0], color="#333333", linewidth=3.0, solid_capstyle='round', label="Wall")
        legend_elements.append(wall_line)

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(-0.05, 1.05),
                  frameon=True, facecolor='white', edgecolor='#cccccc',
                  fontsize=10, title="Elements", title_fontsize=11)

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"PNG image saved to {os.path.abspath(save_path)}")

def visualize_floorplan_with_furniture(
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
    save_path="floorplan_furniture_visualization.png",
):
    # Loading a saved solution
    num_rooms, width, length = x_array.shape
    num_furnitures = []
    for k in range(num_rooms):
        num_furnitures.append(furniture_array[k].shape[0])

    # Create a room allocation matrix (each grid stores the room number)
    room_matrix = np.full((width, length), -1)  # -1 means unassigned

    room_name_list = list(furniture_info.keys())
    furniture_name_list = []
    for room_name in room_name_list:
        f_list = []
        for f in furniture_info[room_name]:
            f_list.append(f["name"])
        furniture_name_list.append(f_list)

    # If passage_array is provided, mark the passage area first
    if passage_array is not None:
        for i in range(width):
            for j in range(length):
                if passage_array[i, j] > 0.5:
                    room_matrix[i, j] = -1  # Passage area marked as -1

    # Mark the room area
    for i in range(width):
        for j in range(length):
            for k in range(num_rooms):
                if x_array[k, i, j] > 0.5:  # Keep [k,i,j] order
                    room_matrix[i, j] = k
                    break
    
    # Creating a visualization
    fig, ax = plt.subplots(figsize=(14, 10))  # Further increase the image size
    unique_rooms = np.unique(room_matrix)
    num_unique_rooms = len(unique_rooms)

    # Modern pastel color palette (slightly deeper)
    harmonious_colors = [
        [0.65, 0.78, 0.88, 1.0],  # Steel Blue (light)
        [0.85, 0.75, 0.60, 1.0],  # Warm Sand
        [0.65, 0.82, 0.65, 1.0],  # Sage Green
        [0.88, 0.68, 0.68, 1.0],  # Dusty Rose
        [0.75, 0.65, 0.85, 1.0],  # Muted Purple
        [0.92, 0.82, 0.55, 1.0],  # Mustard/Gold (light)
        [0.55, 0.75, 0.85, 1.0],  # Ocean Blue
        [0.85, 0.65, 0.65, 1.0],  # Coral (muted)
    ]

    # If the number of rooms exceeds the predefined colors, use the color cycle of matplotlib to generate more colors
    if num_unique_rooms - 1 > len(harmonious_colors):
        # Generate additional colors using the hsv color space
        import colorsys
        extra_colors_needed = num_unique_rooms - 1 - len(harmonious_colors)
        for i in range(extra_colors_needed):
            h = i / extra_colors_needed
            r, g, b = colorsys.hsv_to_rgb(h, 0.4, 0.85) # Slightly more saturated and darker
            harmonious_colors.append([r, g, b, 1.0])

    # Assign colors to each room
    colors = []
    for i in range(num_unique_rooms):
        if unique_rooms[i] == -1:
            colors.append([1, 1, 1, 1])  # Unallocated areas are white
        else:
            if unique_rooms[i] != -1:
                colors.append(
                    harmonious_colors[int(unique_rooms[i]) % len(harmonious_colors)]
                )

    cmap = ListedColormap(colors)

    # Draw the matrix graph
    im = ax.imshow(room_matrix, cmap=cmap, interpolation="nearest")

    # Create legend item list
    legend_elements = []

    # Add color bar and ensure label alignment with color bar
    cbar = plt.colorbar(im, ticks=unique_rooms, ax=ax, fraction=0.046, pad=0.04)
    room_labels = ["Passage" if r == -1 else textwrap.fill(f"{room_name_list[r]}", width=30) for r in unique_rooms]

    # Adjust label position to ensure alignment with color bar
    cbar.ax.set_yticklabels(room_labels)
    cbar.outline.set_visible(False) # Remove colorbar border
    cbar.ax.tick_params(size=0) # Remove colorbar ticks

    ax.set_title("Floor Plan Visualization", fontsize=16, pad=20, fontweight='bold', color='#333333')
    ax.set_xlabel("Length", fontsize=12, labelpad=10)
    ax.set_ylabel("Width", fontsize=12, labelpad=10)
    
    # Grid lines (softer)
    for i in range(width + 1):
        ax.axhline(i - 0.5, color="#d0d0d0", linestyle="--", linewidth=0.8)
    for j in range(length + 1):
        ax.axvline(j - 0.5, color="#d0d0d0", linestyle="--", linewidth=0.8)

    # Visualize obstacle
    if outdoor_space_array is not None:
        for i in range(width):
            for j in range(length):
                if outdoor_space_array[i, j] > 0.5:
                    plt.fill_between(
                        [j - 0.5, j + 0.5],
                        i - 0.5,
                        i + 0.5,
                        color="#b0b0b0",
                        alpha=0.8,
                        hatch="////",
                        edgecolor="#888888",
                        linewidth=0.5
                    )
        # Visualize obstacle legend
        obstacle_patch = mpatches.Patch(
            facecolor="#b0b0b0", 
            alpha=0.8, 
            hatch="////", 
            edgecolor="#888888", 
            label="Obstacle"
        )
        legend_elements.append(obstacle_patch)

    # Visualize furniture
    # Rectangle geometry (real-size clamped to the allocated grid cell, with
    # wall-alignment snapping) is shared with the CAD-style renderer -- see
    # utils/furniture_layout_geometry.py for why the clamp alone is enough to
    # guarantee no overlap between different furniture items.
    drawn_furniture_rects = compute_furniture_draw_rects(
        num_rooms,
        width,
        length,
        x_array,
        furniture_array,
        furniture_orientation_sigma_array,
        furniture_orientation_mu_array,
        f_rect_min_i_array,
        f_rect_min_j_array,
        furniture_info,
        furniture_constraints,
    )
    warn_on_furniture_overlaps(drawn_furniture_rects, room_name_list)

    # Draw furniture
    for k in range(num_rooms):
        for l in range(num_furnitures[k]):
            furniture_name = furniture_name_list[k][l]
            j_min, i_min, j_max, i_max = drawn_furniture_rects[k][l]
            j_size = j_max - j_min
            i_size = i_max - i_min
            j_center = (j_min + j_max) / 2
            i_center = (i_min + i_max) / 2
            
            # Draw the furniture rectangle
            plt.fill_between(
                [j_center - j_size / 2, j_center + j_size / 2],
                i_center - i_size / 2,
                i_center + i_size / 2,
                color="#e5e5e5",
                alpha=0.9,
                edgecolor="#444444",
                linewidth=1.2,
                zorder=3
            )
            # Add furniture label
            plt.text(j_center, i_center, furniture_name,
                ha='center',
                va='center',
                fontsize=8,
                color='#222222',
                zorder=4,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
            )
            # Add furniture orientation (arrow)
            if furniture_orientation_sigma_array[k][l] < 0.5 and furniture_orientation_mu_array[k][l] < 0.5:
                # East
                plt.arrow(
                    j_center + j_size / 2,
                    i_center,
                    0.2,
                    0,
                    head_width=0.08 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.1,
                    head_length=0.15 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.2,
                    fc='#666666',
                    ec='#666666',
                    zorder=4
                )
            elif furniture_orientation_sigma_array[k][l] < 0.5 and furniture_orientation_mu_array[k][l] > 0.5:
                # West
                plt.arrow(
                    j_center - j_size / 2,
                    i_center,
                    -0.2,
                    0,
                    head_width=0.08 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.1,
                    head_length=0.15 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.2,
                    fc='#666666',
                    ec='#666666',
                    zorder=4
                )
            elif furniture_orientation_sigma_array[k][l] > 0.5 and furniture_orientation_mu_array[k][l] < 0.5:
                # South
                plt.arrow(
                    j_center,
                    i_center + i_size / 2,
                    0,
                    0.2,
                    head_width=0.08 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.1,
                    head_length=0.15 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.2,
                    fc='#666666',
                    ec='#666666',
                    zorder=4
                )
            elif furniture_orientation_sigma_array[k][l] > 0.5 and furniture_orientation_mu_array[k][l] > 0.5:
                # North
                plt.arrow(
                    j_center,
                    i_center - i_size / 2,
                    0,
                    -0.2,
                    head_width=0.08 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.1,
                    head_length=0.15 * min(j_size, i_size) if min(j_size, i_size) > 0 else 0.2,
                    fc='#666666',
                    ec='#666666',
                    zorder=4
                )

    # Visualize wall if provided
    if wall_array is not None:
        # Direction d: 0=East, 1=West, 2=South, 3=North
        _visualize_walls(wall_array, width, length)

    # Visualize doors and windows
    if door_array is not None:
        _visualize_doors(door_array)
        door_patch = mpatches.Patch(
            facecolor="#FFB74D", # Soft Orange
            alpha=0.9, 
            edgecolor="#F57C00", 
            linewidth=1.2,
            label="Door"
        )
        legend_elements.append(door_patch)

    if window_array is not None:
        _visualize_windows(window_array)
        window_patch = mpatches.Patch(
            facecolor="#81D4FA", # Soft Light Blue
            alpha=0.9, 
            edgecolor="#0288D1", 
            linewidth=1.2,
            label="Window"
        )
        legend_elements.append(window_patch)

    # Add custom legend
    if legend_elements:
        # Adjust legend position to avoid overlap
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(-0.05, 1.05),
                  frameon=True, facecolor='white', edgecolor='#e0e0e0',
                  fontsize=10, title="Elements", title_fontsize=11)

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300) # Add DPI for higher quality
    plt.close()
    print(f"PNG image saved to {os.path.abspath(save_path)}")

def _visualize_walls(wall_array, width, length):
    for d in range(4):
        for i in range(width):
            for j in range(length):
                if wall_array[d, i, j] > 0.5:
                    if d == 0:  # East
                        plt.plot(
                            [j + 0.5, j + 0.5],
                            [i - 0.5, i + 0.5],
                            color="#333333",
                            linewidth=3.0,
                            solid_capstyle='round'
                        )
                    elif d == 1:  # West
                        plt.plot(
                            [j - 0.5, j - 0.5],
                            [i - 0.5, i + 0.5],
                            color="#333333",
                            linewidth=3.0,
                            solid_capstyle='round'
                        )
                    elif d == 2:  # South
                        plt.plot(
                            [j - 0.5, j + 0.5],
                            [i + 0.5, i + 0.5],
                            color="#333333",
                            linewidth=3.0,
                            solid_capstyle='round'
                        )
                    elif d == 3:  # North
                        plt.plot(
                            [j - 0.5, j + 0.5],
                            [i - 0.5, i - 0.5],
                            color="#333333",
                            linewidth=3.0,
                            solid_capstyle='round'
                        )

def _visualize_doors(door_array):
    door_size = round(DOOR_SIZE / F)
    for door in door_array:
        door_dirc, door_face, door_i, door_j = door
        if door_dirc <= 1:
            j_size = door_size
            i_size = 0.2
            if door_dirc == 0:
                j_center = door_j + (j_size - 1) / 2
            else:
                j_center = door_j - (j_size - 1) / 2
            if door_face == 2:
                i_center = door_i + 0.4
            else:
                i_center = door_i - 0.4
        else:
            j_size = 0.2
            i_size = door_size
            if door_dirc == 2:
                i_center = door_i + (i_size - 1) / 2
            else:
                i_center = door_i - (i_size - 1) / 2
            if door_face == 0:
                j_center = door_j + 0.4
            else:
                j_center = door_j - 0.4

        plt.fill_between(
                [j_center - j_size / 2, j_center + j_size / 2],
                i_center - i_size / 2,
                i_center + i_size / 2,
                color="#FFB74D",
                alpha=0.9,
                edgecolor="#F57C00",
                linewidth=1.2,
                zorder=2
            )

def _visualize_windows(window_array):
    window_size = round(WINDOW_SIZE / F)
    for window in window_array:
        window_dirc, window_face, window_i, window_j = window
        if window_dirc <= 1:
            j_size = window_size
            i_size = 0.2
            if window_dirc == 0:
                j_center = window_j + (j_size - 1) / 2
            else:
                j_center = window_j - (j_size - 1) / 2
            if window_face == 2:
                i_center = window_i + 0.5
            else:
                i_center = window_i - 0.5
        else:
            j_size = 0.2
            i_size = window_size
            if window_dirc == 2:
                i_center = window_i + (i_size - 1) / 2
            else:
                i_center = window_i - (i_size - 1) / 2
            if window_face == 0:
                j_center = window_j + 0.5
            else:
                j_center = window_j - 0.5

        plt.fill_between(
                [j_center - j_size / 2, j_center + j_size / 2],
                i_center - i_size / 2,
                i_center + i_size / 2,
                color="#81D4FA",
                alpha=0.9,
                edgecolor="#0288D1",
                linewidth=1.2,
                zorder=2
            )

