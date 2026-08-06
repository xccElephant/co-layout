"""Shared geometry helpers for laying out furniture rectangles in floorplan visualizations.

Both the PNG-style renderer (``utils/floorplan_visualization.py``) and the
CAD/SVG-style renderer (``utils/floorplan_visualization_cadstyle.py``) need to
turn a furniture item's *allocated grid cell* (an integer-sized rectangle
coming out of the IP solve, see ``furniture["width"]``/``furniture["length"]``
in ``run_optimization.py``) into a *drawn rectangle* that uses the item's real
dimensions (``original_width``/``original_length``), optionally snapped flush
against whichever room boundary it touches. This module centralizes that
logic so the two renderers can't drift apart.

Why the clamp matters: the grid cell size is obtained via ``round()`` on the
real size (in grid units), so the real size can occasionally end up slightly
*larger* than the allocated grid cell (whenever the fractional part is below
0.5). If left unclamped, drawing at the real size could make a furniture
item's rectangle spill into a neighboring grid cell -- which may belong to a
different furniture item -- causing a visible overlap. Clamping the drawn
size to ``min(original_size, allocated_size)`` keeps every drawn rectangle
inside its own allocated cell. Since the IP model already guarantees no two
furniture items in the same room share a grid cell, this is enough to
guarantee the drawn rectangles never overlap -- no post-hoc collision
detection/splitting pass is required.
"""


def compute_furniture_draw_rects(
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
):
    """Compute the ``[j_min, i_min, j_max, i_max]`` rectangle to draw for every furniture item.

    Returns a list (indexed by room ``k``) of lists (indexed by furniture ``l``)
    of ``[j_min, i_min, j_max, i_max]`` rectangles, in the same grid coordinate
    system as ``x_array``.
    """
    room_name_list = list(furniture_info.keys())
    num_furnitures = [furniture_array[k].shape[0] for k in range(num_rooms)]

    drawn_furniture_rects = []
    for k in range(num_rooms):
        room_name = room_name_list[k]
        furniture_rect_list = []
        for l in range(num_furnitures[k]):
            f_info = furniture_info[room_name][l]
            parallel_size = f_info["width"]
            vertical_size = f_info["length"]
            original_parallel_size = f_info["original_width"]
            original_vertical_size = f_info["original_length"]

            sigma = furniture_orientation_sigma_array[k][l]
            mu = furniture_orientation_mu_array[k][l]

            if sigma > 0.5:
                # Parallel to j-axis
                i_range = parallel_size
                j_range = vertical_size
                i_size = original_parallel_size
                j_size = original_vertical_size
            else:
                # Parallel to i-axis
                i_range = vertical_size
                j_range = parallel_size
                i_size = original_vertical_size
                j_size = original_parallel_size

            # Clamp the real size to the grid cell allocated by the optimizer
            # (see module docstring): this is what guarantees the drawn
            # rectangle can never spill into a neighboring furniture's cell.
            i_size = min(i_size, i_range)
            j_size = min(j_size, j_range)

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

            if f_info["name"] in furniture_constraints[room_name]["boundary_items"]:
                if sigma < 0.5 and mu < 0.5:
                    if west_boundary:
                        j_center -= (j_range - j_size) / 2
                        west_boundary = False
                    elif north_boundary:
                        i_center -= (i_range - i_size) / 2
                        north_boundary = False
                    elif south_boundary:
                        i_center += (i_range - i_size) / 2
                        south_boundary = False
                elif sigma < 0.5 and mu > 0.5:
                    if east_boundary:
                        j_center += (j_range - j_size) / 2
                        east_boundary = False
                    elif north_boundary:
                        i_center -= (i_range - i_size) / 2
                        north_boundary = False
                    elif south_boundary:
                        i_center += (i_range - i_size) / 2
                        south_boundary = False
                elif sigma > 0.5 and mu < 0.5:
                    if north_boundary:
                        i_center -= (i_range - i_size) / 2
                        north_boundary = False
                    elif west_boundary:
                        j_center -= (j_range - j_size) / 2
                        west_boundary = False
                    elif east_boundary:
                        j_center += (j_range - j_size) / 2
                        east_boundary = False
                elif sigma > 0.5 and mu > 0.5:
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

            current_j_min = j_center - j_size / 2
            current_i_min = i_center - i_size / 2
            current_j_max = j_center + j_size / 2
            current_i_max = i_center + i_size / 2
            furniture_rect_list.append(
                [current_j_min, current_i_min, current_j_max, current_i_max]
            )
        drawn_furniture_rects.append(furniture_rect_list)

    return drawn_furniture_rects


def warn_on_furniture_overlaps(drawn_furniture_rects, room_name_list):
    """Defensive sanity check: log a warning if any two drawn rectangles overlap.

    With the clamping in ``compute_furniture_draw_rects`` this should never
    trigger; it is kept as a cheap safety net to surface unexpected data
    (e.g. floating-point noise from the solver, or future logic changes)
    instead of silently rendering a wrong picture.
    """
    import itertools

    for k, rect_list in enumerate(drawn_furniture_rects):
        room_name = room_name_list[k] if k < len(room_name_list) else k
        for l1, l2 in itertools.combinations(range(len(rect_list)), 2):
            j1_min, i1_min, j1_max, i1_max = rect_list[l1]
            j2_min, i2_min, j2_max, i2_max = rect_list[l2]
            overlap_j = min(j1_max, j2_max) - max(j1_min, j2_min)
            overlap_i = min(i1_max, i2_max) - max(i1_min, i2_min)
            if overlap_j > 1e-6 and overlap_i > 1e-6:
                print(
                    f"WARNING: unexpected furniture overlap in room '{room_name}' "
                    f"between items {l1} and {l2} "
                    f"(overlap area ~{overlap_j * overlap_i:.4f})."
                )
