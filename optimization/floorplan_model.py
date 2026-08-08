import os
import time

import numpy as np
from gurobipy import *

from constants import *
from optimization.grid_model import PassageGrid, RoomGrid
from optimization.model_utils import (
    add_room_rectangularity_constraints,
    add_room_adjacency_constraints,
    add_room_bounding_box_constraints,
    add_room_corner_constraints,
    build_layout_context,
    configure_model_log,
    run_staged_optimization,
)
from utils.connectivity import connected_components, separator_cuts
from utils.floorplan_visualization import visualize_floorplan
from utils.floorplan_visualization_cadstyle import visualize_floorplan_cadstyle
from utils.post_process import add_wall


class FloorplanModel:
    def __init__(
        self,
        model_name: str,
        session_id: str,
        building_params: dict,
        rooms: dict,
        room_constraints: dict,
    ):
        self.model_name = model_name
        self.session_id = session_id
        self.output_dir = get_optimization_dir(session_id)
        self.visualization_dir = get_visualization_dir(session_id)
        self.width = building_params["width"]
        self.length = building_params["length"]
        self.entrance = building_params["entrance_location"]
        self.outdoor_space_coordinates = building_params["outdoor_space"]
        self.rooms = rooms
        self.room_constraints = room_constraints
        self.weights = COARSE_MODEL_WEIGHTS

        # Gurobi Model Definition
        self.model = Model(model_name)

        # Model Parameters
        self.model.Params.MIPGap = 0.01
        self.model.Params.TimeLimit = 300
        self.model.setParam("Threads", CPU_THREADS)
        self.set_log()

        # Preprocessing
        self.pre_process()
        # Result arrays
        self.create_arrays()

        # Variable definitions
        self.x = RoomGrid(self.model, self.room_num, self.valid_coordinates, name="x")
        self.passage = PassageGrid(self.model, self.valid_coordinates, name="passage")

        self.objective_function = QuadExpr()
        # Constraint definitions
        self.set_constraints()
        # Objective function
        self.set_objective_function()

    def pre_process(self):
        context = build_layout_context(
            self.width,
            self.length,
            self.rooms,
            self.outdoor_space_coordinates,
            print_big_m=True,
        )
        for key, value in context.items():
            setattr(self, key, value)

    def create_arrays(self):
        self.x_array = np.zeros((self.room_num, self.width, self.length))
        self.passage_array = np.zeros((self.width, self.length))
        self.outdoor_space_array = np.zeros((self.width, self.length))

        for coordinate in self.outdoor_space_coordinates:
            i, j = coordinate
            self.outdoor_space_array[i, j] = 1

    def post_process(self):
        self.wall_array = add_wall(self)

    def _build_visualization_callback(self):
        def visualization_callback(model, where):
            if not hasattr(model, "_callback_data"):
                model._callback_data = {
                    "last_update": time.time(),
                    "iteration": 0,
                    "update_interval": 5,
                }
            data = model._callback_data
            current_time = time.time()
            if (
                where == GRB.Callback.MIPSOL
                and current_time - data["last_update"]
                >= data["update_interval"]
            ):
                data["iteration"] += 1
                for i, j in self.valid_coordinates:
                    self.passage_array[i, j] = model.cbGetSolution(
                        self.passage[i, j]
                    )
                    for room_index in range(self.room_num):
                        self.x_array[room_index, i, j] = (
                            model.cbGetSolution(
                                self.x[room_index, i, j]
                            )
                        )
                visualize_floorplan(
                    self.x_array,
                    self.passage_array,
                    self.outdoor_space_array,
                    None,
                    self.room_name_list,
                    str(self.visualization_dir / f"{self.model_name}.png"),
                )
                data["last_update"] = current_time

        return visualization_callback

    def _build_connectivity_callback(self, visualization_callback=None):
        self.model._connectivity_cut_keys = set()
        self.model._connectivity_lazy_stats = {
            "mipsol_calls": 0,
            "rejected_candidates": 0,
            "accepted_candidates": 0,
            "room_cuts": 0,
            "free_cuts": 0,
            "cuts_added": 0,
        }
        room_solution_keys = [
            (room_index, coordinate)
            for room_index in range(self.room_num)
            for coordinate in self.valid_coordinates
        ]
        room_solution_vars = [
            self.x[room_index, *coordinate]
            for room_index, coordinate in room_solution_keys
        ]

        def connectivity_callback(model, where):
            if where != GRB.Callback.MIPSOL:
                if visualization_callback is not None:
                    visualization_callback(model, where)
                return

            stats = model._connectivity_lazy_stats
            stats["mipsol_calls"] += 1
            room_solution = model.cbGetSolution(room_solution_vars)
            active_by_room = [set() for _ in range(self.room_num)]
            for (
                room_index,
                coordinate,
            ), value in zip(room_solution_keys, room_solution):
                if value > 0.5:
                    active_by_room[room_index].add(coordinate)

            candidate_is_disconnected = False
            for room_index in range(self.room_num):
                components = connected_components(
                    active_by_room[room_index],
                    self.valid_coordinates,
                )
                cuts = separator_cuts(components, self.valid_coordinates)
                if cuts:
                    candidate_is_disconnected = True
                for cut in cuts:
                    cut_key = (
                        "room",
                        room_index,
                        cut.component,
                        cut.boundary,
                        cut.source,
                        cut.target,
                    )
                    if cut_key in model._connectivity_cut_keys:
                        continue
                    model._connectivity_cut_keys.add(cut_key)
                    model.cbLazy(
                        quicksum(
                            self.x[room_index, *coordinate]
                            for coordinate in sorted(cut.boundary)
                        )
                        >= (
                            self.x[room_index, *cut.source]
                            + self.x[room_index, *cut.target]
                            - 1
                        )
                    )
                    stats["room_cuts"] += 1
                    stats["cuts_added"] += 1

            if candidate_is_disconnected:
                stats["rejected_candidates"] += 1
                return

            stats["accepted_candidates"] += 1
            if visualization_callback is not None:
                visualization_callback(model, where)

        connectivity_callback.reset = (
            lambda: self.model._connectivity_cut_keys.clear()
        )
        return connectivity_callback

    def set_objective_function(self):
        # Area Error Term
        # Create auxiliary variables to represent area errors.
        area_error = self.model.addVars(
            self.room_num,
            vtype=GRB.INTEGER,
            lb=0,
            # ub=self.width * self.length,
            name="area_error",
        )
        for k in range(self.room_num):
            # Define area error
            self.model.addConstr(
                area_error[k]
                >= quicksum(self.x[k, i, j] for (i, j) in self.valid_coordinates)
                - self.room_area_list[k]
            )
            self.model.addConstr(
                area_error[k]
                >= self.room_area_list[k]
                - quicksum(self.x[k, i, j] for (i, j) in self.valid_coordinates)
            )  # Add to the objective function
            self.objective_function += self.weights["area"] * area_error[k]

        # Perimeter Term
        # Minimize room perimeter/maximize room interior adjacency
        for k in range(self.room_num):
            for i, j in self.valid_coordinates:
                # East
                self.objective_function += (
                    self.weights["perimeter"]
                    * (1 - self.x[k, i, j + 1])
                    * self.x[k, i, j]
                    if (i, j + 1) in self.valid_coordinates
                    else 0
                )
                # West
                self.objective_function += (
                    self.weights["perimeter"]
                    * (1 - self.x[k, i, j - 1])
                    * self.x[k, i, j]
                    if (i, j - 1) in self.valid_coordinates
                    else 0
                )
                # South
                self.objective_function += (
                    self.weights["perimeter"]
                    * (1 - self.x[k, i + 1, j])
                    * self.x[k, i, j]
                    if (i + 1, j) in self.valid_coordinates
                    else 0
                )
                # North
                self.objective_function += (
                    self.weights["perimeter"]
                    * (1 - self.x[k, i - 1, j])
                    * self.x[k, i, j]
                    if (i - 1, j) in self.valid_coordinates
                    else 0
                )

        # Rectangularity Term
        # Penalize the difference between the room area and its bounding box area
        rect_diff_slack = add_room_rectangularity_constraints(
            self.model,
            self.x,
            room_num=self.room_num,
            valid_coordinates=self.valid_coordinates,
            rect_len_i=self.rect_len_i,
            rect_len_j=self.rect_len_j,
            width=self.width,
            length=self.length,
        )
        for k in range(self.room_num):
            self.objective_function += (
                self.weights["rectangularity"] * rect_diff_slack[k]
            )

        # Shape ratio Term
        # Creating Slack Variables
        shape_ratio_slack = self.model.addVars(
            self.room_num, vtype=GRB.INTEGER, lb=0, name="shape_ratio_slack"
        )
        for k in range(self.room_num):
            self.model.addConstr(
                shape_ratio_slack[k] >= self.rect_len_i[k] - self.rect_len_j[k]
            )
            self.model.addConstr(
                shape_ratio_slack[k] >= self.rect_len_j[k] - self.rect_len_i[k]
            )
            self.objective_function += self.weights["shape"] * shape_ratio_slack[k]

        # Privacy Term
        # Precompute distances only for valid coordinates if needed, but using all might be fine.
        privacy_distance = {
            (i, j): abs(i - self.entrance[1]) + abs(j - self.entrance[2])
            for i in range(self.width)
            for j in range(self.length)
            # Optionally restrict to: if (i, j) in self.valid_coordinates_set
        }

        # Create a room privacy score expression
        privacy_score = [LinExpr() for _ in range(self.room_num)]

        # Calculate score using only valid coordinates for summation
        for k in range(self.room_num):
            # Avoid division by zero if room area is unexpectedly zero
            room_area_k = self.room_area_list[k] if self.room_area_list[k] > 0 else 1
            privacy_score[k] = (1.0 / room_area_k) * quicksum(
                self.x[k, i, j] * privacy_distance.get((i, j), 0)  # Use .get for safety
                for i, j in self.valid_coordinates  # Sum over valid coordinates
            )
        privacy_slack = self.model.addVars(
            len(self.privacy_order) - 1,
            vtype=GRB.INTEGER,
            lb=0,
            name="privacy_slack",
        )

        # Add order relation soft constraints according to privacy_order
        for i in range(len(self.privacy_order) - 1):
            k1 = self.privacy_order[i]  # Room indexing for higher privacy levels
            k2 = self.privacy_order[i + 1]  # Room index for lower privacy levels
            self.model.addConstr(
                privacy_score[k1] + privacy_slack[i] >= privacy_score[k2],
                name=f"privacy_order_{k1}_{k2}",
            )
            self.objective_function += self.weights["privacy"] * privacy_slack[i]

    def optimize(self, stage_num: int = 2):
        self.model.Params.LazyConstraints = 1
        connectivity_callback = self._build_connectivity_callback(
            self._build_visualization_callback()
        )
        run_staged_optimization(
            self.model,
            self.model_name,
            self.objective_function,
            stage_num,
            self.output_dir,
            callback=connectivity_callback,
        )
        connectivity_stats = self.model._connectivity_lazy_stats
        print(
            "Coarse lazy connectivity: "
            f"callbacks={connectivity_stats['mipsol_calls']} "
            f"rejected={connectivity_stats['rejected_candidates']} "
            f"cuts={connectivity_stats['cuts_added']}"
        )

        # If the model is infeasible, output infeasible constraint report
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write(
                os.path.join(
                    self.output_dir,
                    f"{self.model_name}.ilp",
                )
            )

        if self.model.status in [
            GRB.OPTIMAL,
            GRB.TIME_LIMIT,
            GRB.SUBOPTIMAL,
            GRB.INTERRUPTED,
        ]:
            # Output the status of the solution
            status_map = {
                GRB.OPTIMAL: "Optimal solution",
                GRB.TIME_LIMIT: "Time limit reached",
                GRB.SUBOPTIMAL: "Suboptimal solution",
                GRB.INTERRUPTED: "User interruption",
            }
            print(f"State: {status_map.get(self.model.status, 'Unknown status')}")

            # Solution existence check
            if self.model.SolCount > 0:
                # Extract the solution
                for i, j in self.valid_coordinates:
                    self.passage_array[i, j] = self.passage[i, j].X
                    for k in range(self.room_num):
                        self.x_array[k, i, j] = self.x[k, i, j].X
                self.post_process()
                # Final Visulizaiton
                visualize_floorplan(
                    self.x_array,
                    self.passage_array,
                    self.outdoor_space_array,
                    self.wall_array,
                    self.room_name_list,
                    str(self.visualization_dir / f"{self.model_name}.png"),
                )
                visualize_floorplan_cadstyle(
                    self.x_array,
                    self.passage_array,
                    self.outdoor_space_array,
                    self.wall_array,
                    str(self.visualization_dir / f"{self.model_name}.svg"),
                )
            else:
                print("No feasible solution found")
        else:
            print(f"Model solving failed, status code: {self.model.status}")

    def _add_room_connectivity_cuts(self, round_index: int) -> int:
        cuts_added = 0
        for room_index in range(self.room_num):
            active_coordinates = {
                coordinate
                for coordinate in self.valid_coordinates
                if self.x[room_index, *coordinate].X > 0.5
            }
            components = connected_components(
                active_coordinates,
                self.valid_coordinates,
            )
            cuts = separator_cuts(components, self.valid_coordinates)
            for cut_index, cut in enumerate(cuts):
                self.model.addConstr(
                    quicksum(
                        self.x[room_index, *coordinate]
                        for coordinate in sorted(cut.boundary)
                    )
                    >= (
                        self.x[room_index, *cut.source]
                        + self.x[room_index, *cut.target]
                        - 1
                    ),
                    name=(
                        f"room_connectivity_r{round_index}"
                        f"_k{room_index}_c{cut_index}"
                    ),
                )
                cuts_added += 1
        return cuts_added

    def set_log(self):
        configure_model_log(self.model, self.model_name, self.output_dir)

    def set_constraints(self):
        self.add_passage_basic_constraints()
        self.add_room_nonempty_constraints()
        self.add_flow_constraints()

        self.add_room_bounding_box_constraints()
        self.add_room_at_corner_constraints()
        self.add_room_adjacent_constraints()
        self.add_room_accessibility_constraints()

    def add_room_nonempty_constraints(self):
        for room_index in range(self.room_num):
            self.model.addConstr(
                quicksum(
                    self.x[room_index, *coordinate]
                    for coordinate in self.valid_coordinates
                )
                >= 1,
                name=f"room_nonempty_{room_index}",
            )

    def add_room_at_corner_constraints(self):
        add_room_corner_constraints(
            self.model,
            self.x,
            self.room_constraints,
            self.room_name_list,
            self.north_west,
            self.north_east,
            self.south_west,
            self.south_east,
        )

    def add_room_bounding_box_constraints(self):
        # Bounding box constraints for each room
        self.rect_min_i = self.model.addVars(
            self.room_num,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.width - 1,
            name="rect_min_i",
        )
        self.rect_len_i = self.model.addVars(
            self.room_num,
            vtype=GRB.INTEGER,
            lb=1,
            ub=self.width,
            name="rect_len_i",
        )
        self.rect_min_j = self.model.addVars(
            self.room_num,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.length - 1,
            name="rect_min_j",
        )
        self.rect_len_j = self.model.addVars(
            self.room_num,
            vtype=GRB.INTEGER,
            lb=1,
            ub=self.length,
            name="rect_len_j",
        )
        add_room_bounding_box_constraints(
            self.model,
            self.x,
            self.room_num,
            self.valid_coordinates,
            self.BigM_i,
            self.BigM_j,
            self.rect_min_i,
            self.rect_len_i,
            self.rect_min_j,
            self.rect_len_j,
        )

    def add_passage_basic_constraints(self):
        # passage[i,j] = 1 if and only if the cell is valid and does not belong to any room
        for i, j in self.valid_coordinates:
            self.model.addConstr(
                self.passage[i, j]
                + quicksum(self.x[k, i, j] for k in range(self.room_num))
                == 1,
                name=f"passage_cell_assignment_{i}_{j}",
            )

        # entrance belongs to passage
        self.model.addConstr(
            self.passage[self.entrance[1], self.entrance[2]] == 1,
        )

    def add_flow_constraints(self):
        # Flow constraint, take the entrance as the source point, flow can only move between the corridor and open rooms.
        f_passage = self.model.addVars(
            4,
            self.valid_coordinates,
            vtype=GRB.INTEGER,
            name="passage_flow",
            lb=0,
            ub=len(self.valid_coordinates),
        )

        passage_or_openroom_expr = {}
        for i, j in self.valid_coordinates:
            passage_or_openroom_expr[(i, j)] = self.passage[i, j] + quicksum(
                self.x[k, i, j] for k in self.open_room_indices
            )

        for i, j in self.valid_coordinates:
            outflow_passage = LinExpr()
            inflow_passage = LinExpr()

            # East Neighbor (i, j+1)
            neighbor_coord_east = (i, j + 1)
            if neighbor_coord_east in self.valid_coordinates_set:
                outflow_passage += f_passage[EAST, i, j]  # Flowing out to the east
                inflow_passage += f_passage[WEST, i, j + 1]  # Flowing in from the east
                self.model.addConstr(
                    f_passage[EAST, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[(i, j)],  # Flow from valid node
                    name=f"passage_flow_limit_east_from_{i}_{j}",
                )
                self.model.addConstr(
                    f_passage[EAST, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[
                        neighbor_coord_east
                    ],  # Flow to valid node
                    name=f"passage_flow_limit_east_to_{i}_{j+1}",
                )

            # West Neighbor (i, j-1)
            neighbor_coord_west = (i, j - 1)
            if neighbor_coord_west in self.valid_coordinates_set:
                outflow_passage += f_passage[WEST, i, j]  # Flowing out to the west
                inflow_passage += f_passage[EAST, i, j - 1]  # Flowing in from the west
                self.model.addConstr(
                    f_passage[WEST, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)],
                    name=f"passage_flow_limit_west_from_{i}_{j}",
                )
                self.model.addConstr(
                    f_passage[WEST, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_west],
                    name=f"passage_flow_limit_west_to_{i}_{j-1}",
                )

            # South Neighbor (i+1, j)
            neighbor_coord_south = (i + 1, j)
            if neighbor_coord_south in self.valid_coordinates_set:
                outflow_passage += f_passage[SOUTH, i, j]  # Flowing out to the south
                inflow_passage += f_passage[
                    NORTH, i + 1, j
                ]  # Flowing in from the south
                self.model.addConstr(
                    f_passage[SOUTH, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)],
                    name=f"passage_flow_limit_south_from_{i}_{j}",
                )
                self.model.addConstr(
                    f_passage[SOUTH, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_south],
                    name=f"passage_flow_limit_south_to_{i+1}_{j}",
                )

            # North Neighbor (i-1, j)
            neighbor_coord_north = (i - 1, j)
            if neighbor_coord_north in self.valid_coordinates_set:
                outflow_passage += f_passage[NORTH, i, j]  # Flowing out to the north
                inflow_passage += f_passage[
                    SOUTH, i - 1, j
                ]  # Flowing in from the north
                self.model.addConstr(
                    f_passage[NORTH, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)],
                    name=f"passage_flow_limit_north_from_{i}_{j}",
                )
                self.model.addConstr(
                    f_passage[NORTH, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_north],
                    name=f"passage_flow_limit_north_to_{i-1}_{j}",
                )

            # Net flow constraint:
            entrance_i, entrance_j = self.entrance[1], self.entrance[2]
            if (i, j) == (entrance_i, entrance_j):
                # Source node constraint
                self.model.addConstr(
                    outflow_passage - inflow_passage
                    == quicksum(
                        passage_or_openroom_expr[p, q]
                        for p, q in self.valid_coordinates
                    )
                    - 1,
                    name=f"passage_flow_net_source_{i}_{j}",
                )
            else:
                # Sink node constraint (demand = 1 if it's a passage cell, else 0)
                self.model.addConstr(
                    outflow_passage - inflow_passage == -passage_or_openroom_expr[i, j],
                    name=f"passage_flow_net_node_{i}_{j}",
                )

    def add_room_adjacent_constraints(self):
        open_room_names = {
            self.room_name_list[room_index]
            for room_index in self.open_room_indices
        }
        room_adjacency = self.room_constraints["room_adjacency"]
        coarse_adjacency = [
            pair
            for pair in room_adjacency
            if pair[0] in open_room_names or pair[1] in open_room_names
        ]
        deferred_adjacency = [
            pair for pair in room_adjacency if pair not in coarse_adjacency
        ]
        if deferred_adjacency:
            print(
                "Coarse grid defers closed-room adjacency constraints "
                f"to Fine grid: {deferred_adjacency}"
            )
        coarse_room_constraints = {
            **self.room_constraints,
            "room_adjacency": coarse_adjacency,
        }
        add_room_adjacency_constraints(
            self.model,
            self.x,
            coarse_room_constraints,
            self.room_name_list,
            self.valid_coordinates,
        )

    def add_room_accessibility_constraints(self):
        # Accessibility: Each room is adjacent to passage or open rooms (Exclude open rooms)
        for k in range(self.room_num):
            # Ensure at least one adjacency point exists over valid coordinates
            if k not in self.open_room_indices:
                room_adj_passage_or_openroom = self.model.addVars(
                    self.valid_coordinates,  # Define only for valid coords
                    vtype=GRB.BINARY,
                    name=f"room_adj_passage_or_openroom_{k}",
                )
                self.model.addConstr(
                    quicksum(
                        room_adj_passage_or_openroom[i, j]
                        for i, j in self.valid_coordinates  # Sum over valid coords
                    )
                    >= round(DOOR_SIZE / F),
                    name=f"room_access_at_least_one_{k}",
                )
                for i, j in self.valid_coordinates:
                    neighbors = LinExpr()
                    # Neighbors can be passage or open rooms
                    neighbors = self.passage[i + 1, j] + quicksum(
                        self.x[l, i + 1, j] for l in self.open_room_indices
                    )  # South
                    neighbors += self.passage[i - 1, j] + quicksum(
                        self.x[l, i - 1, j] for l in self.open_room_indices
                    )  # North
                    neighbors += self.passage[i, j + 1] + quicksum(
                        self.x[l, i, j + 1] for l in self.open_room_indices
                    )  # East
                    neighbors += self.passage[i, j - 1] + quicksum(
                        self.x[l, i, j - 1] for l in self.open_room_indices
                    )  # West

                    # If room_adj_passage_or_openroom[k,i,j]=1, then cell (i,j) must belong to room k
                    self.model.addConstr(
                        neighbors >= room_adj_passage_or_openroom[i, j],
                        name=f"room_access_neighbor_link_{k}_{i}_{j}",
                    )
                    self.model.addConstr(
                        room_adj_passage_or_openroom[i, j] <= self.x[k, i, j],
                        name=f"room_access_room_link_{k}_{i}_{j}",
                    )

    def get_solutions(self):
        return self.x_array
