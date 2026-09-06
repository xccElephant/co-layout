import os
import time
import logging
import tempfile
import numpy as np
from gurobipy import *


from constants import *
from optimization.grid_model import FurnitureGrid, PassageGrid, RoomGrid
from optimization.model_utils import (
    ConnectivityOptimizationError,
    add_fixed_root_connectivity,
    add_room_rectangularity_constraints,
    add_room_adjacency_constraints,
    add_room_bounding_box_constraints,
    add_room_corner_constraints,
    build_layout_context,
    configure_model_log,
    run_connectivity_cut_loop,
    run_staged_optimization,
)
from utils.connectivity import (
    connected_components,
    separator_cuts,
    validate_layout_connectivity,
)
from utils.post_process import add_door, add_wall, add_window, rearrange
from utils.floorplan_visualization import visualize_floorplan_with_furniture
from utils.floorplan_visualization_cadstyle import visualize_floorplan_with_furniture_cadstyle


logger = logging.getLogger(__name__)


class CooptModel:
    def __init__(
        self,
        model_name: str,
        session_id: str,
        building_params: dict,
        rooms: dict,
        room_constraints: dict,
        furnitures: dict,
        furniture_constraints: dict
    ):
        self.model_name = model_name
        self.session_id = session_id
        self.output_dir = get_optimization_dir(session_id)
        self.visualization_dir = get_visualization_dir(session_id)
        self.width = building_params["width"]
        self.length = building_params["length"]
        self.rooms = rooms
        self.room_constraints = room_constraints
        self.outdoor_space_coordinates = building_params["outdoor_space"]
        self.entrance = building_params["entrance_location"]
        self.furnitures = furnitures
        self.furniture_constraints = furniture_constraints
        self.model = Model(model_name)
        self.weights = FINE_MODEL_WEIGHTS
        # Model Parameters
        self.model.Params.MIPGap = 0.01
        self.model.Params.TimeLimit = 600
        self.model.setParam("Threads", CPU_THREADS)
        self.set_log()

        # Pre-processing
        self.pre_process()
        # Result Arrays
        self.create_arrays()

        # Variable Definitions
        self.set_variables()
        self.objective_function = QuadExpr()

        # Constraints
        self.set_constraints()
        self._furniture_relation_objective = (
            self.objective_function.copy()
        )
        # Objective Function
        self.set_objective_function()

    def pre_process(self):
        context = build_layout_context(
            self.width,
            self.length,
            self.rooms,
            self.outdoor_space_coordinates,
            print_big_m=False,
        )
        for key, value in context.items():
            setattr(self, key, value)

        self.furniture_num_list = [
            len(self.furnitures[room_name]) for room_name in self.room_name_list
        ]
        self.furniture_name_list = []
        for room_name in self.room_name_list:
            f_list = []
            for f in self.furnitures[room_name]:
                f_list.append(f["name"])
            self.furniture_name_list.append(f_list)
        # Considering the orientation of the furniture, some are parallel to the orientation along the long side (like a bed), while others are along the short side (like a sofa), so these attributes need to be determined in advance.
        self.furniture_parallel_size = []
        self.furniture_vertical_size = []
        self.furniture_area_list = []
        for k in range(self.room_num):
            p_list = []
            v_list = []
            area_list = []
            room_name = self.room_name_list[k]
            for l in range(self.furniture_num_list[k]):
                f_length = self.furnitures[room_name][l]["length"]
                f_width = self.furnitures[room_name][l]["width"]
                area_list.append(f_length * f_width)
                p_list.append(f_width)
                v_list.append(f_length)
            self.furniture_parallel_size.append(p_list)
            self.furniture_vertical_size.append(v_list)
            self.furniture_area_list.append(area_list)

        # Create a furniture index list, which contains the unique identifiers of all furniture in the data (room index + furniture index in the room)
        self.furniture_indices = []
        for room_idx in range(self.room_num):
            for furniture_idx in range(self.furniture_num_list[room_idx]):
                self.furniture_indices.append((room_idx, furniture_idx))

    def set_variables(self):
        self.x = RoomGrid(self.model, self.room_num, self.valid_coordinates, name="x")
        self.passage = PassageGrid(self.model, self.valid_coordinates, name="passage")

        # Define variables for all the furniture in the room - using sparse indexing
        self.furniture = FurnitureGrid(
            self.model,
            self.furniture_indices,
            self.valid_coordinates,
            name="furniture",
        )

        self.f_rect_min_i = self.model.addVars(
            self.furniture_indices,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.width - 1,
            name="f_rect_min_i",
        )

        self.f_rect_min_j = self.model.addVars(
            self.furniture_indices,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.length - 1,
            name="f_rect_min_j",
        )

        # Introduce binary variables to represent the orientation of the furniture.
        # (sigma, mu) = 
        # (0, 0) -> Right/East
        # (0, 1) -> Left/West
        # (1, 0) -> Down/South
        # (1, 1) -> Up/North
        self.sigma = self.model.addVars(
            self.furniture_indices, vtype=GRB.BINARY, name="sigma"
        )
        self.mu = self.model.addVars(
            self.furniture_indices, vtype=GRB.BINARY, name="mu"
        )

        self.flow = self.model.addVars(
            4,
            self.valid_coordinates,
            vtype=GRB.INTEGER,
            name="flow",
            lb=0,
            ub=len(self.valid_coordinates),
        )

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

    def create_arrays(self):
        self.x_array = np.zeros((self.room_num, self.width, self.length))
        self.passage_array = np.zeros((self.width, self.length))
        self.outdoor_space_array = np.zeros((self.width, self.length))
        self.wall_array = np.zeros((4, self.width, self.length))
        for coordinate in self.outdoor_space_coordinates:
            i, j = coordinate
            self.outdoor_space_array[i, j] = 1

        self.furniture_array = []
        self.furniture_orientation_sigma_array = []
        self.furniture_orientation_mu_array = []
        self.f_rect_min_i_array = []
        self.f_rect_min_j_array = []

        for k in range(self.room_num):
            self.furniture_array.append(
                np.zeros((self.furniture_num_list[k], self.width, self.length))
            )
            self.furniture_orientation_sigma_array.append(
                np.zeros(self.furniture_num_list[k])
            )
            self.furniture_orientation_mu_array.append(
                np.zeros(self.furniture_num_list[k])
            )
            self.f_rect_min_i_array.append(np.zeros(self.furniture_num_list[k]))
            self.f_rect_min_j_array.append(np.zeros(self.furniture_num_list[k]))

        self.rect_len_i_array = np.zeros(self.room_num)
        self.rect_len_j_array = np.zeros(self.room_num)

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
            if where == GRB.Callback.MIPSOL:
                if current_time - data["last_update"] >= data["update_interval"]:
                    data["iteration"] += 1
                    for i, j in self.valid_coordinates:
                        self.passage_array[i, j] = model.cbGetSolution(self.passage[i, j])
                        for k in range(self.room_num):
                            self.x_array[k, i, j] = model.cbGetSolution(self.x[k, i, j])

                    for k in range(self.room_num):
                        for l in range(self.furniture_num_list[k]):
                            for i, j in self.valid_coordinates:
                                self.furniture_array[k][l, i, j] = model.cbGetSolution(
                                    self.furniture[k, l, i, j]
                                )
                            self.furniture_orientation_sigma_array[k][l] = model.cbGetSolution(
                                self.sigma[k, l]
                            )
                            self.furniture_orientation_mu_array[k][l] = model.cbGetSolution(
                                self.mu[k, l]
                            )
                            self.f_rect_min_i_array[k][l] = model.cbGetSolution(
                                self.f_rect_min_i[k, l]
                            )
                            self.f_rect_min_j_array[k][l] = model.cbGetSolution(
                                self.f_rect_min_j[k, l]
                            )

                    visualize_floorplan_with_furniture(
                        self.x_array,
                        self.passage_array,
                        self.furniture_array,
                        self.furniture_orientation_sigma_array,
                        self.furniture_orientation_mu_array,
                        self.f_rect_min_i_array,
                        self.f_rect_min_j_array,
                        self.outdoor_space_array,
                        None,
                        None,
                        None,
                        self.furnitures,
                        self.furniture_constraints,
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
        furniture_solution_keys = [
            (room_index, furniture_index, coordinate)
            for room_index in self.common_room_indices
            for furniture_index in range(
                self.furniture_num_list[room_index]
            )
            for coordinate in self.valid_coordinates
        ]
        furniture_solution_vars = [
            self.furniture[
                room_index,
                furniture_index,
                *coordinate,
            ]
            for (
                room_index,
                furniture_index,
                coordinate,
            ) in furniture_solution_keys
        ]

        def connectivity_callback(model, where):
            if where != GRB.Callback.MIPSOL:
                if visualization_callback is not None:
                    visualization_callback(model, where)
                return

            stats = model._connectivity_lazy_stats
            stats["mipsol_calls"] += 1
            room_solution = model.cbGetSolution(room_solution_vars)
            room_value_by_coordinate = {}
            active_by_room = [set() for _ in range(self.room_num)]
            for (
                room_index,
                coordinate,
            ), value in zip(room_solution_keys, room_solution):
                room_value_by_coordinate[room_index, coordinate] = value
                if value > 0.5:
                    active_by_room[room_index].add(coordinate)

            furniture_occupancy = {
                (room_index, coordinate): 0.0
                for room_index in self.common_room_indices
                for coordinate in self.valid_coordinates
            }
            if furniture_solution_vars:
                furniture_solution = model.cbGetSolution(
                    furniture_solution_vars
                )
                for (
                    room_index,
                    _,
                    coordinate,
                ), value in zip(
                    furniture_solution_keys,
                    furniture_solution,
                ):
                    furniture_occupancy[room_index, coordinate] += value

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

            for room_index in self.common_room_indices:
                free_coordinates = {
                    coordinate
                    for coordinate in self.valid_coordinates
                    if (
                        room_value_by_coordinate[room_index, coordinate]
                        - furniture_occupancy[room_index, coordinate]
                        > 0.5
                    )
                }
                components = connected_components(
                    free_coordinates,
                    self.valid_coordinates,
                )
                cuts = separator_cuts(components, self.valid_coordinates)
                if cuts:
                    candidate_is_disconnected = True
                for cut in cuts:
                    cut_key = (
                        "free",
                        room_index,
                        cut.component,
                        cut.boundary,
                        cut.source,
                        cut.target,
                    )
                    if cut_key in model._connectivity_cut_keys:
                        continue
                    model._connectivity_cut_keys.add(cut_key)
                    boundary_free = quicksum(
                        self.x[room_index, *coordinate]
                        - quicksum(
                            self.furniture[
                                room_index,
                                furniture_index,
                                *coordinate,
                            ]
                            for furniture_index in range(
                                self.furniture_num_list[room_index]
                            )
                        )
                        for coordinate in sorted(cut.boundary)
                    )
                    source_free = (
                        self.x[room_index, *cut.source]
                        - quicksum(
                            self.furniture[
                                room_index,
                                furniture_index,
                                *cut.source,
                            ]
                            for furniture_index in range(
                                self.furniture_num_list[room_index]
                            )
                        )
                    )
                    target_free = (
                        self.x[room_index, *cut.target]
                        - quicksum(
                            self.furniture[
                                room_index,
                                furniture_index,
                                *cut.target,
                            ]
                            for furniture_index in range(
                                self.furniture_num_list[room_index]
                            )
                        )
                    )
                    model.cbLazy(
                        boundary_free >= source_free + target_free - 1
                    )
                    stats["free_cuts"] += 1
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

    def _render_final_layout(self):
        visualize_floorplan_with_furniture(
            self.x_array,
            self.passage_array,
            self.furniture_array,
            self.furniture_orientation_sigma_array,
            self.furniture_orientation_mu_array,
            self.f_rect_min_i_array,
            self.f_rect_min_j_array,
            self.outdoor_space_array,
            self.wall_array,
            self.door_array,
            self.window_array,
            self.furnitures,
            self.furniture_constraints,
            str(self.visualization_dir / f"{self.model_name}.png"),
        )
        visualize_floorplan_with_furniture_cadstyle(
            self.x_array,
            self.passage_array,
            self.furniture_array,
            self.furniture_orientation_sigma_array,
            self.furniture_orientation_mu_array,
            self.f_rect_min_i_array,
            self.f_rect_min_j_array,
            self.outdoor_space_array,
            self.wall_array,
            self.door_array,
            self.window_array,
            self.furnitures,
            self.furniture_constraints,
            str(self.visualization_dir / f"{self.model_name}.svg"),
            results_path=str(self.output_dir / "result.json"),
        )

    def _finalize_solution(self):
        if self.model.status in [
            GRB.OPTIMAL,
            GRB.TIME_LIMIT,
            GRB.SUBOPTIMAL,
            GRB.INTERRUPTED,
        ]:
            status_map = {
                GRB.OPTIMAL: "Optimal solution",
                GRB.TIME_LIMIT: "Time limit reached",
                GRB.SUBOPTIMAL: "Suboptimal solution",
                GRB.INTERRUPTED: "User interruption",
            }
            print(f"State: {status_map.get(self.model.status, 'Unknown status')}")

            if self.model.SolCount > 0:
                self.get_solution()
                self.post_process()
                self._validate_final_arrays()
                self._render_final_layout()
            else:
                print("No feasible solution found")
        else:
            print(f"Model solving failed, status code: {self.model.status}")

    def _validate_final_arrays(self):
        return validate_layout_connectivity(
            self.x_array,
            self.furniture_array,
            room_names=getattr(
                self,
                "room_name_list",
                [str(index) for index in range(self.room_num)],
            ),
            closed_room_indices=self.common_room_indices,
        )

    def _export_iis_if_infeasible(self):
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write(
                os.path.join(
                    self.output_dir,
                    f"{self.model_name}.ilp",
                )
            )

    def _add_orientation_case_vars(self, sigma_var, mu_var, name_prefix: str, enforce_sum: bool = True):
        z = self.model.addVars(4, vtype=GRB.BINARY, name=name_prefix)
        if enforce_sum:
            self.model.addConstr(quicksum(z) == 1, name=f"{name_prefix}_sum")

        self.model.addConstr(z[0] <= sigma_var, name=f"{name_prefix}_c0")
        self.model.addConstr(z[0] <= mu_var, name=f"{name_prefix}_c1")
        self.model.addConstr(z[0] >= sigma_var + mu_var - 1, name=f"{name_prefix}_c2")

        self.model.addConstr(z[1] <= sigma_var, name=f"{name_prefix}_c3")
        self.model.addConstr(z[1] <= 1 - mu_var, name=f"{name_prefix}_c4")
        self.model.addConstr(z[1] >= sigma_var - mu_var, name=f"{name_prefix}_c5")

        self.model.addConstr(z[2] <= 1 - sigma_var, name=f"{name_prefix}_c6")
        self.model.addConstr(z[2] <= mu_var, name=f"{name_prefix}_c7")
        self.model.addConstr(z[2] >= mu_var - sigma_var, name=f"{name_prefix}_c8")

        self.model.addConstr(z[3] <= 1 - sigma_var, name=f"{name_prefix}_c9")
        self.model.addConstr(z[3] <= 1 - mu_var, name=f"{name_prefix}_c10")
        self.model.addConstr(z[3] >= 1 - sigma_var - mu_var, name=f"{name_prefix}_c11")
        return z

    def set_objective_function(self):
        # Area Error Term
        # Create auxiliary variables to represent area errors.
        area_error = self.model.addVars(
            self.room_num,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.width * self.length,
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
                for neighbor_i, neighbor_j in (
                    (i, j + 1),
                    (i, j - 1),
                    (i + 1, j),
                    (i - 1, j),
                ):
                    if (
                        neighbor_i,
                        neighbor_j,
                    ) in self.valid_coordinates_set:
                        self.objective_function += (
                            self.weights["perimeter"]
                            * (
                                1
                                - self.x[
                                    k,
                                    neighbor_i,
                                    neighbor_j,
                                ]
                            )
                            * self.x[k, i, j]
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
            self.objective_function += self.weights["rectangularity"] * rect_diff_slack[k]
        
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

        self.room_objective_function = (
            self.objective_function
            - self._furniture_relation_objective
        )

        # Furniture Balance Term
        center_error = self.model.addVars(
            self.room_num, 2, lb=0, vtype=GRB.CONTINUOUS, name="center_error"
        )
        for k in range(self.room_num):
            # compute room center
            center_room_i = self.rect_min_i[k] + self.rect_len_i[k] / 2
            center_room_j = self.rect_min_j[k] + self.rect_len_j[k] / 2
            # compute furniture area weighted center:
            center_furniture_i = 0
            center_furniture_j = 0
            for l in range(self.furniture_num_list[k]):
                center_furniture_i += (
                    (
                        self.f_rect_min_i[k, l]
                        + ((1 - self.sigma[k, l]) * self.furniture_vertical_size[k][l] + self.sigma[k, l] * self.furniture_parallel_size[k][l]) / 2
                    )
                    * self.furniture_area_list[k][l]
                )
                center_furniture_j += (
                    (
                        self.f_rect_min_j[k, l]
                        + (self.sigma[k, l] * self.furniture_vertical_size[k][l] + (1 - self.sigma[k, l]) * self.furniture_parallel_size[k][l]) / 2
                    )
                    * self.furniture_area_list[k][l]
                )
            center_furniture_i /= sum(self.furniture_area_list[k][l] for l in range(self.furniture_num_list[k]))
            center_furniture_j /= sum(self.furniture_area_list[k][l] for l in range(self.furniture_num_list[k]))

            self.model.addConstr(
                center_error[k, 0] >= center_room_i - center_furniture_i
            )
            self.model.addConstr(
                center_error[k, 0] >= center_furniture_i - center_room_i
            )
            self.model.addConstr(
                center_error[k, 1] >= center_room_j - center_furniture_j
            )
            self.model.addConstr(
                center_error[k, 1] >= center_furniture_j - center_room_j
            )
            self.objective_function += self.weights["balance"] * (
                center_error[k, 0] + center_error[k, 1]
            )

    def _prepare_connected_start(
        self,
        reference_array,
        time_limit: float,
    ) -> float:
        """Build a valid incumbent without constraining the final solve."""
        started = time.monotonic()
        original_time_limit = self.model.Params.TimeLimit
        original_mip_focus = self.model.Params.MIPFocus
        original_model_sense = self.model.ModelSense
        original_objective = self.model.getObjective()
        original_variables = tuple(self.model.getVars())
        temporary_artifacts = []
        fixed_room_constraints = []
        mip_start_path = None

        def clear_mip_starts():
            self.model.NumStart = 0
            self.model.update()
            for variable in original_variables:
                variable.Start = GRB.UNDEFINED
            self.model.update()

        def remove_room_connectivity():
            constraints = [
                constraint
                for artifacts in temporary_artifacts
                for constraint in artifacts.constraints
            ]
            variables = [
                variable
                for artifacts in temporary_artifacts
                for variable in artifacts.variables
            ]
            if constraints:
                self.model.remove(constraints)
            if variables:
                self.model.remove(variables)
            self.model.update()
            temporary_artifacts.clear()

        def add_room_connectivity(room_roots, name_prefix):
            for room_index, room_root in enumerate(room_roots):
                temporary_artifacts.append(
                    add_fixed_root_connectivity(
                        self.model,
                        {
                            coordinate: self.x[
                                room_index,
                                *coordinate,
                            ]
                            for coordinate in self.valid_coordinates
                        },
                        self.valid_coordinates,
                        root_coordinate=room_root,
                        name=f"{name_prefix}_{room_index}",
                    )
                )
            self.model.update()

        try:
            reference = np.asarray(reference_array)
            if reference.ndim != 3 or reference.shape[0] != self.room_num:
                raise ValueError(
                    "Connected-start reference must have shape "
                    "(room_count, width, length)"
                )
            for i, j in self.valid_coordinates:
                if i >= reference.shape[1] or j >= reference.shape[2]:
                    raise ValueError(
                        "Connected-start reference does not cover "
                        f"valid coordinate {(i, j)}"
                    )

            clear_mip_starts()
            valid_coordinates = set(self.valid_coordinates)
            reference_roots = []
            for room_index in range(self.room_num):
                active_coordinates = {
                    coordinate
                    for coordinate in self.valid_coordinates
                    if reference[room_index, *coordinate] > 0.5
                }
                components = connected_components(
                    active_coordinates,
                    valid_coordinates,
                )
                if len(components) != 1:
                    raise ValueError(
                        "Connected-start reference room "
                        f"{room_index} has {len(components)} components"
                    )
                reference_roots.append(min(components[0]))
            add_room_connectivity(
                reference_roots,
                "room_seed_connectivity",
            )

            retained_reference_cells = []
            for room_index in range(self.room_num):
                for i, j in self.valid_coordinates:
                    start_value = (
                        1.0
                        if reference[room_index, i, j] > 0.5
                        else 0.0
                    )
                    self.x[room_index, i, j].Start = start_value
                    if start_value > 0.5:
                        retained_reference_cells.append(
                            self.x[room_index, i, j]
                        )
            if hasattr(self, "passage"):
                for i, j in self.valid_coordinates:
                    occupied = any(
                        reference[room_index, i, j] > 0.5
                        for room_index in range(self.room_num)
                    )
                    self.passage[i, j].Start = 0.0 if occupied else 1.0
            self.model.update()
            print(
                "Connected-start reference roots: "
                f"{reference_roots}"
            )

            self.model.setParam("MIPFocus", 1)
            reference_retention_objective = quicksum(
                1 - variable
                for variable in retained_reference_cells
            )
            self.model.setObjective(
                reference_retention_objective,
                GRB.MINIMIZE,
            )
            remaining = time_limit - (time.monotonic() - started)
            seed_time_limit = min(15.0, remaining)
            if seed_time_limit <= 0:
                print("Connected-start room solve has no time remaining.")
                return time.monotonic() - started

            self.model.Params.TimeLimit = seed_time_limit
            self.model.optimize()
            used_fallback = self.model.SolCount == 0
            if self.model.SolCount == 0:
                print(
                    "Connected-start reference could not be completed; "
                    "falling back to a full-model feasible seed."
                )
                remove_room_connectivity()
                self.model.reset()
                clear_mip_starts()
                self.model.setObjective(0)
                remaining = time_limit - (time.monotonic() - started)
                base_time_limit = min(60.0, max(0.0, remaining - 30.0))
                if base_time_limit <= 0:
                    print(
                        "Connected-start base solve has no time remaining."
                    )
                    return time.monotonic() - started
                self.model.Params.TimeLimit = base_time_limit
                self.model.optimize()
                if self.model.SolCount == 0:
                    print("Connected-start base solve found no incumbent.")
                    return time.monotonic() - started

                base_start = tuple(
                    (variable, variable.X)
                    for variable in original_variables
                )
                room_roots = []
                for room_index in range(self.room_num):
                    active_coordinates = {
                        coordinate
                        for coordinate in self.valid_coordinates
                        if self.x[room_index, *coordinate].X > 0.5
                    }
                    components = connected_components(
                        active_coordinates,
                        valid_coordinates,
                    )
                    room_roots.append(min(components[0]))
                add_room_connectivity(
                    room_roots,
                    "room_repair_connectivity",
                )
                for variable, value in base_start:
                    variable.Start = value
                self.model.update()
                print(
                    "Connected-start repair roots: "
                    f"{room_roots}"
                )

                remaining = time_limit - (time.monotonic() - started)
                cut_reserve = max(5.0, remaining * 0.25)
                room_time_limit = remaining - cut_reserve
                if room_time_limit <= 0:
                    print(
                        "Connected-start repair has no time remaining."
                    )
                    return time.monotonic() - started
                self.model.Params.TimeLimit = room_time_limit
                self.model.optimize()
                if self.model.SolCount == 0:
                    print(
                        "Connected-start room repair found no incumbent."
                    )
                    return time.monotonic() - started
            else:
                room_roots = reference_roots
                print("Connected-start reference completed successfully.")

            if used_fallback:
                remaining = time_limit - (time.monotonic() - started)
                refinement_reserve = max(30.0, remaining * 0.5)
                refinement_time_limit = min(
                    60.0,
                    max(0.0, remaining - refinement_reserve),
                )
                if refinement_time_limit > 0:
                    self.model.Params.TimeLimit = refinement_time_limit
                    self.model.setParam("MIPFocus", 2)
                    self.model.setObjective(
                        self.room_objective_function
                        + self.weights["reference"]
                        * reference_retention_objective,
                        GRB.MINIMIZE,
                    )
                    self.model.optimize()
                    if self.model.SolCount == 0:
                        print(
                            "Connected-start reference refinement "
                            "found no incumbent."
                        )
                        return time.monotonic() - started
                    print(
                        "Connected-start room refinement: "
                        f"seconds={refinement_time_limit:.1f} "
                        f"objective={self.model.ObjVal:.6f}"
                    )

            room_layout = {
                (room_index, i, j): round(
                    self.x[room_index, i, j].X
                )
                for room_index in range(self.room_num)
                for i, j in self.valid_coordinates
            }

            for room_index in range(self.room_num):
                for i, j in self.valid_coordinates:
                    fixed_room_constraints.append(
                        self.model.addConstr(
                            self.x[room_index, i, j]
                            == room_layout[room_index, i, j],
                            name=f"connected_start_fix_room_{room_index}_{i}_{j}",
                        )
                    )

            self.model.update()
            self.model.setParam("MIPFocus", 1)
            self.model.setObjective(0)
            remaining = time_limit - (time.monotonic() - started)
            if remaining <= 0:
                print("Connected-start furniture solve has no time remaining.")
                return time.monotonic() - started

            cut_loop_reserve = max(5.0, remaining * 0.25)
            furniture_time_limit = remaining - cut_loop_reserve
            if furniture_time_limit <= 0:
                print(
                    "Connected-start furniture solve cannot preserve "
                    "cut-loop time."
                )
                return time.monotonic() - started
            self.model.Params.TimeLimit = furniture_time_limit
            self.model.optimize()
            if self.model.SolCount == 0:
                print("Connected-start fixed-room solve found no incumbent.")
                return time.monotonic() - started

            remaining = time_limit - (time.monotonic() - started)
            if remaining <= 0:
                print("Connected-start cut loop has no time remaining.")
                return time.monotonic() - started
            cut_stats = run_connectivity_cut_loop(
                self.model,
                self._add_connectivity_cuts,
                time_limit=remaining,
            )
            self.get_solution()
            self._validate_final_arrays()
            descriptor, temporary_mip_start_path = tempfile.mkstemp(
                prefix="co_layout_connected_",
                suffix=".mst",
            )
            os.close(descriptor)
            os.unlink(temporary_mip_start_path)
            try:
                self.model.write(temporary_mip_start_path)
            except Exception:
                if os.path.exists(temporary_mip_start_path):
                    os.remove(temporary_mip_start_path)
                raise
            mip_start_path = temporary_mip_start_path
            print(
                "Connected start prepared: "
                f"room_roots={room_roots} "
                f"cut_rounds={cut_stats.rounds} "
                f"cuts={cut_stats.cuts_added}"
            )
        except ConnectivityOptimizationError as error:
            print(f"Connected-start preparation failed: {error}")
        finally:
            try:
                temporary_constraints = [
                    constraint
                    for artifacts in temporary_artifacts
                    for constraint in artifacts.constraints
                ]
                temporary_variables = [
                    variable
                    for artifacts in temporary_artifacts
                    for variable in artifacts.variables
                ]
                if fixed_room_constraints or temporary_constraints:
                    self.model.remove(
                        fixed_room_constraints + temporary_constraints
                    )
                if temporary_variables:
                    self.model.remove(temporary_variables)
                self.model.update()
                if mip_start_path is not None:
                    self.model.reset()
                    self.model.NumStart = 0
                    self.model.update()
                    try:
                        self.model.read(mip_start_path)
                    finally:
                        os.remove(mip_start_path)
            finally:
                self.model.setObjective(
                    original_objective,
                    original_model_sense,
                )
                self.model.Params.MIPFocus = original_mip_focus
                self.model.Params.TimeLimit = original_time_limit
                self.model.update()

        return time.monotonic() - started

    def optimize(self, reference_array = None, stage_num: int = 2):
        solve_started = time.monotonic()
        original_time_limit = self.model.Params.TimeLimit
        if reference_array is not None:
            #Apply Coarse-to-Fine Strategy
            mip_start_count = 0
            for k in range(self.room_num):
                for i, j in self.valid_coordinates:
                    start_val = (
                        1.0 if reference_array[k, i, j] > 0.5 else 0.0
                    )
                    self.x[k, i, j].Start = start_val
                    if start_val > 0.5:
                        mip_start_count += 1
            print(f"Set MIP Start for {mip_start_count} x variables.")
            penalty_count = 0
            for i, j in self.valid_coordinates:
                assigned_room = -1
                for k in range(self.room_num):
                    if reference_array[k, i, j] > 0.5:
                        assigned_room = k
                        break
                if assigned_room != -1:
                    self.objective_function += self.weights["reference"] * (
                        1 - self.x[assigned_room, i, j]
                    )
                    penalty_count += 1
            print(
                f"Added objective penalty terms for {penalty_count} grid cells based on coarse solution.\n"
            )

        if reference_array is not None and stage_num == 1:
            preparation_budget = min(
                original_time_limit * 0.6,
                max(0.0, original_time_limit - 1),
            )
            if preparation_budget > 0:
                self._prepare_connected_start(
                    reference_array,
                    preparation_budget,
                )

        elapsed = time.monotonic() - solve_started
        self.model.Params.TimeLimit = max(
            1.0,
            original_time_limit - elapsed,
        )
        self.model.Params.LazyConstraints = 1
        connectivity_callback = self._build_connectivity_callback(
            self._build_visualization_callback()
        )
        try:
            run_staged_optimization(
                self.model,
                self.model_name,
                self.objective_function,
                stage_num,
                self.output_dir,
                callback=connectivity_callback,
            )
        finally:
            self.model.Params.TimeLimit = original_time_limit
        connectivity_stats = self.model._connectivity_lazy_stats
        logger.info(
            "Fine lazy connectivity: callbacks=%d rejected=%d "
            "room_cuts=%d free_cuts=%d cuts=%d",
            connectivity_stats["mipsol_calls"],
            connectivity_stats["rejected_candidates"],
            connectivity_stats["room_cuts"],
            connectivity_stats["free_cuts"],
            connectivity_stats["cuts_added"],
        )
        print(
            "Fine lazy connectivity: "
            f"callbacks={connectivity_stats['mipsol_calls']} "
            f"rejected={connectivity_stats['rejected_candidates']} "
            f"room_cuts={connectivity_stats['room_cuts']} "
            f"free_cuts={connectivity_stats['free_cuts']} "
            f"cuts={connectivity_stats['cuts_added']}"
        )
        self._export_iis_if_infeasible()
        self._finalize_solution()

    def two_stage_optimize(self, reference_array):
        # only used for ablation test (comment room constraints before using this function!)
        for k in range(self.room_num):
            for (i, j) in self.valid_coordinates:
                if reference_array[k, i, j] > 0.5:
                    self.model.addConstr(
                        self.x[k, i, j] == 1
                    )
                else:
                    self.model.addConstr(
                        self.x[k, i, j] == 0
                    )
        self.model.Params.LazyConstraints = 1
        connectivity_callback = self._build_connectivity_callback(
            self._build_visualization_callback()
        )
        run_staged_optimization(
            self.model,
            self.model_name,
            self.objective_function,
            1,
            self.output_dir,
            callback=connectivity_callback,
        )
        connectivity_stats = self.model._connectivity_lazy_stats
        logger.info(
            "Fine lazy connectivity: callbacks=%d rejected=%d "
            "room_cuts=%d free_cuts=%d cuts=%d",
            connectivity_stats["mipsol_calls"],
            connectivity_stats["rejected_candidates"],
            connectivity_stats["room_cuts"],
            connectivity_stats["free_cuts"],
            connectivity_stats["cuts_added"],
        )
        print(
            "Fine lazy connectivity: "
            f"callbacks={connectivity_stats['mipsol_calls']} "
            f"rejected={connectivity_stats['rejected_candidates']} "
            f"room_cuts={connectivity_stats['room_cuts']} "
            f"free_cuts={connectivity_stats['free_cuts']} "
            f"cuts={connectivity_stats['cuts_added']}"
        )
        self._export_iis_if_infeasible()
        self._finalize_solution()

    def _add_connectivity_cuts(self, round_index: int) -> int:
        room_cuts_added = 0
        free_cuts_added = 0

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
                room_cuts_added += 1

        for room_index in self.common_room_indices:
            free_coordinates = {
                coordinate
                for coordinate in self.valid_coordinates
                if (
                    self.x[room_index, *coordinate].X
                    - sum(
                        self.furniture[
                            room_index,
                            furniture_index,
                            *coordinate,
                        ].X
                        for furniture_index in range(
                            self.furniture_num_list[room_index]
                        )
                    )
                    > 0.5
                )
            }
            components = connected_components(
                free_coordinates,
                self.valid_coordinates,
            )
            cuts = separator_cuts(components, self.valid_coordinates)
            for cut_index, cut in enumerate(cuts):
                boundary_free = quicksum(
                    self.x[room_index, *coordinate]
                    - quicksum(
                        self.furniture[
                            room_index,
                            furniture_index,
                            *coordinate,
                        ]
                        for furniture_index in range(
                            self.furniture_num_list[room_index]
                        )
                    )
                    for coordinate in sorted(cut.boundary)
                )
                source_free = (
                    self.x[room_index, *cut.source]
                    - quicksum(
                        self.furniture[
                            room_index,
                            furniture_index,
                            *cut.source,
                        ]
                        for furniture_index in range(
                            self.furniture_num_list[room_index]
                        )
                    )
                )
                target_free = (
                    self.x[room_index, *cut.target]
                    - quicksum(
                        self.furniture[
                            room_index,
                            furniture_index,
                            *cut.target,
                        ]
                        for furniture_index in range(
                            self.furniture_num_list[room_index]
                        )
                    )
                )
                self.model.addConstr(
                    boundary_free >= source_free + target_free - 1,
                    name=(
                        f"free_connectivity_r{round_index}"
                        f"_k{room_index}_c{cut_index}"
                    ),
                )
                free_cuts_added += 1

        logger.info(
            "Fine connectivity round %d: room_cuts=%d free_cuts=%d",
            round_index,
            room_cuts_added,
            free_cuts_added,
        )
        print(
            f"Fine connectivity round {round_index}: "
            f"room_cuts={room_cuts_added} "
            f"free_cuts={free_cuts_added}"
        )
        return room_cuts_added + free_cuts_added

    def get_solution(self):
        for i, j in self.valid_coordinates:
            self.passage_array[i, j] = self.passage[i, j].X
            for k in range(self.room_num):
                self.x_array[k, i, j] = self.x[k, i, j].X
                for l in range(self.furniture_num_list[k]):
                    self.furniture_array[k][l, i, j] = self.furniture[k, l, i, j].X
        for k in range(self.room_num):
            for l in range(self.furniture_num_list[k]):
                self.furniture_orientation_sigma_array[k][l] = self.sigma[k, l].X
                self.furniture_orientation_mu_array[k][l] = self.mu[k, l].X
            self.rect_len_i_array[k] = self.rect_len_i[k].X
            self.rect_len_j_array[k] = self.rect_len_j[k].X

        for k_room_idx, l_furn_idx in self.furniture_indices:
            self.f_rect_min_i_array[k_room_idx][l_furn_idx] = self.f_rect_min_i[
                k_room_idx, l_furn_idx
            ].X
            self.f_rect_min_j_array[k_room_idx][l_furn_idx] = self.f_rect_min_j[
                k_room_idx, l_furn_idx
            ].X

    def set_log(self):
        configure_model_log(self.model, self.model_name, self.output_dir)

    def set_constraints(self):
        self.add_room_nonempty_constraints()
        self.add_room_adjacent_constraints()
        self.add_room_bounding_box_constraints()
        self.add_room_at_corner_constraints()
        self.add_room_accessibility_constraints()
        self.add_flow_constraints()
        self.add_furniture_basic_constraints()
        self.add_furniture_relation_constraints()
        self.add_furniture_boundary_constraints()
        self.add_room_furniture_basic_constraints()

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

    def post_process(self):
        self.door_array = add_door(self)
        self.window_array = add_window(self)
        self.wall_array = add_wall(self)

        # check if door is blocked
        door_size = round(DOOR_SIZE / F)
        for door in self.door_array:
            dirc, facing, di, dj = door
            room_idx = None
            for k in range(self.room_num):
                if self.x_array[k, di, dj] > 0.5:
                    room_idx = k
                    break
            if room_idx is not None:
                blocked_furnitures = []
                i_dir = 1
                j_dir = 1
                if dirc == 1 or facing == 0:
                    j_dir = -1
                if dirc == 3 or facing == 2:
                    i_dir = -1
                for si in range(door_size):
                    for sj in range(door_size):
                        for l in range(self.furniture_num_list[k]):
                            if self.furniture_array[k][l, di + i_dir * si, dj + j_dir * sj] > 0.5:
                                blocked_furnitures.append(l)
                # Rearrange only when blocked furniture is detected.
                if blocked_furnitures:
                    print("WARNING: Detected unreasonable furniture arrangement, starting to attempt rearrangement.")
                    for l in blocked_furnitures:
                        rearrange(self, k, l)



    def add_room_furniture_basic_constraints(self):
        self.model.addConstrs(
            self.passage[i, j] + quicksum(self.x[k, i, j] for k in range(self.room_num))
            == 1
            for (i, j) in self.valid_coordinates
        )

        # entrance belongs to passage
        self.model.addConstr(
            self.passage[self.entrance[1], self.entrance[2]] == 1
        )

        # The furniture in the room should belong to this room.
        self.model.addConstrs(
            quicksum(
                self.furniture[k, l, i, j] for l in range(self.furniture_num_list[k])
            )
            <= self.x[k, i, j]
            for k in range(self.room_num)
            for (i, j) in self.valid_coordinates
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

    def add_flow_constraints(self):
        # Flow constraint, take the entrance as the source point, flow can only move between the corridor and open rooms.
        passage_or_openroom_expr = {}
        for i, j in self.valid_coordinates:
            passage_or_openroom_expr[(i, j)] = (self.passage[i, j] + quicksum(
                    self.x[k, i, j] - quicksum(
                        self.furniture[k, f, i, j] for f in range(self.furniture_num_list[k])
                    ) for k in self.open_room_indices)
            )
        for i, j in self.valid_coordinates:
            outflow_passage = LinExpr()
            inflow_passage = LinExpr()
            neighbor_coord_east = (i, j + 1)
            if neighbor_coord_east in self.valid_coordinates_set:
                outflow_passage += self.flow[EAST, i, j]
                inflow_passage += self.flow[WEST, i, j + 1]
                self.model.addConstr(
                    self.flow[EAST, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)]
                )
                self.model.addConstr(
                    self.flow[EAST, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_east]
                )

            neighbor_coord_west = (i, j - 1)
            if neighbor_coord_west in self.valid_coordinates_set:
                outflow_passage += self.flow[WEST, i, j]
                inflow_passage += self.flow[EAST, i, j - 1]
                self.model.addConstr(
                    self.flow[WEST, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)]
                )
                self.model.addConstr(
                    self.flow[WEST, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_west]
                )
            
            neighbor_coord_south = (i + 1, j)
            if neighbor_coord_south in self.valid_coordinates_set:
                outflow_passage += self.flow[SOUTH, i, j]
                inflow_passage += self.flow[NORTH, i + 1, j]
                self.model.addConstr(
                    self.flow[SOUTH, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)]
                )
                self.model.addConstr(
                    self.flow[SOUTH, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_south]
                )
            
            neighbor_coord_north = (i - 1, j)
            if neighbor_coord_north in self.valid_coordinates_set:
                outflow_passage += self.flow[NORTH, i, j]
                inflow_passage += self.flow[SOUTH, i - 1, j]
                self.model.addConstr(
                    self.flow[NORTH, i, j]
                    <= self.BigM_flow * passage_or_openroom_expr[(i, j)]
                )
                self.model.addConstr(
                    self.flow[NORTH, i, j]
                    <= self.BigM_flow
                    * passage_or_openroom_expr[neighbor_coord_north]
                )

            if (i, j) == (self.entrance[1], self.entrance[2]):
                self.model.addConstr(
                    outflow_passage - inflow_passage
                    == quicksum(passage_or_openroom_expr[p, q] for p, q in self.valid_coordinates)
                    - 1
                )
            else:
                self.model.addConstr(
                    outflow_passage - inflow_passage == -passage_or_openroom_expr[i, j]
                )

    def add_room_adjacent_constraints(self):
        add_room_adjacency_constraints(
            self.model,
            self.x,
            self.room_constraints,
            self.room_name_list,
            self.valid_coordinates,
        )

    def add_room_accessibility_constraints(self):
        D = round(DOOR_SIZE / F)
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
                    >= D,
                    name=f"room_access_{k}",
                )
                for i, j in self.valid_coordinates:
                    neighbors = LinExpr()
                    # Neighbors can be passage or open rooms
                    neighbors = (
                        self.passage[i + 1, j] + quicksum(
                            self.x[l, i + 1, j] - quicksum(
                                self.furniture[l, f, i + 1, j] for f in range(self.furniture_num_list[l])
                            ) for l in self.open_room_indices)
                    )  # South
                    neighbors += (
                        self.passage[i - 1, j] + quicksum(
                            self.x[l, i - 1, j] - quicksum(
                                self.furniture[l, f, i - 1, j] for f in range(self.furniture_num_list[l])
                            ) for l in self.open_room_indices)
                    )  # North
                    neighbors += (
                        self.passage[i, j + 1] + quicksum(
                            self.x[l, i, j + 1] - quicksum(
                                self.furniture[l, f, i, j + 1] for f in range(self.furniture_num_list[l])
                            ) for l in self.open_room_indices)
                    )  # East
                    neighbors += (
                        self.passage[i, j - 1] + quicksum(
                            self.x[l, i, j - 1] - quicksum(
                                self.furniture[l, f, i, j - 1] for f in range(self.furniture_num_list[l])
                            ) for l in self.open_room_indices)
                    )  # West

                    # If room_adj_passage_or_openroom[k,i,j]=1, then cell (i,j) must belong to room k
                    self.model.addConstr(
                        neighbors >= room_adj_passage_or_openroom[i, j],
                        name=f"room_access_neighbor_link_{k}_{i}_{j}",
                    )
                    self.model.addConstr(
                        room_adj_passage_or_openroom[i, j] <= self.x[k, i, j] - quicksum(
                            self.furniture[k, f, i, j] for f in range(self.furniture_num_list[k])
                        ),
                        name=f"room_access_room_link_{k}_{i}_{j}",
                    )

    def add_furniture_basic_constraints(self):
        """
        Basic constraints of furniture: size, shape, orientation
        """
        for k in range(self.room_num):
            # Furniture area constraint
            for l in range(self.furniture_num_list[k]):
                self.model.addConstr(
                    quicksum(
                        self.furniture[k, l, i, j]
                        for (i, j) in self.valid_coordinates
                    )
                    == self.furniture_area_list[k][l]
                )
                # Furniture shape constraint: rectangular
                self.model.addConstrs(
                    self.f_rect_min_i[k, l]
                    <= i + self.BigM_i * (1 - self.furniture[k, l, i, j])
                    for (i, j) in self.valid_coordinates
                )
                self.model.addConstrs(
                    self.f_rect_min_i[k, l]
                    + self.furniture_parallel_size[k][l] * self.sigma[k, l]
                    + self.furniture_vertical_size[k][l] * (1 - self.sigma[k, l])
                    - 1
                    >= i * self.furniture[k, l, i, j]
                    for (i, j) in self.valid_coordinates
                )
                self.model.addConstrs(
                    self.f_rect_min_j[k, l]
                    <= j + self.BigM_j * (1 - self.furniture[k, l, i, j])
                    for (i, j) in self.valid_coordinates
                )
                self.model.addConstrs(
                    self.f_rect_min_j[k, l]
                    + self.furniture_parallel_size[k][l] * (1 - self.sigma[k, l])
                    + self.furniture_vertical_size[k][l] * self.sigma[k, l]
                    - 1
                    >= j * self.furniture[k, l, i, j]
                    for (i, j) in self.valid_coordinates
                )
                # Furniture should not face the wall
                # Denote the four mutually exclusive cases by the auxiliary variable z
                # z_0 = sigma * mu              -> 1, 1
                # z_1 = sigma * (1 - mu)        -> 1, 0
                # z_2 = (1 - sigma) * mu        -> 0, 1
                # z_3 = (1 - sigma) * (1 - mu)  -> 0, 0
                z = self._add_orientation_case_vars(
                    self.sigma[k, l],
                    self.mu[k, l],
                    f"furniture_orient_aux_{k}_{l}",
                    enforce_sum=True,
                )
                for i, j in self.valid_coordinates:
                    # Case z_0
                    self.model.addConstr(
                        self.x[k, i - 1, j]
                        >= self.furniture[k, l, i, j]
                        - self.BigM_binary * (1 - z[0])
                    )
                    # Case z_1
                    self.model.addConstr(
                        self.x[k, i + 1, j]
                        >= self.furniture[k, l, i, j]
                        - self.BigM_binary * (1 - z[1])
                    )
                    # Case z_2
                    self.model.addConstr(
                        self.x[k, i, j - 1]
                        >= self.furniture[k, l, i, j]
                        - self.BigM_binary * (1 - z[2])
                    )
                    # Case z_3
                    self.model.addConstr(
                        self.x[k, i, j + 1]
                        >= self.furniture[k, l, i, j]
                        - self.BigM_binary * (1 - z[3])
                    )

    def add_furniture_relation_constraints(self):
        # Add the objective function for the relative distance between furniture.
        for room in self.room_name_list:
            k = self.room_name_list.index(room)
            furniture_dist_pairs = self.furniture_constraints[room]["distance_constraints"]
            for pair in furniture_dist_pairs:
                name1, name2, d1, d2 = pair
                l1 = self.furniture_name_list[k].index(name1) if name1 in self.furniture_name_list[k] else None
                l2 = self.furniture_name_list[k].index(name2) if name2 in self.furniture_name_list[k] else None
                if l1 is not None and l2 is not None:
                    # set objective function
                    dist_error_1 = self.model.addVar(
                        vtype=GRB.CONTINUOUS, name=f"dist_error_1_{k}_{l1}_{l2}"
                    )
                    dist_error_2 = self.model.addVar(
                        vtype=GRB.CONTINUOUS, name=f"dist_error_2_{k}_{l1}_{l2}"
                    )
                    # It requires very complex derivations, which are omitted here.
                    # Auxiliary variables for distance errors
                    # Denote the four mutually exclusive cases by the auxiliary variable z
                    # z_0 = sigma_2 * mu_2               -> 1, 1
                    # z_1 = sigma_2 * (1 - mu_2)         -> 1, 0
                    # z_2 = (1 - sigma_2) * mu_2         -> 0, 1
                    # z_3 = (1 - sigma_2) * (1 - mu_2)   -> 0, 0
                    z = self._add_orientation_case_vars(
                        self.sigma[k, l2],
                        self.mu[k, l2],
                        f"dist_error_aux_{k}_{l1}_{l2}",
                        enforce_sum=False,
                    )

                    # Case z_0
                    self.model.addConstr(
                        dist_error_1 >= self.f_rect_min_i[k, l2] + self.furniture_parallel_size[k][l2] / 2 - self.f_rect_min_i[k, l1] - 
                        ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2 - d1
                        - self.BigM * (1 - z[0])
                    )
                    self.model.addConstr(
                        dist_error_1 >= -self.f_rect_min_i[k, l2] - self.furniture_parallel_size[k][l2] / 2 + self.f_rect_min_i[k, l1] + 
                        ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2 + d1
                        - self.BigM * (1 - z[0])
                    )
                    self.model.addConstr(
                        dist_error_2 >= self.f_rect_min_j[k, l1] + (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2
                        - self.f_rect_min_j[k, l2] - self.furniture_vertical_size[k][l2] / 2 - d2
                        - self.BigM * (1 - z[0])
                    )
                    self.model.addConstr(
                        dist_error_2 >= -self.f_rect_min_j[k, l1] - (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2
                        + self.f_rect_min_j[k, l2] + self.furniture_vertical_size[k][l2] / 2 + d2
                        - self.BigM * (1 - z[0])
                    )
                    # Case z_1
                    self.model.addConstr(
                        dist_error_1 >= self.f_rect_min_i[k, l1] + ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2
                        - self.f_rect_min_i[k, l2] - self.furniture_parallel_size[k][l2] / 2 - d1
                        - self.BigM * (1 - z[1])
                    )
                    self.model.addConstr(
                        dist_error_1 >= -self.f_rect_min_i[k, l1] + ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] - self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2
                        + self.f_rect_min_i[k, l2] + self.furniture_parallel_size[k][l2] / 2 + d1
                        - self.BigM * (1 - z[1])
                    )
                    self.model.addConstr(
                        dist_error_2 >= self.f_rect_min_j[k, l2] + self.furniture_vertical_size[k][l2] / 2 - self.f_rect_min_j[k, l1] - 
                        (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2 - d2
                        - self.BigM * (1 - z[1])
                    )
                    self.model.addConstr(
                        dist_error_2 >= -self.f_rect_min_j[k, l2] - self.furniture_vertical_size[k][l2] / 2 + self.f_rect_min_j[k, l1] + 
                        (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2 + d2
                        - self.BigM * (1 - z[1])
                    )
                    # Case z_2
                    self.model.addConstr(
                        dist_error_1 >= self.f_rect_min_j[k, l2] + self.furniture_parallel_size[k][l2] / 2 - self.f_rect_min_j[k, l1] - 
                        (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2 - d1
                        - self.BigM * (1 - z[2])
                    )
                    self.model.addConstr(
                        dist_error_1 >= -self.f_rect_min_j[k, l2] - self.furniture_parallel_size[k][l2] / 2 + self.f_rect_min_j[k, l1] + 
                        (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2 + d1
                        - self.BigM * (1 - z[2])
                    )
                    self.model.addConstr(
                        dist_error_2 >= self.f_rect_min_i[k, l2] + self.furniture_vertical_size[k][l2] / 2 - self.f_rect_min_i[k, l1] - 
                        ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2 - d2
                        - self.BigM * (1 - z[2])
                    )
                    self.model.addConstr(
                        dist_error_2 >= -self.f_rect_min_i[k, l2] - self.furniture_vertical_size[k][l2] / 2 + self.f_rect_min_i[k, l1] + 
                        ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2 + d2
                        - self.BigM * (1 - z[2])
                    )
                    # Case z_3
                    self.model.addConstr(
                        dist_error_1 >= self.f_rect_min_j[k, l1] + (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2
                        - self.f_rect_min_j[k, l2] - self.furniture_parallel_size[k][l2] / 2 - d1
                        - self.BigM * (1 - z[3])
                    )
                    self.model.addConstr(
                        dist_error_1 >= -self.f_rect_min_j[k, l1] - (self.sigma[k, l1]*self.furniture_vertical_size[k][l1] + (1 - self.sigma[k, l1])*self.furniture_parallel_size[k][l1]) / 2
                        + self.f_rect_min_j[k, l2] + self.furniture_parallel_size[k][l2] / 2 + d1
                        - self.BigM * (1 - z[3])
                    )
                    self.model.addConstr(
                        dist_error_2 >= self.f_rect_min_i[k, l1] + ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2
                        - self.f_rect_min_i[k, l2] - self.furniture_vertical_size[k][l2] / 2 - d2
                        - self.BigM * (1 - z[3])
                    )
                    self.model.addConstr(
                        dist_error_2 >= -self.f_rect_min_i[k, l1] - ((1 - self.sigma[k, l1])*self.furniture_vertical_size[k][l1] + self.sigma[k, l1]*self.furniture_parallel_size[k][l1]) / 2
                        + self.f_rect_min_i[k, l2] + self.furniture_vertical_size[k][l2] / 2 + d2
                        - self.BigM * (1 - z[3])
                    )


                    # self.model.addConstr(
                    #     dist_error_1 >= (1 - 2 * self.mu[k, l2]) * (
                    #         self.sigma[k, l2] * (self.f_rect_min_i[k, l1] - self.f_rect_min_i[k, l2])
                    #         + (1 - self.sigma[k, l2]) * (self.f_rect_min_j[k, l1] - self.f_rect_min_j[k, l2])
                    #     ) - (1 - self.mu[k, l2]) * self.furniture_parallel_size[k][l2] - self.mu[k, l2] * (
                    #         (self.sigma[k, l1] + self.sigma[k, l2] - 2 * self.sigma[k, l1] * self.sigma[k, l2]) * self.furniture_vertical_size[k][l1]
                    #         + (1 - self.sigma[k, l1] - self.sigma[k, l2] + 2 * self.sigma[k, l1] * self.sigma[k, l2]) * self.furniture_parallel_size[k][l1]
                    #     ) - d1
                    # )
                    # self.model.addConstr(
                    #     dist_error_1 >= -(1 - 2 * self.mu[k, l2]) * (
                    #         self.sigma[k, l2] * (self.f_rect_min_i[k, l1] - self.f_rect_min_i[k, l2])
                    #         + (1 - self.sigma[k, l2]) * (self.f_rect_min_j[k, l1] - self.f_rect_min_j[k, l2])
                    #     ) + (1 - self.mu[k, l2]) * self.furniture_parallel_size[k][l2] + self.mu[k, l2] * (
                    #         (self.sigma[k, l1] + self.sigma[k, l2] - 2 * self.sigma[k, l1] * self.sigma[k, l2]) * self.furniture_vertical_size[k][l1]
                    #         + (1 - self.sigma[k, l1] - self.sigma[k, l2] + 2 * self.sigma[k, l1] * self.sigma[k, l2]) * self.furniture_parallel_size[k][l1]
                    #     ) + d1
                    # )
                    # self.model.addConstr(
                    #     dist_error_2 >= (1 - 2 * self.mu[k, l2]) * (
                    #         self.sigma[k, l2] * (self.f_rect_min_j[k, l2] - self.f_rect_min_j[k, l1])
                    #         + (1 - self.sigma[k, l2]) * (self.f_rect_min_i[k, l1] - self.f_rect_min_i[k, l2])
                    #     ) - (1 - self.sigma[k, l2] - self.mu[k , l2] + 2 * self.sigma[k, l2] * self.mu[k, l2]) * self.furniture_vertical_size[k][l2]
                    #     - (1 - self.sigma[k, l2]) * self.mu[k, l2] * (
                    #         (1 - self.sigma[k, l1]) * self.furniture_vertical_size[k][l1] + self.sigma[k, l1] * self.furniture_parallel_size[k][l1])
                    #     - (1 - self.mu[k, l2]) * self.sigma[k, l2] * (
                    #         self.sigma[k, l1] * self.furniture_vertical_size[k][l1] + (1- self.sigma[k, l1]) * self.furniture_parallel_size[k][l1])
                    #     -d2
                    # )
                    # self.model.addConstr(
                    #     dist_error_2 >= -(1 - 2 * self.mu[k, l2]) * (
                    #         self.sigma[k, l2] * (self.f_rect_min_j[k, l2] - self.f_rect_min_j[k, l1])
                    #         + (1 - self.sigma[k, l2]) * (self.f_rect_min_i[k, l1] - self.f_rect_min_i[k, l2])
                    #     ) + (1 - self.sigma[k, l2] - self.mu[k , l2] + 2 * self.sigma[k, l2] * self.mu[k, l2]) * self.furniture_vertical_size[k][l2]
                    #     + (1 - self.sigma[k, l2]) * self.mu[k, l2] * (
                    #         (1 - self.sigma[k, l1]) * self.furniture_vertical_size[k][l1] + self.sigma[k, l1] * self.furniture_parallel_size[k][l1])
                    #     + (1 - self.mu[k, l2]) * self.sigma[k, l2] * (
                    #         self.sigma[k, l1] * self.furniture_vertical_size[k][l1] + (1- self.sigma[k, l1]) * self.furniture_parallel_size[k][l1])
                    #     +d2
                    # )
                    self.objective_function += self.weights["distance"] * (dist_error_1 + dist_error_2)
                    # if -0.1 < d1 < 0.1:
                    #     if d2 > 0:
                    #         self.objective_function += self.weights["distance"] * (self.sigma[k, l1] + 1 - self.mu[k, l1])
                    #     else:
                    #         self.objective_function += self.weights["distance"] * (self.sigma[k, l1] + self.mu[k, l1])

                    # if -0.1 < d2 < 0.1:
                    #     if d1 > 0:
                    #         self.objective_function += self.weights["distance"] * (1 - self.sigma[k, l1] + self.mu[k, l1])
                    #     else:
                    #         self.objective_function += self.weights["distance"] * (1 - self.sigma[k, l1] + 1 - self.mu[k, l1])
            # Align Constraints
            furniture_align_pairs = self.furniture_constraints[room]["alignment_constraints"]
            for pair in furniture_align_pairs:
                l1 = self.furniture_name_list[k].index(pair[0]) if pair[0] in self.furniture_name_list[k] else None
                l2 = self.furniture_name_list[k].index(pair[1]) if pair[1] in self.furniture_name_list[k] else None
                if l1 is not None and l2 is not None:
                    self.model.addConstr(
                        self.sigma[k, l1] == self.sigma[k, l2]
                    )
                    # if pair[2] == 1:
                    #     self.model.addConstr(self.mu[k, l1] == self.mu[k, l2])
                    #     # self.objective_function += self.weights['distance'] * (self.mu[k, l1] * self.mu[k, l2] + 
                    #     #                            (1 - self.mu[k, l1]) * (1 - self.mu[k, l2]))
                    # else:
                    #     self.model.addConstr(self.mu[k, l1] + self.mu[k, l2] == 1)
                    #     #self.objective_function += self.weights['distance'] * (self.mu[k, l1] * (1 - self.mu[k, l2]) + 
                    #     #                            (1 - self.mu[k, l1]) * self.mu[k, l2])
                    # align_error = self.model.addVar(
                    #     vtype=GRB.CONTINUOUS, name=f"align_error_1_{k}_{l1}_{l2}"
                    # )
                    # self.model.addConstr(
                    #     align_error >= self.f_rect_min_i[k, l1] + self.furniture_vertical_size[k][l1] / 2 -
                    #     self.f_rect_min_i[k, l2] - self.furniture_vertical_size[k][l2] / 2 - self.BigM * self.sigma[k, l1]
                    # )
                    # self.model.addConstr(
                    #     align_error >= self.f_rect_min_i[k, l2] + self.furniture_vertical_size[k][l2] / 2 -
                    #     self.f_rect_min_i[k, l1] - self.furniture_vertical_size[k][l1] / 2 - self.BigM * self.sigma[k, l1]
                    # )
                    # self.model.addConstr(
                    #     align_error >= self.f_rect_min_j[k, l1] + self.furniture_parallel_size[k][l1] / 2 -
                    #     self.f_rect_min_j[k, l2] - self.furniture_parallel_size[k][l2] / 2 - self.BigM * (1 - self.sigma[k, l1])
                    # )
                    # self.model.addConstr(
                    #     align_error >= self.f_rect_min_j[k, l2] + self.furniture_parallel_size[k][l2] / 2 -
                    #     self.f_rect_min_j[k, l1] - self.furniture_parallel_size[k][l1] / 2 - self.BigM * (1 - self.sigma[k, l1])
                    # )
                    # self.objective_function += self.weights["distance"] * align_error
            # Facing Constraints
            furniture_facing_pairs = self.furniture_constraints[room]["facing_constraints"]
            for pair in furniture_facing_pairs:
                l1 = self.furniture_name_list[k].index(pair[0]) if pair[0] in self.furniture_name_list[k] else None
                l2 = self.furniture_name_list[k].index(pair[1]) if pair[1] in self.furniture_name_list[k] else None
                if l1 is not None and l2 is not None:
                    # Denote the four mutually exclusive cases by the auxiliary variable z
                    # z_0 = sigma * mu              -> 1, 1
                    # z_1 = sigma * (1 - mu)        -> 1, 0
                    # z_2 = (1 - sigma) * mu        -> 0, 1
                    # z_3 = (1 - sigma) * (1 - mu)  -> 0, 0
                    z = self._add_orientation_case_vars(
                        self.sigma[k, l1],
                        self.mu[k, l1],
                        f"facing_aux_{k}_{l1}_{l2}",
                        enforce_sum=True,
                    )
                    # Case z_0
                    self.model.addConstr(
                        self.f_rect_min_i[k, l1] - 1
                        >= self.f_rect_min_i[k, l2]
                        - self.BigM_i * (1 - z[0])
                    )
                    # Case z_1
                    self.model.addConstr(
                        self.f_rect_min_i[k, l1] + 1
                        <= self.f_rect_min_i[k, l2]
                        + self.BigM_i * (1 - z[1])
                    )
                    # Case z_2
                    self.model.addConstr(
                        self.f_rect_min_j[k, l1] - 1
                        >= self.f_rect_min_j[k, l2]
                        - self.BigM_j * (1 - z[2])
                    )
                    # Case z_3
                    self.model.addConstr(
                        self.f_rect_min_j[k, l1] + 1
                        <= self.f_rect_min_j[k, l2]
                        + self.BigM_j * (1 - z[3])
                    )


    def add_furniture_boundary_constraints(self):
        """
        The furniture is on the boundary of the room.
        """
        for room in self.room_name_list:
            boundary_items = self.furniture_constraints[room]["boundary_items"]
            k = self.room_name_list.index(room)
            for item in boundary_items:
                l = self.furniture_name_list[k].index(item) if item in self.furniture_name_list[k] else None
                # Just need the furniture to be adjacent to a "block that doesn't belong to the room."
                if l is not None:
                    # Creating auxiliary variables
                    furniture_boundary = self.model.addVars(
                        self.valid_coordinates,
                        vtype=GRB.BINARY,
                        name=f"furniture_boundary_{k}_{l}",
                    )
                    self.model.addConstr(
                        quicksum(
                            furniture_boundary[i, j]
                            for i, j in self.valid_coordinates
                        )
                        == self.furniture_vertical_size[k][l],
                        name=f"furniture_boundary_at_least_{k}_{l}",
                    )
                    for i, j in self.valid_coordinates:
                        neighbors = QuadExpr()
                        neighbors += (1 - self.x[k, i - 1, j]) * self.sigma[k, l]
                        neighbors += (1 - self.x[k, i + 1, j]) * self.sigma[k, l]
                        neighbors += (1 - self.x[k, i, j - 1]) * (1 - self.sigma[k, l])
                        neighbors += (1 - self.x[k, i, j + 1]) * (1 - self.sigma[k, l])
                        self.model.addConstr(
                            neighbors >= furniture_boundary[i, j],
                            name=f"furniture_boundary_neighbor_link_{k}_{l}_{i}_{j}",
                        )
                        self.model.addConstr(
                            furniture_boundary[i, j] <= self.furniture[k, l, i, j],
                            name=f"furniture_boundary_link_{k}_{l}_{i}_{j}",
                        )
