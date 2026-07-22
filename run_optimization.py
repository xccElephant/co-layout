import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_PATH = str(Path(__file__).parent)
sys.path.append(ROOT_PATH)

from constants import *
from optimization.coopt_model import CooptModel
from optimization.floorplan_model import FloorplanModel
from utils.find_boundary import (calculate_boundary_direction,
                                 find_boundary_coordinates)
from utils.pre_process import extract_data


def _build_outdoor_coordinates(params):
    outdoor_coordinates = []
    scale = params["scale"]
    scaled_width = params["width"]
    scaled_length = params["length"]
    for slice in params["outdoor_space"]:
        width_range, length_range = slice.split(",")
        width_range = width_range.split(":")
        width_range = [
            max(0, int(float(width_range[0]) / scale)),
            min(scaled_width, int(float(width_range[1]) / scale)),
        ]
        length_range = length_range.split(":")
        length_range = [
            max(0, int(float(length_range[0]) / scale)),
            min(scaled_length, int(float(length_range[1]) / scale)),
        ]
        for i in range(width_range[0], width_range[1]):
            for j in range(length_range[0], length_range[1]):
                if i >= 0 and i < scaled_width and j >= 0 and j < scaled_length:
                    outdoor_coordinates.append((i, j))
    return list(set(outdoor_coordinates))


def _project_entrance_to_boundary(params):
    scaled_width = params["width"]
    scaled_length = params["length"]
    outdoor_coordinates = params["outdoor_space"]
    scale = params["scale"]
    
    boundary_coordinates = find_boundary_coordinates(
        scaled_width,
        scaled_length,
        outdoor_coordinates,
    )
    entrance_coordinates = (
        int(params["entrance_location"][0] / scale),
        int(params["entrance_location"][1] / scale),
    )
    min_dis = float("inf")
    nearest_boundary_coordinate = None
    for coord in boundary_coordinates:
        dist = (coord[0] - entrance_coordinates[0]) ** 2 + (
            coord[1] - entrance_coordinates[1]
        ) ** 2
        if dist < min_dis:
            min_dis = dist
            nearest_boundary_coordinate = coord
    entrance_direction = calculate_boundary_direction(
        scaled_width,
        scaled_length,
        outdoor_coordinates,
        nearest_boundary_coordinate,
    )
    return [
        entrance_direction,
        nearest_boundary_coordinate[0],
        nearest_boundary_coordinate[1],
    ]


def synthesis(session_id):
    json_file = PATH_OF_SESSIONS / session_id / "memory.json"
        
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    building_params, rooms, furniture, room_constraints, furniture_constraints = (
        extract_data(json_data)
    )

    # Coarse Model
    building_params_coarse = copy.deepcopy(building_params)
    rooms_coarse = copy.deepcopy(rooms)
    building_params_coarse["width"] = round(building_params["width"] / C + 0.001)  # avoid floating point precision issues
    building_params_coarse["length"] = round(building_params["length"] / C + 0.001)
    building_params_coarse["scale"] = C

    # process outdoor coordinates
    outdoor_coordinates = _build_outdoor_coordinates(building_params_coarse)
    building_params_coarse["outdoor_space"] = outdoor_coordinates

    # Since the entrance given by LLM may not be on the boundary, we need to project it to the nearest boundary
    building_params_coarse["entrance_location"] = _project_entrance_to_boundary(building_params_coarse)

    for _, room in rooms_coarse.items():
        room["area"] = round(room["area"] / C / C)

    coarse_model = FloorplanModel(
        "Coarse", session_id, building_params_coarse, rooms_coarse, room_constraints
    )
    coarse_model.optimize()
    coarse_x_array = coarse_model.get_solutions()

    # Fine Model
    building_params_fine = copy.deepcopy(building_params)
    rooms_fine = copy.deepcopy(rooms)
    building_params_fine["width"] = round(building_params["width"] / F)
    building_params_fine["length"] = round(building_params["length"] / F)
    building_params_fine["scale"] = F

    for _, room in rooms_fine.items():
        room["area"] = round(room["area"] / F / F)

    # process outdoor coordinates
    outdoor_coordinates = _build_outdoor_coordinates(building_params_fine)
    building_params_fine["outdoor_space"] = outdoor_coordinates

    # process entrance
    # Since the entrance given by LLM may not be on the boundary, we need to project it to the nearest boundary
    building_params_fine["entrance_location"] = _project_entrance_to_boundary(building_params_fine)

    for _, furniture_list in furniture.items():
        for f in furniture_list:
            f["original_width"] = f["width"] / F
            f["original_length"] = f["length"] / F
            f["width"] = round(f["width"] / F) if f["width"] / F > 0.5 else 1  # avoid rounding to zero
            f["length"] = round(f["length"] / F) if f["length"] / F > 0.5 else 1

    for _, constraints in furniture_constraints.items():
        constraints["distance_constraints"] = [
            (item1, item2, dis1 / F, dis2 / F)
            for item1, item2, dis1, dis2 in constraints["distance_constraints"]
        ]

    # coarse to fine mapping
    fine_x_array = np.kron(coarse_x_array, np.ones((N, N), dtype=int))

    fine_model = CooptModel(
        "Fine",
        session_id,
        building_params_fine,
        rooms_fine,
        room_constraints,
        furniture,
        furniture_constraints,
    )
    fine_model.optimize(fine_x_array)
    # fine_model.two_stage_optimize(fine_x_array)  # only for ablation test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the grid-based optimization pipeline.")
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        required=True,
        help="Session ID generated by the agents pipeline (e.g. 2026-02-06_18-03-50_yHPGOu)",
    )
    args = parser.parse_args()

    start_time = time.time()
    synthesis(args.session)
    end_time = time.time()
    print(f"Synthesis completed in {end_time - start_time} seconds.")
