import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_optimization


class OptimizationPipelineTest(unittest.TestCase):
    def test_room_refinement_feeds_single_stage_fine_optimization(self):
        calls = {}

        class RecordingFloorplanModel:
            class Params:
                TimeLimit = None

            class ModelState:
                def __init__(self, name):
                    self.Params = RecordingFloorplanModel.Params()
                    self.SolCount = 1
                    self.name = name

                def dispose(self):
                    calls[f"{self.name}_disposed"] = True

            def __init__(self, *args, **kwargs):
                self.name = args[0]
                self.model = self.ModelState(self.name)
                calls[f"{self.name}_init_kwargs"] = kwargs

            def optimize(self, stage_num=2, reference_array=None):
                calls[f"{self.name}_optimize"] = {
                    "stage_num": stage_num,
                    "reference_array": reference_array,
                    "time_limit": self.model.Params.TimeLimit,
                }

            def get_solutions(self):
                if self.name == "FineRoom":
                    return np.ones((1, 4, 4))
                return np.ones((1, 2, 2))

        class RecordingFineModel:
            def __init__(self, *args):
                calls["fine_init"] = args

            def optimize(self, reference_array=None, stage_num=2):
                calls["fine_reference"] = reference_array
                calls["fine_stage_num"] = stage_num

        building_params = {
            "width": 4,
            "length": 4,
            "entrance_location": [0, 0],
            "outdoor_space": [],
        }
        rooms = {
            "Room": {
                "area": 12,
                "is_open": True,
                "privacy_level": 1,
            }
        }
        furniture = {"Room": []}
        room_constraints = {
            "room_adjacency": [],
            "corner_rooms": {},
        }
        furniture_constraints = {
            "Room": {
                "boundary_items": [],
                "distance_constraints": [],
                "alignment_constraints": [],
                "facing_constraints": [],
            }
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            sessions_path = Path(temporary_directory)
            session_id = "single-stage"
            session_path = sessions_path / session_id
            session_path.mkdir()
            (session_path / "memory.json").write_text(
                json.dumps({}),
                encoding="utf-8",
            )

            with (
                patch.object(
                    run_optimization,
                    "PATH_OF_SESSIONS",
                    sessions_path,
                ),
                patch.object(
                    run_optimization,
                    "extract_data",
                    return_value=(
                        building_params,
                        rooms,
                        furniture,
                        room_constraints,
                        furniture_constraints,
                    ),
                ),
                patch.object(
                    run_optimization,
                    "FloorplanModel",
                    RecordingFloorplanModel,
                ),
                patch.object(
                    run_optimization,
                    "CooptModel",
                    RecordingFineModel,
                ),
                patch.object(
                    run_optimization,
                    "_build_outdoor_coordinates",
                    return_value=[],
                ),
                patch.object(
                    run_optimization,
                    "_project_entrance_to_boundary",
                    return_value=["North", 0, 0],
                ),
            ):
                run_optimization.synthesis(session_id)

        self.assertEqual(
            calls["Coarse_optimize"]["time_limit"],
            30,
        )
        self.assertEqual(
            calls["FineRoom_optimize"]["reference_array"].shape,
            (1, 4, 4),
        )
        self.assertFalse(
            calls["FineRoom_init_kwargs"][
                "defer_closed_room_adjacency"
            ]
        )
        self.assertTrue(calls["Coarse_disposed"])
        self.assertTrue(calls["FineRoom_disposed"])
        self.assertEqual(calls["fine_stage_num"], 1)
        self.assertEqual(calls["fine_reference"].shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
