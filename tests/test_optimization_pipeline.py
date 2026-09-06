import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_optimization


class OptimizationPipelineTest(unittest.TestCase):
    def test_coarse_falls_back_when_strict_grid_has_no_layout(self):
        attempts = []
        disposed = []

        class FallbackFloorplanModel:
            class Params:
                TimeLimit = None

            class ModelState:
                def __init__(self, has_solution, attempt):
                    self.Params = FallbackFloorplanModel.Params()
                    self.SolCount = 1 if has_solution else 0
                    self.attempt = attempt

                def dispose(self):
                    disposed.append(self.attempt)

            def __init__(self, *args, **kwargs):
                defer_closed = kwargs["defer_closed_room_adjacency"]
                attempts.append(defer_closed)
                self.model = self.ModelState(
                    has_solution=defer_closed,
                    attempt=defer_closed,
                )

            def optimize(self):
                pass

            def get_solutions(self):
                return np.array([[[1]]])

        with patch.object(
            run_optimization,
            "FloorplanModel",
            FallbackFloorplanModel,
        ):
            solution = run_optimization._solve_coarse_layout(
                "fallback",
                {},
                {},
                {},
            )

        self.assertEqual(attempts, [False, True])
        self.assertEqual(disposed, [False, True])
        np.testing.assert_array_equal(solution, np.array([[[1]]]))

    def test_raises_when_both_coarse_attempts_find_no_layout(self):
        class EmptyFloorplanModel:
            class Params:
                TimeLimit = None

            class ModelState:
                def __init__(self):
                    self.Params = EmptyFloorplanModel.Params()
                    self.SolCount = 0

                def dispose(self):
                    pass

            def __init__(self, *args, **kwargs):
                self.model = self.ModelState()

            def optimize(self, stage_num=2, reference_array=None):
                pass

            def get_solutions(self):
                raise AssertionError(
                    "get_solutions must not run without an incumbent"
                )

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
        extracted = (
            building_params,
            rooms,
            {"Room": []},
            {"room_adjacency": [], "corner_rooms": {}},
            {
                "Room": {
                    "boundary_items": [],
                    "distance_constraints": [],
                    "alignment_constraints": [],
                    "facing_constraints": [],
                }
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            sessions_path = Path(temporary_directory)
            session_path = sessions_path / "no-coarse-layout"
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
                    return_value=extracted,
                ),
                patch.object(
                    run_optimization,
                    "FloorplanModel",
                    EmptyFloorplanModel,
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
                self.assertRaisesRegex(
                    RuntimeError,
                    "Coarse",
                ),
            ):
                run_optimization.synthesis("no-coarse-layout")

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
        self.assertFalse(
            calls["Coarse_init_kwargs"][
                "defer_closed_room_adjacency"
            ]
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
