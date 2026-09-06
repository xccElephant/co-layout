import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from gurobipy import GRB, Env, Model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from optimization.coopt_model import CooptModel
from optimization.grid_model import FurnitureGrid, RoomGrid
from optimization.model_utils import add_fixed_root_connectivity


class FixedRootConnectivityTest(unittest.TestCase):
    def make_model(self, name):
        environment = Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
        model = Model(name, env=environment)
        model.Params.OutputFlag = 0
        self.addCleanup(environment.dispose)
        self.addCleanup(model.dispose)
        return model

    def test_forced_endpoints_select_the_connecting_cell(self):
        model = self.make_model("fixed_root_connectivity")
        coordinates = [(0, 0), (0, 1), (0, 2)]
        selected = model.addVars(
            coordinates,
            vtype=GRB.BINARY,
            name="selected",
        )
        model.addConstr(selected[0, 0] == 1)
        model.addConstr(selected[0, 2] == 1)
        model.setObjective(selected[0, 1], GRB.MINIMIZE)

        add_fixed_root_connectivity(
            model,
            {
                coordinate: selected[coordinate]
                for coordinate in coordinates
            },
            coordinates,
            root_coordinate=(0, 0),
            name="selected_connectivity",
        )
        model.optimize()

        self.assertEqual(model.Status, GRB.OPTIMAL)
        self.assertGreater(selected[0, 1].X, 0.5)

    def test_fixed_root_must_be_active(self):
        model = self.make_model("fixed_root_active")
        coordinates = [(0, 0), (0, 1)]
        selected = model.addVars(
            coordinates,
            vtype=GRB.BINARY,
            name="selected",
        )
        model.addConstr(selected[0, 0] == 0)

        add_fixed_root_connectivity(
            model,
            {
                coordinate: selected[coordinate]
                for coordinate in coordinates
            },
            coordinates,
            root_coordinate=(0, 0),
            name="fixed_root_connectivity",
        )
        model.optimize()

        self.assertEqual(model.Status, GRB.INFEASIBLE)

    def test_flow_adds_no_binary_variables(self):
        model = self.make_model("continuous_connectivity_flow")
        coordinates = [(0, 0), (0, 1)]
        selected = model.addVars(
            coordinates,
            vtype=GRB.BINARY,
            name="selected",
        )
        model.update()
        binary_variables_before = model.NumBinVars

        add_fixed_root_connectivity(
            model,
            {
                coordinate: selected[coordinate]
                for coordinate in coordinates
            },
            coordinates,
            root_coordinate=(0, 0),
            name="continuous_flow",
        )
        model.update()

        self.assertEqual(model.NumBinVars, binary_variables_before)

    def test_connected_start_constraints_are_temporary(self):
        model = self.make_model("temporary_connected_start")
        coordinates = [(0, 0), (0, 1), (0, 2)]
        coopt = CooptModel.__new__(CooptModel)
        coopt.model = model
        coopt.room_num = 1
        coopt.valid_coordinates = coordinates
        coopt.valid_coordinates_set = set(coordinates)
        coopt.common_room_indices = []
        coopt.open_room_indices = [0]
        coopt.furniture_num_list = [0]
        coopt.furniture_indices = []
        coopt.x = RoomGrid(model, 1, coordinates, name="x")
        coopt.furniture = FurnitureGrid(
            model,
            [],
            coordinates,
            name="furniture",
        )
        coopt.objective_function = 0
        model.addConstr(coopt.x[0, 0, 0] == 1)
        model.addConstr(coopt.x[0, 0, 2] == 1)
        model.Params.MIPFocus = 2
        model.setObjective(coopt.x[0, 0, 1], GRB.MAXIMIZE)
        coopt.get_solution = lambda: None
        coopt._validate_final_arrays = lambda: None
        model.update()
        variable_count_before = model.NumVars

        reference = np.ones((1, 1, 3), dtype=int)
        elapsed = coopt._prepare_connected_start(reference, 120)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(model.NumVars, variable_count_before)
        self.assertEqual(model.Params.MIPFocus, 2)
        self.assertEqual(model.ModelSense, GRB.MAXIMIZE)
        restored_objective = model.getObjective()
        self.assertEqual(restored_objective.size(), 1)
        self.assertEqual(
            restored_objective.getVar(0).VarName,
            coopt.x[0, 0, 1].VarName,
        )
        self.assertEqual(restored_objective.getCoeff(0), 1)

        model.Params.LazyConstraints = 1
        callback = coopt._build_connectivity_callback()
        model.setObjective(0)
        model.optimize(callback)

        self.assertEqual(model.Status, GRB.OPTIMAL)
        self.assertGreaterEqual(
            model._connectivity_lazy_stats["accepted_candidates"],
            1,
        )
        self.assertEqual(
            model._connectivity_lazy_stats["rejected_candidates"],
            0,
        )

        model.setObjective(coopt.x[0, 0, 1], GRB.MINIMIZE)
        model.optimize()

        self.assertEqual(model.Status, GRB.OPTIMAL)
        self.assertLess(coopt.x[0, 0, 1].X, 0.5)

        disconnected_reference = np.array([[[1, 0, 1]]], dtype=int)
        with self.assertRaisesRegex(
            ValueError,
            "reference room 0 has 2 components",
        ):
            coopt._prepare_connected_start(disconnected_reference, 1)

    def test_connected_start_repairs_uncompletable_reference(self):
        model = self.make_model("connected_start_fallback")
        coordinates = [(0, 0), (0, 1), (0, 2)]
        coopt = CooptModel.__new__(CooptModel)
        coopt.model = model
        coopt.room_num = 1
        coopt.valid_coordinates = coordinates
        coopt.valid_coordinates_set = set(coordinates)
        coopt.common_room_indices = []
        coopt.open_room_indices = [0]
        coopt.furniture_num_list = [0]
        coopt.furniture_indices = []
        coopt.x = RoomGrid(model, 1, coordinates, name="x")
        coopt.furniture = FurnitureGrid(
            model,
            [],
            coordinates,
            name="furniture",
        )
        coopt.objective_function = 0
        coopt.room_objective_function = 0
        coopt.weights = {"reference": 0.3}
        model.addConstr(coopt.x[0, 0, 0] == 0)
        model.addConstr(coopt.x[0, 0, 2] == 1)
        coopt.get_solution = lambda: None
        coopt._validate_final_arrays = lambda: None
        model.update()

        reference = np.array([[[1, 1, 0]]], dtype=int)
        coopt._prepare_connected_start(reference, 60)

        self.assertGreater(model.NumStart, 0)

        model.Params.LazyConstraints = 1
        callback = coopt._build_connectivity_callback()
        model.setObjective(0)
        model.optimize(callback)

        self.assertGreaterEqual(model.SolCount, 1)
        self.assertGreaterEqual(
            model._connectivity_lazy_stats["accepted_candidates"],
            1,
        )

    def test_final_solve_restores_time_limit_after_error(self):
        model = self.make_model("restore_final_time_limit")
        coopt = CooptModel.__new__(CooptModel)
        coopt.model = model
        coopt.model_name = "restore_final_time_limit"
        coopt.objective_function = 0
        coopt.output_dir = "."
        coopt._build_visualization_callback = lambda: (lambda *_: None)
        coopt._build_connectivity_callback = lambda _: (lambda *_: None)
        model.Params.TimeLimit = 17

        with (
            patch(
                "optimization.coopt_model.run_staged_optimization",
                side_effect=RuntimeError("solve failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "solve failed"),
        ):
            coopt.optimize(stage_num=1)

        self.assertEqual(model.Params.TimeLimit, 17)


if __name__ == "__main__":
    unittest.main()
