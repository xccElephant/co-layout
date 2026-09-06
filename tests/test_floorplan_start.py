import unittest

import numpy as np
from gurobipy import Env, Model

from optimization.floorplan_model import FloorplanModel
from optimization.grid_model import PassageGrid, RoomGrid


class FloorplanReferenceStartTest(unittest.TestCase):
    def setUp(self):
        self.environment = Env(empty=True)
        self.environment.setParam("OutputFlag", 0)
        self.environment.start()
        self.model = Model("floorplan_reference_start", env=self.environment)
        self.model.Params.OutputFlag = 0

        self.floorplan = FloorplanModel.__new__(FloorplanModel)
        self.floorplan.model = self.model
        self.floorplan.room_num = 1
        self.floorplan.width = 2
        self.floorplan.length = 2
        self.floorplan.valid_coordinates = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        self.floorplan.x = RoomGrid(
            self.model,
            self.floorplan.room_num,
            self.floorplan.valid_coordinates,
            name="x",
        )
        self.floorplan.passage = PassageGrid(
            self.model,
            self.floorplan.valid_coordinates,
            name="passage",
        )
        self.model.update()

    def tearDown(self):
        self.model.dispose()
        self.environment.dispose()

    def test_sets_room_and_complementary_passage_start(self):
        reference = np.array([[[1, 1], [0, 0]]], dtype=int)

        count = self.floorplan._set_reference_start(reference)

        self.assertEqual(count, 2)
        self.assertEqual(self.floorplan.x[0, 0, 0].Start, 1)
        self.assertEqual(self.floorplan.x[0, 1, 0].Start, 0)
        self.assertEqual(self.floorplan.passage[0, 0].Start, 0)
        self.assertEqual(self.floorplan.passage[1, 0].Start, 1)

    def test_rejects_reference_with_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "reference_array"):
            self.floorplan._set_reference_start(np.zeros((1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
