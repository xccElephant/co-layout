import unittest

import numpy as np

from run_optimization import _upsample_room_reference


class UpsampleRoomReferenceTest(unittest.TestCase):
    def test_crops_odd_target_dimensions(self):
        coarse = np.array([[[1, 2], [3, 4]]])

        fine = _upsample_room_reference(
            coarse,
            scale_factor=2,
            target_width=3,
            target_length=3,
        )

        np.testing.assert_array_equal(
            fine,
            np.array(
                [
                    [
                        [1, 1, 2],
                        [1, 1, 2],
                        [3, 3, 4],
                    ],
                ]
            ),
        )

    def test_rejects_target_larger_than_upsampled_grid(self):
        with self.assertRaisesRegex(ValueError, "does not cover target"):
            _upsample_room_reference(
                np.ones((1, 2, 2)),
                scale_factor=2,
                target_width=5,
                target_length=4,
            )


if __name__ == "__main__":
    unittest.main()
