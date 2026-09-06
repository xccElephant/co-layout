import unittest

from utils.pre_process import extract_data


def make_memory():
    return {
        "basic_information": {
            "building_parameters": {
                "building_envelope": {
                    "value": {"width": 10, "length": 8},
                },
                "target_floor_height": {"value": 3},
            },
        },
        "entrance_analysis": {"value": [8, 4]},
        "outdoor_space_analysis": {"value": []},
        "room_analysis": {
            "Living Room": {
                "area": {"value": 20},
                "open_room": True,
                "privacy": 1,
            },
        },
        "furniture_analysis": {
            "Living Room": {
                "furniture_list": [
                    {
                        "name": "Sofa",
                        "length": 2.4,
                        "width": 0.9,
                        "height": 0.8,
                    },
                ],
            },
        },
        "room_constraint_analysis": {
            "adjacent_rooms": [],
            "corner_rooms": {},
        },
        "furniture_constraint_analysis": {
            "Living Room": {
                "boundary": ["Sofa"],
                "distance": [],
                "align": [],
                "facing": [],
            },
        },
    }


class ExtractDataParsingTest(unittest.TestCase):
    def test_accepts_explicit_metric_units_from_llm_output(self):
        memory = make_memory()
        memory["room_analysis"]["Living Room"]["area"]["value"] = "25 m²"
        memory["furniture_analysis"]["Living Room"]["furniture_list"][0][
            "length"
        ] = "2.4 m"

        _, rooms, furniture, _, _ = extract_data(memory)

        self.assertEqual(rooms["Living Room"]["area"], 25.0)
        self.assertEqual(furniture["Living Room"][0]["length"], 2.4)

    def test_coerces_only_unambiguous_strings(self):
        memory = make_memory()
        envelope = memory["basic_information"]["building_parameters"][
            "building_envelope"
        ]["value"]
        envelope["width"] = "10 m"
        envelope["length"] = "8"
        memory["basic_information"]["building_parameters"][
            "target_floor_height"
        ]["value"] = "3 metres"
        memory["entrance_analysis"]["value"] = ["8 m", "4"]
        memory["room_analysis"]["Living Room"]["open_room"] = "false"
        memory["room_analysis"]["Living Room"]["privacy"] = "3"
        distance = ["Sofa", "Sofa", "1.5 m", "-0.25"]
        memory["furniture_constraint_analysis"]["Living Room"][
            "distance"
        ] = [distance]

        building, rooms, _, _, constraints = extract_data(memory)

        self.assertEqual(building["width"], 10)
        self.assertEqual(building["floor_height"], 3.0)
        self.assertFalse(rooms["Living Room"]["is_open"])
        self.assertEqual(rooms["Living Room"]["privacy_level"], 3)
        self.assertEqual(
            constraints["Living Room"]["distance_constraints"][0][2:],
            (1.5, -0.25),
        )

    def test_rejects_approximate_area_with_field_path(self):
        memory = make_memory()
        memory["room_analysis"]["Living Room"]["area"]["value"] = (
            "about 25 m²"
        )

        with self.assertRaisesRegex(
            ValueError,
            r"room_analysis\.Living Room\.area\.value",
        ):
            extract_data(memory)

    def test_rejects_non_boolean_open_room(self):
        memory = make_memory()
        memory["room_analysis"]["Living Room"]["open_room"] = "yes"

        with self.assertRaisesRegex(
            ValueError,
            r"room_analysis\.Living Room\.open_room",
        ):
            extract_data(memory)

    def test_rejects_privacy_outside_supported_range(self):
        memory = make_memory()
        memory["room_analysis"]["Living Room"]["privacy"] = 4

        with self.assertRaisesRegex(
            ValueError,
            r"room_analysis\.Living Room\.privacy",
        ):
            extract_data(memory)

    def test_rejects_unsupported_furniture_unit(self):
        memory = make_memory()
        memory["furniture_analysis"]["Living Room"]["furniture_list"][0][
            "width"
        ] = "3 feet"

        with self.assertRaisesRegex(
            ValueError,
            r"furniture_analysis\.Living Room\.furniture_list\[0\]\.width",
        ):
            extract_data(memory)

    def test_rejects_malformed_metric_unit(self):
        memory = make_memory()
        memory["basic_information"]["building_parameters"][
            "building_envelope"
        ]["value"]["width"] = "10 m-"

        with self.assertRaisesRegex(ValueError, "unsupported unit"):
            extract_data(memory)


if __name__ == "__main__":
    unittest.main()
