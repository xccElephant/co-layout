"""
This file is used to extract structured data from the JSON data.
"""


def extract_data(json_data):
    """Extract structured data from JSON

    Args:
        json_data (dict): Parsed JSON data

    Returns:
        dict: A structured Dict containing all extracted parameters
    """
    building_params = {
        "width": json_data["basic_information"]["building_parameters"][
            "building_envelope"
        ]["value"]["width"],
        "length": json_data["basic_information"]["building_parameters"][
            "building_envelope"
        ]["value"]["length"],
        "floor_height": json_data["basic_information"]["building_parameters"][
            "target_floor_height"
        ]["value"],
        "entrance_location": json_data["entrance_analysis"]["value"],
        "outdoor_space": json_data["outdoor_space_analysis"]["value"],
    }

    rooms = {}
    for room_name, room_data in json_data["room_analysis"].items():
        rooms[room_name] = {
            "area": float(room_data["area"]["value"]),
            "is_open": room_data["open_room"],
            "privacy_level": int(room_data["privacy"]),
        }

    furniture = {}
    for room_name, room_data in json_data["furniture_analysis"].items():
        furniture[room_name] = []
        for item in room_data["furniture_list"]:
            if "bed" in item["name"].casefold():
                item["length"], item["width"] = item["width"], item["length"]
            furniture[room_name].append(
                {
                    "name": item["name"],
                    "length": item["length"],
                    "width": item["width"],
                    "height": item.get("height", 0),
                }
            )

    room_constraints = {
        "room_adjacency": json_data["room_constraint_analysis"]["adjacent_rooms"],
        "corner_rooms": json_data["room_constraint_analysis"]["corner_rooms"],
    }

    furniture_constraints = {}
    for room_name, constraints_data in json_data[
        "furniture_constraint_analysis"
    ].items():
        furniture_constraints[room_name] = {
            "boundary_items": constraints_data["boundary"],
            "distance_constraints": [
                (item[0], item[1], item[2], item[3])
                for item in constraints_data.get("distance", [])
            ],
            "alignment_constraints": constraints_data.get("align", []),
            "facing_constraints": constraints_data.get("facing", []),
        }

    return building_params, rooms, furniture, room_constraints, furniture_constraints
