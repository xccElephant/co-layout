"""
This file is used to extract structured data from the JSON data.
"""

import math
import re
from numbers import Real


_NUMBER_WITH_UNIT = re.compile(
    r"""
    \s*
    (?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    \s*
    (?P<unit>[^\d\s].*?)?
    \s*
    """,
    re.VERBOSE,
)
_LINEAR_METER_UNITS = {"m", "meter", "meters", "metre", "metres"}
_SQUARE_METER_UNITS = {
    "m2",
    "m^2",
    "m²",
    "sqm",
    "squaremeter",
    "squaremeters",
    "squaremetre",
    "squaremetres",
}


def _normalize_unit(unit):
    return "".join(unit.casefold().split())


def _parse_number(
    value,
    *,
    path,
    units=(),
    minimum=None,
    maximum=None,
    integer=False,
):
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number, got boolean {value!r}")

    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        match = _NUMBER_WITH_UNIT.fullmatch(value)
        if match is None:
            raise ValueError(f"{path} must be a numeric value, got {value!r}")
        unit = match.group("unit")
        if unit and _normalize_unit(unit) not in {
            _normalize_unit(allowed_unit) for allowed_unit in units
        }:
            raise ValueError(
                f"{path} has unsupported unit {unit.strip()!r}"
            )
        number = float(match.group("number"))
    else:
        raise ValueError(
            f"{path} must be a number or numeric string, "
            f"got {type(value).__name__}"
        )

    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{path} must be at least {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{path} must be at most {maximum}, got {number}")
    if integer:
        if not number.is_integer():
            raise ValueError(f"{path} must be an integer, got {number}")
        return int(number)
    return number


def _parse_bool(value, *, path):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        f"{path} must be a JSON boolean or 'true'/'false', got {value!r}"
    )


def extract_data(json_data):
    """Extract structured data from JSON

    Args:
        json_data (dict): Parsed JSON data

    Returns:
        dict: A structured Dict containing all extracted parameters
    """
    building_parameters = json_data["basic_information"]["building_parameters"]
    envelope = building_parameters["building_envelope"]["value"]
    entrance = json_data["entrance_analysis"]["value"]
    if not isinstance(entrance, (list, tuple)) or len(entrance) != 2:
        raise ValueError(
            "entrance_analysis.value must contain exactly two coordinates"
        )

    building_params = {
        "width": _parse_number(
            envelope["width"],
            path="basic_information.building_parameters."
            "building_envelope.value.width",
            units=_LINEAR_METER_UNITS,
            minimum=1,
            integer=True,
        ),
        "length": _parse_number(
            envelope["length"],
            path="basic_information.building_parameters."
            "building_envelope.value.length",
            units=_LINEAR_METER_UNITS,
            minimum=1,
            integer=True,
        ),
        "floor_height": _parse_number(
            building_parameters["target_floor_height"]["value"],
            path="basic_information.building_parameters."
            "target_floor_height.value",
            units=_LINEAR_METER_UNITS,
            minimum=0,
        ),
        "entrance_location": [
            _parse_number(
                coordinate,
                path=f"entrance_analysis.value[{index}]",
                units=_LINEAR_METER_UNITS,
                minimum=0,
            )
            for index, coordinate in enumerate(entrance)
        ],
        "outdoor_space": json_data["outdoor_space_analysis"]["value"],
    }

    rooms = {}
    for room_name, room_data in json_data["room_analysis"].items():
        room_path = f"room_analysis.{room_name}"
        rooms[room_name] = {
            "area": _parse_number(
                room_data["area"]["value"],
                path=f"{room_path}.area.value",
                units=_SQUARE_METER_UNITS,
                minimum=0.000001,
            ),
            "is_open": _parse_bool(
                room_data["open_room"],
                path=f"{room_path}.open_room",
            ),
            "privacy_level": _parse_number(
                room_data["privacy"],
                path=f"{room_path}.privacy",
                minimum=1,
                maximum=3,
                integer=True,
            ),
        }

    furniture = {}
    for room_name, room_data in json_data["furniture_analysis"].items():
        furniture[room_name] = []
        for item_index, item in enumerate(room_data["furniture_list"]):
            item_path = (
                f"furniture_analysis.{room_name}."
                f"furniture_list[{item_index}]"
            )
            length = _parse_number(
                item["length"],
                path=f"{item_path}.length",
                units=_LINEAR_METER_UNITS,
                minimum=0.000001,
            )
            width = _parse_number(
                item["width"],
                path=f"{item_path}.width",
                units=_LINEAR_METER_UNITS,
                minimum=0.000001,
            )
            if "bed" in item["name"].casefold():
                length, width = width, length
            furniture[room_name].append(
                {
                    "name": item["name"],
                    "length": length,
                    "width": width,
                    "height": _parse_number(
                        item.get("height", 0),
                        path=f"{item_path}.height",
                        units=_LINEAR_METER_UNITS,
                        minimum=0,
                    ),
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
        distance_constraints = []
        for pair_index, item in enumerate(
            constraints_data.get("distance", [])
        ):
            path = (
                f"furniture_constraint_analysis.{room_name}."
                f"distance[{pair_index}]"
            )
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError(
                    f"{path} must contain two names and two offsets"
                )
            distance_constraints.append(
                (
                    item[0],
                    item[1],
                    _parse_number(
                        item[2],
                        path=f"{path}[2]",
                        units=_LINEAR_METER_UNITS,
                    ),
                    _parse_number(
                        item[3],
                        path=f"{path}[3]",
                        units=_LINEAR_METER_UNITS,
                    ),
                )
            )
        furniture_constraints[room_name] = {
            "boundary_items": constraints_data["boundary"],
            "distance_constraints": distance_constraints,
            "alignment_constraints": constraints_data.get("align", []),
            "facing_constraints": constraints_data.get("facing", []),
        }

    return building_params, rooms, furniture, room_constraints, furniture_constraints
