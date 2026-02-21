import random
from constants import *

def find_boundary_coordinates(width, length, outdoor_space_coordinates):
    """
    Find the bounding plot coordinates of the floor plan (excluding outdoor areas)
    """
    outdoor_space = set((i, j) for i, j in outdoor_space_coordinates)
    boundary = []

    for i in range(width):
        for j in range(length):
            if (i, j) not in outdoor_space:
                # East
                if j == length - 1 or (i, j + 1) in outdoor_space:
                    boundary.append((i, j))
                # West
                if j == 0 or (i, j - 1) in outdoor_space:
                    boundary.append((i, j))
                # North
                if i == 0 or (i - 1, j) in outdoor_space:
                    boundary.append((i, j))
                # South
                if i == width - 1 or (i + 1, j) in outdoor_space:
                    boundary.append((i, j))

    boundary = sorted(list(set(boundary)))  # Deduplication + Sorting
    return boundary


def calculate_boundary_direction(
    width, length, outdoor_space_coordinates, queried_coordinate
):
    """
    Find the boundary directions of the queried_coordinate block
    """
    i, j = queried_coordinate
    outdoor_space = set((i, j) for i, j in outdoor_space_coordinates)
    directions = []
    # East
    if j == length - 1 or (i, j + 1) in outdoor_space:
        directions.append(EAST)
    # West
    if j == 0 or (i, j - 1) in outdoor_space:
        directions.append(WEST)
    # North
    if i == 0 or (i - 1, j) in outdoor_space:
        directions.append(NORTH)
    # South
    if i == width - 1 or (i + 1, j) in outdoor_space:
        directions.append(SOUTH)

    if len(directions) >= 1:
        # Randomly return a direction
        return random.choice(directions)
    else:
        return None


if __name__ == "__main__":
    length = 16
    width = 12
    values = ["0:5,12:16", "0:6,0:4"]
    outdoor_space_coordinates = []
    for section in values:
        width_range, length_range = section.split(",")
        width_range = width_range.split(":")
        width_range = [
            max(0, int(width_range[0])),
            min(width, int(width_range[1])),
        ]
        length_range = length_range.split(":")
        length_range = [
            max(0, int(length_range[0])), 
            min(length, int(length_range[1])),
        ]
        for i in range(width_range[0], width_range[1]):
            for j in range(length_range[0], length_range[1]):
                outdoor_space_coordinates.append((i, j))

    outdoor_space_coordinates = list(set(outdoor_space_coordinates))
    boundary = find_boundary_coordinates(width, length, outdoor_space_coordinates)
    for i in range(width):
        for j in range(length):
            if (i, j) in boundary:
                print("-", end=" ")
            elif (i, j) in outdoor_space_coordinates:
                print("X", end=" ")
            else:
                print("O", end=" ")
        print()
