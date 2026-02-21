from gurobipy import GRB


class RoomGrid:
    def __init__(self, model, room_num, valid_coordinates, name="x"):
        """
        Room division class, which serves to package Gurobi model variables

        Args:
            model: Gurobi model
            room_num: number of rooms
            valid_coordinates: valid coordinates
            name: variable name
        """
        self.valid_coordinates = valid_coordinates
        self.valid_coordinates_set = set(valid_coordinates)

        self.vars = model.addVars(
            room_num, valid_coordinates, vtype=GRB.BINARY, name=name
        )

    def __getitem__(self, key):
        """
        Get the variable value at the specified index
        If the index has None, return 0

        Args:
            key: tuple (k, i, j), representing the index of the room and the coordinates of the plot

        Returns:
            the corresponding variable, if the index has None, return 0
        """
        if isinstance(key, tuple) and len(key) == 3:
            k, i, j = key
            if k is None or i is None or j is None:
                return 0
            if (i, j) not in self.valid_coordinates_set:
                return 0
            return self.vars[k, i, j]
        else:
            raise IndexError("The index must be a tuple containing three elements (k, i, j)")


class PassageGrid:
    def __init__(self, model, valid_coordinates, name="p"):
        self.valid_coordinates = valid_coordinates
        self.valid_coordinates_set = set(valid_coordinates)
        self.vars = model.addVars(valid_coordinates, vtype=GRB.BINARY, name=name)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if i is None or j is None:
                return 0
            if (i, j) not in self.valid_coordinates_set:
                return 0
            return self.vars[i, j]
        else:
            raise IndexError("The index must be a tuple containing three elements (i, j)")
        

class FurnitureGrid:
    def __init__(self, model, furniture_indices, valid_coordinates, name="furniture"):
        """
        Furniture division class, which serves to package Gurobi model variables

        Args:
            model: Gurobi model
            furniture_indices: furniture indices
            valid_coordinates: valid coordinates
            name: variable name
        """
        self.valid_coordinates = valid_coordinates
        self.valid_coordinates_set = set(valid_coordinates)

        self.vars = model.addVars(
            furniture_indices, valid_coordinates, vtype=GRB.BINARY, name=name
        )

    def __getitem__(self, key):
        """
        Get the variable value at the specified index
        If the index has None, return 0

        Args:
            key: tuple (k, l, i, j), representing the index of the room, furniture, and the coordinates of the plot

        Returns:
            the corresponding variable, if the index has None, return 0
        """
        if isinstance(key, tuple) and len(key) == 4:
            k, l, i, j = key
            if k is None or l is None or i is None or j is None:
                return 0
            if (i, j) not in self.valid_coordinates_set:
                return 0
            return self.vars[k, l, i, j]
        else:
            raise IndexError("The index must be a tuple containing four elements (k, l, i, j)")
