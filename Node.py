

class Node1D:
    def __init__(self, coords: list[float]):
        self.fCoords = coords
        self.fIndex = -1

    def __str__(self):
        return f"Node index={self.fIndex}, coords={self.fCoords})"