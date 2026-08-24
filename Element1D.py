from Node import *

from abc import ABC, abstractmethod
import numpy as np

class Element1D(ABC):
    def __init__(self, nodes: list[Node1D]):
        if len(nodes) != 2:
            raise ValueError("1D Element must have 2 nodes")

        self.fNodes = nodes
        self.fIndex = -1
        self.fEFT = None
        self._bignumber = 1.e12

    def __str__(self):
        return f"Element index={self.fIndex}, nodes=[{self.fNodes[0]}, {self.fNodes[1]}])"

    def GetNNodes(self):
        return len(self.fNodes)

    def GetNDOFs(self):
        Total = 0
        for node in self.fNodes:
            if node.fIndex == -1:
                raise ValueError("Node index not set for one of the nodes in the element")

            Total += self.GetNStateVars()

        return Total

    def BuildEFT(self):
        self.fEFT = np.empty(self.GetNDOFs(), dtype=int)

        for i, node in enumerate(self.fNodes):
            for j in range(self.GetNStateVars()):
                self.fEFT[i * self.GetNStateVars() + j] = node.fIndex * self.GetNStateVars() + j

    @abstractmethod
    def GetNStateVars(self):
        raise NotImplementedError("GetNStateVars must be implemented in the derived class")

    @abstractmethod
    def Shape(self, qsiVec: list[float]):
        raise NotImplementedError("Shape must be implemented in the derived class")

    @abstractmethod
    def Jacobian(self, qsiVec: list[float]):
        raise NotImplementedError("Jacobian must be implemented in the derived class")