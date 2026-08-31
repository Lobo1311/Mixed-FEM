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

        return self.fEFT

    def XMap(self, qsiVec):
        N, _ = self.Shape(qsiVec)
        elnodevec = np.array([self.fNodes[i].fCoords for i in range(self.GetNNodes())])

        return N @ elnodevec

    def CalcStiffness(self):
        intRule = self.GetIntegrationRule()
        qsiPoints, weights = intRule.rule()

        n = self.GetNDOFs()
        KEl = np.zeros((n, n))
        fEl = np.zeros(n)

        for i in range(intRule.numPoints()):
            qsi = qsiPoints[i]
            weight = weights[i]

            N, dNdqsi = self.Shape(qsi)
            J, InvJ, DetJ = self.Jacobian(qsi, dNdqsi)
            dNdx = self.PhysicalDerivatives(dNdqsi, InvJ)

            B = self.CreateB(dNdx)

            kel += self.Dmat * B.T @ B * DetJ * weight

            if callable(self.tx):
                tx = self.tx(self.xmap(qsi)[0])
            else:
                tx = self.tx

            fel += N * tx * DetJ * weight

        return KEl, fEl

    @abstractmethod
    def GetNStateVars(self):
        raise NotImplementedError("GetNStateVars must be implemented in the derived class")

    @abstractmethod
    def Shape(self, qsiVec):
        raise NotImplementedError("Shape must be implemented in the derived class")

    @abstractmethod
    def Jacobian(self, qsiVec, dNdqsi):
        raise NotImplementedError("Jacobian must be implemented in the derived class")

    @abstractmethod
    def QsiNode(self, nodeIndex):
        raise NotImplementedError("QsiNode must be implemented in the derived class")

    @abstractmethod
    def NodeOrder(self):
        raise NotImplementedError("NodeOrder must be implemented in the derived class")

    @abstractmethod
    def GetIntegrationRule(self):
        raise NotImplementedError("GetIntegrationRule must be implemented in the derived class")

    @abstractmethod
    def IsBC(self):
        raise NotImplementedError("IsBC must be implemented in the derived class")