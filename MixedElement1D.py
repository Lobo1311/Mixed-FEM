from Node import *

from abc import ABC, abstractmethod
import numpy as np

class MixedElement1D(ABC):
    def __init__(self, nodes: list[Node1D]):
        self.fNodes = nodes
        self.fIndex = -1
        self.fEFT = None
        self._bignumber = 1.e12

    def __str__(self):
        return f"Element index={self.fIndex}, nodes=[{self.fNodes[0]}, {self.fNodes[1]}])"

    def GetNNodes(self):
        return len(self.fNodes)

    def GetNDOFs(self):
        Total = self.GetNodesNDOFs()
        Total += self.GetNElemVars()

        return Total

    def GetNodesNDOFs(self):
        Total = 0
        for node in self.fNodes:
            if node.fIndex == -1:
                raise ValueError("Node index not set for one of the nodes in the element")

            Total += self.GetNNodeVars()

        return Total

    def BuildEFT(self, total_flow_dofs):
        self.fEFT = np.empty(self.GetNDOFs(), dtype=int)

        for i, node in enumerate(self.fNodes):
            for j in range(self.GetNNodeVars()):
                self.fEFT[i * self.GetNNodeVars() + j] = node.fIndex * self.GetNNodeVars() + j

        for k in range(self.GetNElemVars()):
            self.fEFT[len(self.fNodes) * self.GetNNodeVars() + k] = total_flow_dofs + 1 + self.fIndex * self.GetNElemVars() + k

        return self.fEFT

    def XMap(self, qsi):
        N, _ = self.Shape(qsi)
        elnodevec = np.array([self.fNodes[i].fCoords for i in range(self.GetNNodes())])

        return N[:self.GetNodesNDOFs()] @ elnodevec

    def CalcStiffness(self):
        intRule = self.GetIntegrationRule()
        qsiPoints, weights = intRule.rule()

        n = self.GetNDOFs()
        KEl = np.zeros((n, n))
        fEl = np.zeros(n)

        n_flow = self.GetNodesNDOFs()
        n_pressure = self.GetNElemVars()

        for i in range(intRule.numPoints()):
            qsi = qsiPoints[i]
            weight = weights[i]

            N, dNdqsi = self.Shape(qsi)

            flow_matrix = np.zeros((n_flow, n_flow))
            for i_flow in range(n_flow):
                for j_flow in range(n_flow):
                    flow_matrix[i_flow, j_flow] = self.fDMat * N[i_flow] * N[j_flow] * self.CalcDetJac() * weight

            pressure_matrix = np.zeros((n_flow, n_pressure))
            for i_flow in range(n_flow):
                for j_pressure in range(n_pressure):
                    pressure_matrix[i_flow, j_pressure] = -N[n_flow + j_pressure] * dNdqsi[i_flow] * weight
                
            KEl[:n_flow, :n_flow] += flow_matrix
            KEl[:n_flow, n_flow:] += pressure_matrix
            KEl[n_flow:, :n_flow] += pressure_matrix.T

            if callable(self.fQ):
                q = self.fQ(self.XMap(qsi)[0])
            else:
                q = self.fQ

            for i_pressure in range(n_pressure):
                fEl[n_flow + i_pressure] += -N[n_flow + i_pressure] * q * self.CalcDetJac() * weight

        return KEl, fEl

    @abstractmethod
    def GetNNodeVars(self):
        raise NotImplementedError("GetNNodeVars must be implemented in the derived class")

    @abstractmethod
    def GetNElemVars(self):
        raise NotImplementedError("GetNElemVars must be implemented in the derived class")

    @abstractmethod
    def Shape(self, qsi):
        raise NotImplementedError("Shape must be implemented in the derived class")

    @abstractmethod
    def QsiNode(self, nodeIndex):
        raise NotImplementedError("QsiNode must be implemented in the derived class")

    @abstractmethod
    def GetIntegrationRule(self):
        raise NotImplementedError("GetIntegrationRule must be implemented in the derived class")

    @abstractmethod
    def IsBC(self):
        raise NotImplementedError("IsBC must be implemented in the derived class")