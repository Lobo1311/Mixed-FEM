from MixedElement1D import MixedElement1D
from Node import Node1D
from Solver import Integrate1D

import numpy as np

class BaseMixedDarcyEl(MixedElement1D):
    def __init__(self, k, mu, q, nodes):
        if k < 0 or mu <= 0:
            raise ValueError("Inconsistent material properties")

        super().__init__(nodes)
        self.fK = k
        self.fMu = mu
        self.fQ = q
        self.fDMat = mu / k
        self.fPOrder = 1

    def GetNNodeVars(self):
        return 1

    def GetNElemVars(self):
        return 1

    def GetIntegrationRule(self):
        return Integrate1D(self.fPOrder)

    def IsBC(self):
        return False

    def CalcDetJac(self):
        node1 = self.fNodes[0]
        node2 = self.fNodes[1]
        l = np.linalg.norm(np.array(node2.fCoords) - np.array(node1.fCoords))

        return l / 2.0

#* ----------------------------------------------------------
    
class MixedDarcyEl(BaseMixedDarcyEl):
    def __init__(self, k, mu, q, nodes):
        if len(nodes) > 2 or len(nodes) < 1:
            raise ValueError("Inconsistent nodes")

        super().__init__(k, mu, q, nodes)

    def QsiNode(self, nodeIndex):
        if nodeIndex < 0 or nodeIndex > 1:
            raise ValueError("Invalid node index")

        if nodeIndex == 0:
            return [-1.0]
        elif nodeIndex == 1:
            return [1.0]

    def Shape(self, qsi):
        N = np.array([(1.0 - qsi) / 2.0, (1.0 + qsi) / 2.0, 0.5])
        dNdqsi = np.array([-0.5, 0.5, 0.0])

        return N, dNdqsi

#* ----------------------------------------------------------

class MixedDarcyElBC(BaseMixedDarcyEl):
    def __init__(self, nodeVec: list[Node1D], bcType: str, bcVal: list[float]):

        if len(bcVal) != 1:
            raise ValueError("Val must have 1 position")
      
        super().__init__(1.0, 1.0, 0.0, nodeVec)
        self.bcType = bcType
        self.bcValue = bcVal

    def CalcStiffness(self):
        kel = np.zeros((1, 1))
        fel = np.zeros(1)

        if self.bcType.lower() == "dirichlet":
            fel += np.array(self.bcValue)
        elif self.bcType.lower() == "neumann":
            kel += np.eye(1) * self._bignumber
            fel += np.array(self.bcValue) * self._bignumber
        else:
            raise ValueError("Not defined or implemented boundary condition")

        return kel, fel

    def GetNDOFs(self):
        return self.GetNodesNDOFs()

    def GetNElemVars(self):
        return 0

    def BuildEFT(self, _):
        self.fEFT = np.empty(self.GetNDOFs(), dtype=int)

        for i, node in enumerate(self.fNodes):
            for j in range(self.GetNNodeVars()):
                self.fEFT[i * self.GetNNodeVars() + j] = node.fIndex * self.GetNNodeVars() + j

        return self.fEFT

    def QsiNode(self, nodeIndex):
        raise ValueError("Should not be called")

    def Shape(self, qsi):
        raise ValueError("Should not be called")

    def GetIntegrationRule(self):
        raise ValueError("Should not be called")

    def IsBC(self):
        return True