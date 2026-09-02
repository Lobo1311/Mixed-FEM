from Element1D import *
from Solver import Integrate1D

class BaseDarcyEl(Element1D):
    def __init__(self, k, mu, q, nodes):
        if k < 0 or mu <= 0:
            raise ValueError("Inconsistent material properties")

        super().__init__(nodes)
        self.fK = k
        self.fMu = mu
        self.fQ = q
        self.fDMat = k / mu

    def PhysicalDerivatives(self, dNdqsi, invjac):
        dNdx = np.array(dNdqsi) * invjac

        return dNdx

    def CreateB(self, dNdx):
        B = np.array([dNdx])

        return B

    def Jacobian(self, qsi, dNdqsi):
        node1 = self.fNodes[0]
        node2 = self.fNodes[1]

        l = np.linalg.norm(np.array(node2.fCoords) - np.array(node1.fCoords))
        jac = l / 2.0
        invjac = 1.0 / jac
        detjac = jac

        return jac, invjac, detjac

    def GetNStateVars(self):
        return 1

    def GetIntegrationRule(self):
        return Integrate1D(self.fPOrder)

    def IsBC(self):
        return False

#* ----------------------------------------------------------
    
class DarcyEl(BaseDarcyEl):
    def __init__(self, k, mu, q, nodes, pOrder = 1):
        if len(nodes) > 2 or len(nodes) < 1:
            raise ValueError("Inconsistent nodes")

        super().__init__(k, mu, q, nodes)
        self.fPOrder = pOrder

    def QsiNode(self, nodeIndex):
        if nodeIndex < 0 or nodeIndex > 1:
            raise ValueError("Invalid node index")

        if nodeIndex == 0:
            return [-1.0]
        elif nodeIndex == 1:
            return [1.0]

    def Shape(self, qsi):
        N = np.array([(1.0 - qsi) / 2.0, (1.0 + qsi) / 2.0])
        dNdqsi = np.array([-0.5, 0.5])

        return N, dNdqsi

#* ----------------------------------------------------------

class DarcyElBC(BaseDarcyEl):
    def __init__(self, nodeVec: list[Node1D], bcType: str, bcVal: list[float]):

        if len(bcVal) != 1:
            raise ValueError("Val must have 1 position")
      
        super().__init__(0.0, 1.0, 0.0, nodeVec)
        self.bcType = bcType
        self.bcValue = bcVal

    def CalcStiffness(self):
        kel = np.zeros((1, 1))
        fel = np.zeros(1)

        if self.bcType.lower() == "dirichlet":
            kel += np.eye(1) * self._bignumber
            fel += np.array(self.bcValue) * self._bignumber
        elif self.bcType.lower() == "neumann":
            fel += np.array(self.bcValue)
        else:
            raise ValueError("Not defined or implemented boundary condition")

        return kel, fel

    def Jacobian(self, qsi, dNdqsi):
        raise ValueError("Should not be called")

    def QsiNode(self, nodeIndex):
        raise ValueError("Should not be called")

    def Shape(self, qsi):
        raise ValueError("Should not be called")

    def GetIntegrationRule(self):
        raise ValueError("Should not be called")

    def isBC(self):
        return True