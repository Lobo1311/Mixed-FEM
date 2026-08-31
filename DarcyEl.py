from Element1D import *
from Solver import Integrate1D

class DarcyEl(Element1D):
    def __init__(self, k, mu, q, nodes, pOrder):
        if k <= 0:
            raise ValueError("Permeability must be positive")
        if mu <= 0:
            raise ValueError("Viscosity must be positive")
        if len(nodes) != 2:
            raise ValueError("1D Element must have 2 nodes")

        super().__init__(nodes)
        self.fK = k
        self.fMu = mu
        self.fQ = q
        self.fPOrder = pOrder
        self.fDMat = k / mu

    def PhysicalDerivatives(self, dNdqsi, invjac):
        dNdx = np.array(dNdqsi) * invjac

        return dNdx

    def CreateB(self, dNdx):
        B = np.array([dNdx])
        return B

    def Jacobian(self, qsiVec, dNdqsi):
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

    def Shape(self, qsiVec):
        qsi = qsiVec[0]
        N = np.array([(1 - qsi) / 2, (1 + qsi) / 2])
        dNdqsi = np.array([-0.5, 0.5])

        return N, dNdqsi