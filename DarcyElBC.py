from Element1D import *

class DarcyElBC(Element1D):
    def __init__(self, nodeVec: list[Node1D], bcType: str, bcVal: list[float]):

        if len(bcVal) != 1:
            raise ValueError("Val must have 1 position")
      
        super().__init__(nodeVec=nodeVec)
        self.bcType = bcType
        self.bcValue = bcVal
        self.fDim = 0

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

    def isBC(self):
        return True