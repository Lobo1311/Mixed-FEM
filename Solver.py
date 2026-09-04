import numpy as np
import math

class Integrate1D:
    def __init__(self, pOrder: int):
        self.fPOrder = pOrder

        self.fNumPoints = math.ceil((self.fPOrder + 1) / 2)
        self.fX, self.fW = np.polynomial.legendre.leggauss(self.fNumPoints)

    def rule(self):
        return self.fX, self.fW

    def numPoints(self):
        return self.fNumPoints


class Solver:
    def __init__(self, mesh):
        self.fMesh = mesh
        self.K = None
        self.F = None
        self.u = None

    def Assemble(self):
        n = self.fMesh.GetNEquations()
        self.K = np.zeros((n, n))
        self.F = np.zeros(n)

        for el in self.fMesh.fElements:
            EFT = el.fEFT
            kel, fel = el.CalcStiffness()

            for i in range(el.GetNDOFs()):
                for j in range(el.GetNDOFs()):
                    self.K[EFT[i], EFT[j]] += kel[i, j]

                self.F[EFT[i]] += fel[i]

        # debug
        print("Global Stiffness Matrix K:")
        print(self.K)
        print("Global Force Vector F:")
        print(self.F)

    def Solve(self):
        self.u = np.linalg.solve(self.K, self.F)

    def Run(self):
        self.Assemble()
        self.Solve()