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
        self.mesh = mesh
        self.K = None
        self.F = None
        self.u = None

    def Assemble(self):
        n = self.mesh.nequations()
        self.K = np.zeros((n, n))
        self.F = np.zeros(n)

        for el in self.mesh.elvec:
            kel, fel = el.calcstiff()
            for i in range(el.ndofs()):
                for j in range(el.ndofs()):
                    self.K[el.eft[i], el.eft[j]] += kel[i, j]

                self.F[el.eft[i]] += fel[i]

    def Solve(self):
        self.u = np.linalg.solve(self.K, self.F)

    def Run(self):
        self.Assemble()
        self.Solve()