from Mesh1D import *
from Node import *
from DarcyEl import DarcyEl, DarcyElBC
from Solver import *

import numpy as np

def DarcySolution(x, L):
    #! for k = 0.2 and q = 5x
    return 4.1666666667 * (0.24 * L - 0.24 * x + L**3 * x - L * x**3) / L

def main():
    mesh = Mesh1D()

    nEl = 100
    l = 1

    k = 0.2
    mu = 1
    def q(x):
        return 5 * x

    step = l / nEl
    xValues = []
    for i in range(nEl + 1):
        xValues.append(i*step)

    # Creating nodes
    nodes = []
    for i in xValues:
        nodes.append(Node1D(np.array([i, 0])))

    # Creating elements
    elements = []
    for i in range(nEl):
        elem = DarcyEl(k, mu, q, [nodes[i], nodes[i + 1]], 1)
        elements.append(elem)

    # Creating BCs
    bcLeft = DarcyElBC([nodes[0]], "dirichlet", [1])
    bcRight = DarcyElBC([nodes[-1]], "dirichlet", [0])

    for node in nodes:
        mesh.AddNode(node)

    for el in elements:
        mesh.AddElement(el)

    mesh.AddElement(bcLeft)
    mesh.AddElement(bcRight)

    solver = Solver(mesh)
    solver.Run()

    PlotResults(solver, k, mu)

def PlotResults(solver, k, mu):
    import matplotlib.pyplot as plt

    x = []
    pressure = []
    flow = []
    flow_x = []

    for node in solver.fMesh.fNodes:
        x.append(node.fCoords[0])
        pressure.append(solver.u[node.fIndex])

    for element in solver.fMesh.fElements:
        if not isinstance(element, DarcyEl):
            continue

        x0 = element.fNodes[0].fCoords[0]
        x1 = element.fNodes[1].fCoords[0]
        u0 = solver.u[element.fNodes[0].fIndex]
        u1 = solver.u[element.fNodes[1].fIndex]

        q = k / mu * (u0 - u1) / (x1 - x0)
        flow.append(q)
        flow_x.append((x0 + x1) / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, pressure)
    ax1.set_xlabel('x')
    ax1.set_ylabel('pressure')
    ax1.set_title('Pressure solution of the 1D Darcy Problem')
    ax1.plot(x, [DarcySolution(xi, 1) for xi in x], label='Analytical Solution', linestyle='--')
    ax1.grid()

    ax2.plot(flow_x, flow)
    ax2.set_xlabel('x')
    ax2.set_ylabel('flow')
    ax2.set_title('Flow solution of the 1D Darcy Problem')
    ax2.grid()

    plt.show()

if __name__ == "__main__":
    main()