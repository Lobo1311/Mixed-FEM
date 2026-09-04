from Node import Node1D

class Mesh1D:
    def __init__(self):
        self.fNodes = []
        self.fElements = []

    def GetNNodes(self):
        return len(self.fNodes)

    def GetNElements(self):
        return len(self.fElements)

    def AddNode(self, node: Node1D):
        self.fNodes.append(node)
        node.fIndex = len(self.fNodes) - 1

    def AddElement(self, element):
        self.fElements.append(element)
        element.fIndex = len(self.fElements) - 1
        element.BuildEFT()

    def GetNEquations(self):
        Total = 0
        CountedNodes = []

        for el in self.fElements:
            for node in el.fNodes:
                if node.fIndex not in CountedNodes:
                    CountedNodes.append(node.fIndex)
                    Total += el.GetNStateVars()

        return Total

class MixedMesh1D(Mesh1D):
    def AddElement(self, element):
        self.fElements.append(element)
        element.fIndex = len(self.fElements) - 1

    def BuildEFT(self):
        total_not_bc_elements = len([el for el in self.fElements if not el.IsBC()])
        for el in self.fElements:
            el.BuildEFT(total_not_bc_elements)

    def GetNEquations(self):
        Total = 0
        CountedNodes = []

        for el in self.fElements:
            for node in el.fNodes:
                if node.fIndex not in CountedNodes:
                    CountedNodes.append(node.fIndex)
                    Total += el.GetNNodeVars()

            Total += el.GetNElemVars()

        return Total