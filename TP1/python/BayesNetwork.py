import numpy as np

TRUE = 1
FALSE = 0

class RandomVariable:
    name = ""
    values = [FALSE, TRUE]
    proba = [0.5, 0.5]
    parents = []

    def __init__(self, name, proba):
        self.name = name
        self.proba = [1-proba, proba]

    def __init__(self, name, proba, parents):
        self.parents = parents
        # Vérification de la dimension de proba et du type des parents
        dimProba = 1
        for p in parents:
            dimProba *= p.values.size
        assert dimProba == proba.size



class NetworkDefinition:
    graphShape = np.array([], int)
    randomVariables = []

    def __init__(self, graphShape, probabilities, nodesName):
        self.graphShape = graphShape
        self.probabilities = probabilities
        self.nodesName = nodesName

    def addRndVar(self, var):
        self.randomVariables.append(var)
        galepgal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         smsmclass Network:

    def __init__(self, netDefinition):
        self.netDefinition = netDefinition
