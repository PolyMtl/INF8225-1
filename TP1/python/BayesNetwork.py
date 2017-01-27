import numpy as np
import collections as col
import math

import probatoolbox as pt

defaultComputeMethod = "brut"

class VariableNode(object):
    def __init__(self, var, proba, network, parents=[] ):
        self.var = var
        self.net = network
        # Conversion de proba en liste si c'est un float
        if isinstance(proba, float):
            proba = [ proba ]

        # Conversion de parents en liste si il est seul
        if isinstance(parents, VariableNode):
            parents = [parents]
        self.parents = parents

        # Vérification de la dimension de proba et du type des parents
        dimProba = 1
        dimValues = len(var.values)
        dimList = []
        for p in parents:
            dimProba *= len(p.var.values)
            dimList.append(len(p.var.values))
        assert dimProba*(dimValues-1) == len(proba)
        dimList.append(dimValues)

        # Calcul de la probabilité complémentaire
        probaCompl = []
        for i in range(0,dimProba):
            pCumul = sum(proba[i*(dimValues-1):(i+1)*(dimValues-1)])
            probaCompl.append(1-pCumul)
        for p in probaCompl:
            proba.append(p)

        # Stockage des probabilités conditionnelles
        self.proba = np.array(proba)
        self.proba.shape = dimList

    def __str__(self):
        return self.var.name

    def printDef(self):
        list = []
        for valId, value in enumerate(self.var.values[:-1]):
            if not self.parents:
                list.append("p(" + self.var.printEqTo(value) + ") = " + str(self.proba[valId]))
            else:
                parentBase = [e.var for e in self.parents]
                for parentEvent in (self.var == value).listVecEvents(base=parentBase):
                    defStr = "p(" + self.var.printEqTo(value) + "|"
                    defStr += pt.Event().eventToStr(parentEvent, parentBase)
                    defStr += ") = " + str(self.probaAt(parentEvent + [valId]))
                    list.append(defStr)

        return list

    def probaAt(self, event):
        if isinstance(event, pt.Event):
            list = []
            base = self.parents + [self]
            base = [e.var for e in base]
            events = event.listVecEvents(base=base)
            if len(events) == 1:
                return self.probaAt(events[0])
            for e in events:
                list.append([event.eventToStr(e, base), self.probaAt(e)])
            return list

        p = self.proba
        for e in reversed(event):
            p = p[e]
        return p


class Network(object):

    def __init__(self, subset=pt.defaultSubset):
        self.nodes = []
        self.nodesMap = col.Counter()
        self.subset = subset

    def __str__(self):
        defStr = "BayesNetwork["
        if self.nodes:
            defStr += str(self.nodes[0])
        for n in self.nodes[1:]:
            defStr += ","+str(n)
        defStr += "]"
        return defStr

    def node(self, var):
        return self.nodes[self.nodesMap[var]]

    def addNode(self, name, proba, parents=[], values=[True, False]):
        if isinstance(parents, VariableNode) or isinstance(parents, pt.RandomVariable):
            parents = [parents]

        for i, var in enumerate(parents):
            if isinstance(var, pt.RandomVariable):
                parents[i] = self.node(var)

        var = pt.RandomVariable(name, values=values, subset=self.subset)
        node = VariableNode(var, proba, self, parents=parents)
        self.nodesMap[var] = len(self.nodes)
        self.nodes.append(node)

        return var

    def distribution(self, base=[], legendNeeded=False):
        if not base:
            base = self.subset.varList

        list = []
        legend = []
        for event in pt.Event(subset=self).listEvents(base=base):
            list.append(event.p())
            if legendNeeded:
                legend.append(event.eventToStr(base=base))

        if legendNeeded:
            return [list, legend]
        else:
            return list

    def p(self, event, knowing=False, method=""):
        if not event.isValid:
            return 0
        if method == "":
            method = defaultComputeMethod
        if not knowing:
            return self.computeInconditionnalProbability(event, method=method)
        else:
            if knowing == event + knowing:
                return 1
            p1 = self.p(knowing-event, method=method)
            p2 = self.p(knowing+event, method=method)
            return p2 / (p1 + p2)

    def computeInconditionnalProbability(self, event, method=""):

        if method == "":
            method = defaultComputeMethod

        proba = 0

        if method.startswith("exp"):
            expAlgo = 0
            if method == 'exp1':
                expAlgo = 1
            elif method == 'exp2':
                expAlgo = 2

            events = event.listEvents()
            lnProba = 0
            for varNode in self.nodes:
                lnProba += math.log(varNode.probaAt(events[0]))
            proba = math.exp(lnProba)

            for e in event.listEvents()[1:]:
                lnPVar = 0
                # Calcul probabilité jointe
                for varNode in self.nodes:
                    lnPVar += math.log(varNode.probaAt(e))

                # Ajout à la somme de probabilité
                if expAlgo == 0:        # somme directe
                    proba += math.exp(lnPVar)
                elif expAlgo == 1:      # lnSum + ln( 1 + (pVar - sum) )
                    lnProba += math.log1p(math.exp(lnPVar - lnProba))
                elif expAlgo == 2:      # ln(pVar) + ln( 1 + (sum - pVar) )
                    lnProba = lnPVar + math.log1p(math.exp(lnProba - lnPVar))

            if expAlgo == 0:
                return proba
            return math.exp(lnProba)

        for e in event.listEvents():
            pVar = 1
            for varNode in self.nodes:
                pVar *= varNode.probaAt(e)
            proba += pVar
        return proba