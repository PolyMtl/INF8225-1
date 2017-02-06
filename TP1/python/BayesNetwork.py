import numpy as np
import collections as col
import math
from graphviz import Digraph
import matplotlib.pyplot as plt

import probatoolbox as pt

class VariableNode(object):
    def __init__(self, var, cpd, network, parents=[]):
        self.var = var
        self.net = network
        # Conversion de proba en liste si c'est un float
        if isinstance(cpd, float):
            cpd = [cpd]

        # Conversion de parents en liste si il est seul
        if isinstance(parents, VariableNode):
            parents = [parents]
        self.parents = parents

        # Verification de la dimension de proba et du type des parents
        dimProba = 1
        dimValues = len(var.values)
        dimList = []
        for p in parents:
            dimProba *= len(p.var.values)
            dimList.append(len(p.var.values))
        assert dimProba*(dimValues-1) == len(cpd)
        dimList.append(dimValues)

        # Calcul de la probabilite complementaire
        probaCompl = []
        for i in range(0,dimProba):
            pCumul = sum(cpd[i * (dimValues - 1):(i + 1) * (dimValues - 1)])
            probaCompl.append(1-pCumul)
        for p in probaCompl:
            cpd.append(p)

        # Stockage des probabilites conditionnelles
        self.cpd = np.array(cpd)
        self.cpd.shape = [_ for _ in reversed(dimList)]

        # Reordonnement des parents
        if len(parents) > 1:
            pMap = []
            reorderedProba = []
            reorderedDimList = []
            reorderedParents = []
            for var in network.subset.varList:
                for parentId, p in enumerate(parents):
                    if p.var is var:
                        pMap.append(parentId)
                        break
            if pMap != [i for i in range(0, len(pMap))]:
                pMap.append(len(pMap))
                base = [parent.var for parent in parents]+[self.var]
                for e in pt.Event(subset=network.subset).listVecEvents(base=base):
                    p = self.cpd
                    for eId in reversed(pMap):
                        p = p[e[eId]]
                    reorderedProba.append(p)

                for eId in pMap[:-1]:
                    reorderedDimList.append(dimList[eId])
                    reorderedParents.append(parents[eId])
                reorderedDimList.append(len(var.values))

                self.cpd = np.array(reorderedProba)
                self.cpd.shape = [_ for _ in reversed(reorderedDimList)]
                self.parents = reorderedParents

    def __str__(self):
        return self.var.name

    def printDef(self):
        list = []
        for valId, value in enumerate(self.var.values[:-1]):
            if not self.parents:
                list.append("p(" + self.var.printEqTo(value) + ") = " + str(self.cpd[valId]))
            else:
                parentBase = [e.var for e in self.parents]
                for parentEvent in (self.var == value).listVecEvents(base=parentBase):
                    defStr = "p(" + self.var.printEqTo(value) + "|"
                    defStr += pt.Event(subset=self.net.subset).eventToStr(parentEvent, parentBase)
                    defStr += ") = " + str(self.probaAt(parentEvent + [valId]))
                    list.append(defStr)

        return list

    def probaAt(self, event):
        if isinstance(event, pt.Event):
            list = []
            base = [e.var for e in (self.parents + [self])]
            events = event.listVecEvents(base=base)
            if len(events) == 1:
                return self.probaAt(events[0])
            for e in events:
                list.append([event.eventToStr(e, base), self.probaAt(e)])
            return list

        p = self.cpd
        for e in reversed(event):
            p = p[e]
        return p


class Network(object):

    def __init__(self, name="Bayes Network", subset=None):
        if subset is None:
            subset = pt.RandomVariableSubSet()
        self.nodes = []
        self.nodesMap = col.Counter()
        self.subset = subset
        self.isCompile = False
        self.defaultComputeMethod = "brut"
        self.name = name

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

    def createNode(self, name, proba, parents=[], values=[True, False]):
        if isinstance(parents, VariableNode) or isinstance(parents, pt.RandomVariable):
            parents = [parents]
        for i, var in enumerate(parents):
            if isinstance(var, pt.RandomVariable):
                parents[i] = self.node(var)

        var = None
        previousNode = None
        for nodeId, node in enumerate(self.nodes):
            if node.var.name == name:
                var = node.var
                previousNode = node
                break

        if var is None:
            var = pt.RandomVariable(name, values=values, subset=self.subset)

        node = VariableNode(var, proba, self, parents=parents)
        if previousNode is None:
            self.nodesMap[var] = len(self.nodes)
            self.nodes.append(node)
        else:
            self.nodes[self.nodesMap[var]] = node
            for n in self.nodes:
                for parentId, parent in enumerate(n.parents):
                    if parent is previousNode:
                        n.parents[parentId] = node
                        break
        return var

    def printDef(self):
        return [n.printDef() for n in self.nodes]

    def distribution(self, base=[], legendNeeded=False):
        if not base:
            base = self.subset.varList

        list = []
        legend = []
        for event in pt.Event(subset=self.subset).listEvents(base=base):
            list.append(self.computeInconditionnalProbability(event))
            if legendNeeded:
                legend.append(event.eventToStr(base=base))

        if legendNeeded:
            return [list, legend]
        else:
            return list

    def p(self, event, knowing=None, method=""):
        if not event.isValid:
            return 0
        if method == "":    # choisit la méthode de calcul (normal ou log)
            method = self.defaultComputeMethod
        if knowing is None:
            return self.computeInconditionnalProbability(event, method=method)
        else:
            if knowing == event + knowing:
                return 1    # Si event est un sous-evenement de knowing, proba=1
            p = self.p(knowing+event, method=method)
            pK = self.p(knowing, method=method)
            return p / pK

    def computeInconditionnalProbability(self, event, method=""):

        if method == "":
            method = self.defaultComputeMethod

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

            for e in events[1:]:
                lnPVar = 0
                # Calcul probabilite jointe
                for varNode in self.nodes:
                    lnPVar += math.log(varNode.probaAt(e))

                # Ajout a la somme de probabilite
                if expAlgo == 0:        # somme directe
                    proba += math.exp(lnPVar)
                elif expAlgo == 1:      # lnSum + ln( 1 + (pVar - sum) )
                    lnProba += math.log1p(math.exp(lnPVar - lnProba))
                elif expAlgo == 2:      # ln(pVar) + ln( 1 + (sum - pVar) )
                    lnProba = lnPVar + math.log1p(math.exp(lnProba - lnPVar))

            if expAlgo == 0:
                return proba
            return math.exp(lnProba)

        # Calcul de probabilite dans un espace non logarithmique
        proba = 0
        for e in event.listEvents(): # Pour chaque evenement atomique
            pVar = 1
            for varNode in self.nodes:  # Pour chaque variable du modele
                pVar *= varNode.probaAt(e)
            proba += pVar
        return proba

    def drawGraph(self, observed=[], show=False, engine='dot'):
        dot = Digraph(comment=self.name, engine=engine)
        dot.edge_attr.update(dir='back')

        if engine=='sfdp':
            dot.graph_attr.update(rotation='180')

        for node in self.nodes:
            style = 'filled' if node.var in observed else 'solid'
            dot.node(node.var.name, style=style)

        for child in self.nodes:
            for parent in child.parents:
                dot.edge(parent.var.name, child.var.name)

        dot.render('output/'+self.name.replace(' ', '')+'.gv', view=show)
        return dot

    def drawDistribution(self, show=False):
        dist = self.distribution(legendNeeded=True)
        fig, ax = plt.subplots(figsize=(15, 6))

        x = np.arange(len(dist[0]))
        width = 1 / 1.5
        ax.bar(x, dist[0], width, color="#1295DB", linewidth=0)
        plt.axis([0, len(dist[0]), 0, 1])
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(dist[1], rotation=90)

        ax.set_title("Distribution")
        plt.xlabel("Evenement")
        plt.ylabel("Probability")

        if show:
            plt.show()
