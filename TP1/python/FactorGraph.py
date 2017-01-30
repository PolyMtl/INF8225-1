import numpy as np
import collections as col
import math
from graphviz import Digraph
from matplotlib import scale

import probatoolbox as pt
import BayesNetwork as bnet


defaultCompileVerbose = False


class Message(object):
    def __init__(self, fromNode, fromId, toNode, spData=None, msData=None):
        self.spData = spData
        self.msData = msData
        self.fromNode = fromNode
        self.fromId = fromId
        self.toNode = toNode

    def send(self):
        self.toNode.receiveMsg(self)

    def __str__(self):
        return str(self.fromNode) + " -> " + str(self.toNode)


class Node(object):

    def __init__(self):
        self.neighbours = []
        self.idInNeighbours = []

        self.msgsReceived = []
        self.msgsReceivedCount = 0
        self.msgSent = None
        self.msgReturned = None
        self.observedValueId = -1
        self.jobsDone = False
        self.msData = []
        self.spData = []
        self.reset()

    def reset(self, recursive=False):
        if recursive:
            self.recursiveReset(self)
            return

        for i in range(len(self.msgsReceived)):
            self.msgsReceived[i] = None
        self.msgsReceivedCount = 0
        self.msgSent = None
        self.msgReturned = None
        self.jobsDone = False
        self.msData = []
        self.spData = []

    def recursiveReset(self, fromNode=None):
        if defaultCompileVerbose:
            print("reset " + str(self))
        self.reset()

        if self.isObserved():
            return

        for n in self.neighbours:
            if n is not fromNode:
                n.recursiveReset(self)

    def isObserved(self):
        return self.observedValueId != -1

    def addNeighbours(self, node):
        if node in self.neighbours:
            if defaultCompileVerbose:
                print("Error neighbours already added")
            return

        # Add neighbours in own list
        idInSelf = len(self.neighbours)
        self.neighbours.append(node)
        self.msgsReceived.append(None)

        # Add self in neighbour list
        idInNeighbour = len(node.neighbours)
        node.neighbours.append(self)
        node.msgsReceived.append(None)

        # Aknowledge ids
        self.idInNeighbours.append(idInNeighbour)
        node.idInNeighbours.append(idInSelf)

    def receiveMsg(self, msg):
        if self.msgsReceived[msg.fromId] is not None:
            if defaultCompileVerbose:
                print(str(msg.toNode)+" already received a message from "+str(msg.fromNode)+", message ignored...")
            return

        self.msgsReceived[msg.fromId] = msg
        self.msgsReceivedCount += 1

        if len(self.neighbours) - self.msgsReceivedCount == 0:
            self.msgReturned = msg

        return

    def processMsgs(self, knowing, method='base'):

        ngbCount = len(self.neighbours)

        if self.jobsDone:
            return []

        if self.msgReturned is not None:
            # Un message de retour a ete recu et doit etre distribuer a tous les autres noeuds
            list = []
            for nodeId, outNode in enumerate(self.neighbours):
                updateMsgSent = False
                if nodeId == self.msgReturned.fromId:
                    if self.msgSent is None:
                        updateMsgSent = True
                    else:
                        continue    # Sauter l'emmeteur du message

                idForOutNode = self.idInNeighbours[nodeId]

                msg = Message(self, idForOutNode, outNode)
                self.computeMsgData(msg, knowing=knowing, methodName=method)

                list.append(msg)
                if updateMsgSent:
                    self.msgSent = msg
            self.jobsDone = True
            return list

        if self.msgSent is not None or ngbCount - self.msgsReceivedCount > 1:
            # On doit toujours attendre des messages des noeuds voisins...
            return []

        msgs = []
        # Recherche du noeud qui n'a pas encore transmis de message
        outNode = None
        idForOutNode = -1
        for i, m in enumerate(self.msgsReceived):
            if not m:
                outNode = self.neighbours[i]
                idForOutNode = self.idInNeighbours[i]

        msg = Message( self, idForOutNode, outNode)
        self.computeMsgData(msg, knowing=knowing, methodName=method)

        self.msgSent = msg

        return [self.msgSent]

    def processLeafMsgs(self, knowing, method='base'):

        if self.msgSent or (not self.isObserved() and len(self.neighbours) != 1):
            return []

        self.jobsDone = True

        msgList = []
        for i, neigbour in enumerate(self.neighbours):
            outNode = neigbour
            idForOutNode = self.idInNeighbours[i]

            msg = Message(self, idForOutNode, outNode)
            self.computeLeafMsgData(msg, knowing=knowing, methodName=method)

            self.msgSent = msg
            msgList.append(msg)

        return msgList

    def computeMsgData(self, msg, knowing, methodName='base'):
        pass

    def computeLeafMsgData(self, msg, knowing, methodName='base'):
        pass

    def finishJob(self, methodName='base'):

        if self.isObserved():
            for msg in self.msgsReceived:
                if msg is not None:
                    self.msData = msg.msData
                    self.spData = msg.spData
                    if methodName.startswith('exp'):
                        self.msData = [math.exp(e) for e in self.msData]
                        self.spData = [math.exp(e) for e in self.spData]
        else:
            r = range(len(self.msgReturned.msData))
            msInput = [[self.msgReturned.msData[i], self.msgSent.msData[i]] for i in r]
            spInput = [[self.msgReturned.spData[i], self.msgSent.spData[i]] for i in r]

            if methodName.startswith('exp'):
                msData = [ms[0] + ms[1] for ms in msInput]
                spData = [sp[0] + sp[1] for sp in spInput]
                self.msData = [math.exp(e) for e in msData]
                self.spData = [math.exp(e) for e in spData]
            else:
                self.msData = [ms[0] * ms[1] for ms in msInput]
                self.spData = [sp[0] * sp[1] for sp in spInput]

        # Normalisation
        s = sum(self.spData)
        self.spData = [sp / s for sp in self.spData]

        s = sum(self.msData)
        self.msData = [ms / s for ms in self.msData]


class FunctionNode(Node):

    def __init__(self, proba, xNodes, graph):
        Node.__init__(self)

        self.graph = graph

        # Conversion de proba en liste si c'est un float
        if isinstance(proba, float):
            proba = [proba]

        # Conversion de la variable associee en liste si elle est seule et enregistrement des variables
        if isinstance(xNodes, FunctionNode):
            xNodes = [xNodes]

        self.neighboursBase = []
        for x in xNodes:
            x.addNeighbours(self)
            self.neighboursBase.append(x.var)

        # Verification de la dimension de proba
        dimX = []
        globalDim = 1
        for x in xNodes:
            dimX.append(len(x.var.values))
            globalDim *= len(x.var.values)
        assert globalDim == proba.size

        # Stockage des probabilites conditionnelles
        self.proba = np.array(proba)
        self.proba.shape = [_ for _ in reversed(dimX)]

    def __str__(self):
        defStr = "f(" + str(self.neighbours[0])
        for x in self.neighbours[1:]:
            defStr += "," + str(x)
        defStr += ")"
        return defStr

    def base(self):
        b = [e.var for e in self.neighbours]
        return b

    def printDef(self):
        list = []
        for event in pt.Event(subset=self.base()[0].subset).listVecEvents(base=self.base()):
            defStr = "p("
            defStr += pt.Event(subset=self.base()[0].subset).eventToStr(event, self.base())
            defStr += ") = " + str(self.f(event))
            list.append(defStr)
        return list

    def f(self, event):
        if isinstance(event, pt.Event):
            returnList = []
            base = [e.var for e in self.neighbours]
            events = event.listVecEvents(base=base)
            if len(events) == 1:
                return self.f(events[0])
            for e in events:
                returnList.append([event.eventToStr(e, base), self.f(e)])
            return returnList

        if isinstance(event[0], list):
            returnList = []
            for e in event:
                returnList.append(self.f(e))
            return returnList

        p = self.proba
        for e in reversed(event):
            p = p[e]
        return p

    def computeMsgData(self, msg, knowing, methodName='base'):
        spData = []
        msData = []

        method = -1
        valueIni = 0
        if methodName.startswith('exp'):
            method = 0 if len(methodName)<4 else int(methodName[3])
            if method > 0:
                valueIni = -float('inf')

        for valueId in msg.toNode.valuesId():
            # Pour chaque valeur de la variable du noeud destinataire, on calcul la somme des produits des messages
            sp = valueIni
            ms = -float('inf')

            event = (msg.toNode.var == valueId) & knowing
            if not event.isValid:
                spData.append(sp)
                msData.append(ms)
                continue

            for e in event.listVecEvents(self.neighboursBase):

                if knowing == event:
                    complement = pt.Event.eventFromSpace(e, self.neighboursBase) + (msg.toNode.var != valueId)
                    f = sum(self.f(complement.listVecEvents(self.neighboursBase)))
                else:
                    f = self.f(e)

                if method >= 0:
                    f = math.log(f)
                spTemp = f
                msTemp = f

                for i, m in enumerate(self.msgsReceived):
                    if m is None or (m.fromNode is msg.toNode):
                        continue
                    spMsg = m.spData[e[i]]
                    msMsg = m.msData[e[i]]
                    if method < 0:
                        spTemp *= spMsg
                        msTemp *= msMsg
                    else:
                        spTemp += spMsg
                        msTemp += msMsg

                if math.isfinite(spTemp):
                    if method < 0:
                        sp += spTemp
                    elif method == 1:
                        sp = spTemp if math.isfinite(sp) else sp + math.log1p(math.exp(spTemp - sp))
                    elif method == 2:
                        sp = spTemp if math.isfinite(sp) else spTemp + math.log1p(math.exp(sp - spTemp))
                    else:
                        sp += math.exp(spTemp)

                if ms < msTemp:
                    ms = msTemp
            if method == 0:
                sp = math.log(sp)

            spData.append(sp)
            msData.append(ms)
        msg.spData = spData
        msg.msData = msData

    def computeLeafMsgData(self, msg, knowing, methodName='base'):
        self.computeMsgData(msg, knowing=knowing, methodName=methodName)

class VariableNode(Node):
    def __init__(self, var, graph):
        Node.__init__(self)
        self.var = var
        self.graph = graph

    def __str__(self):
        return self.var.name

    def functionLinkingTo(self, var):
        """
        Retreive the function node which link this node to another variable
        :param var: the other variable
        :type var: VariableNode, pt.RandomVariable
        :return: the corresponding function node
        :rtype: FunctionNode
        """
        if isinstance(var, VariableNode):
            var = var.var
        for fNode in self.neighbours:
            for n in fNode.neighbours:
                if n.var is var:
                    return fNode
        return None

    def possibleValuesId(self):
        if self.isObserved():
            return [self.observedValueId]
        return range(len(self.var.values))

    def valuesId(self):
        return range(len(self.var.values))

    def computeMsgData(self, msg, knowing, methodName='base'):
        spData = []
        msData = []

        method = -1
        valueIni = 1
        if methodName.startswith('exp'):
            method = 0
            valueIni = 0

        for valueId in self.valuesId():
            # Pour chaque valeur de la variable du noeud on calcule le produit des messages
            sp = valueIni
            ms = valueIni
            for m in self.msgsReceived:
                if m is None or (m.fromNode is msg.toNode):
                    continue
                if method < 0:
                    sp *= m.spData[valueId]
                    ms *= m.msData[valueId]
                else:
                    sp += m.spData[valueId]
                    ms += m.msData[valueId]
            spData.append(sp)
            msData.append(ms)

        msg.spData = spData
        msg.msData = msData

    def computeLeafMsgData(self, msg, knowing, methodName='base'):
        spData = []
        msData = []

        valueIni = 1
        if methodName.startswith('exp'):
            valueIni = 0

        for i in range(len(self.var.values)):
            spData.append(valueIni)
            msData.append(valueIni)

        msg.spData = spData
        msg.msData = msData


class Graph(object):
    def __init__(self, subset=None):
        if subset is None:
            subset = pt.RandomVariableSubSet()
        self.fctNodes = []
        self.varNodes = []
        self.varMap = col.Counter()
        self.subset = subset
        self.isCompiled = None
        self.knowing = pt.Event(subset=subset)

    def createVar(self, name, values=[True, False]):
        self.isCompiled = None
        var = pt.RandomVariable(name=name, subset=self.subset, values=values)
        self.addVar(var)
        return var

    def addVar(self, var):
        if var.subset is not self.subset:
            return None

        self.isCompiled = None
        self.varMap[var] = len(self.varNodes)
        xNode = VariableNode(var, self)
        self.varNodes.append(xNode)
        return xNode

    def xNode(self, key):
        """
        Return the variable node corresponding to the key
        :param key: index of the node,
        :type key: pt.RandomVariable, int
        :return: the variable node
        :rtype: VariableNode
        """
        if isinstance(key, pt.RandomVariable):
            return self.varNodes[self.varMap[key]]
        return self.varNodes[key]

    def addFct(self, proba, connectedVar):
        self.isCompiled = None
        xNodes = []
        for x in connectedVar:
            if x not in self.varMap:
                newVar = self.addVar(x)
                if newVar is None:
                    return None
                xNodes.append(newVar)
            else:
                xNodes.append(self.xNode(x))
        fNode = FunctionNode(proba, xNodes, graph=self)
        self.fctNodes.append(fNode)
        return fNode

    def printDef(self, show=False):
        defList = []
        for f in self.fctNodes:
            defList.append(str(f) + ": " + str(f.printDef()))

        if show:
            defStr = ""
            for s in defList:
                defStr += s + "\n"
            print(defStr)
        else:
            return defList

    def addObserved(self, event):
        newKnowing = self.knowing.replaceUnion(event)
        if not newKnowing.isMarginalUnion():
            print("Error: the observed variable should have a define value (not multiple possibilities)!")
            return self.knowing
        self.compile(knowing=newKnowing)
        return newKnowing

    def removeObserved(self, var):
        newKnowing = self.knowing.copy()
        newKnowing.space[self.subset.varIdMap[var]] = set([])
        self.compile(knowing=newKnowing)


    def resetGraph(self):
        for n in self.varNodes + self.fctNodes:
            n.reset()
        self.isCompiled = None

    def clearObserved(self):
        self.compile(knowing=pt.Event(subset=self.subset))

    def printObserved(self):
        if self.knowing.isEmpty():
            return ""
        return self.knowing.printSimplified()

    def compile(self, knowing=None, method="", verbose=None, resetAll=False):
        if knowing is not None and not knowing.isMarginalUnion():
            print("Error: the observed variable should have a define value (not multiple possibilities)!")
            return False

        if verbose is None:
            verbose = defaultCompileVerbose

        # Initiate compilation
        if self.isCompiled is not None:
            if method != "" and self.isCompiled != method:
                for x in self.varNodes + self.fctNodes:
                    x.reset()
                self.isCompiled = None
            else:
                method = self.isCompiled
        elif method == "":
            method = "base"

        if knowing is not None and knowing != self.knowing:
            for x in self.varNodes:
                observedValueId = -1
                if knowing.constraintOnVar(x.var):
                    observedValueId = next(iter(knowing.constraintOnVar(x.var)))
                if observedValueId != x.observedValueId:
                    self.isCompiled = None
                    x.observedValueId = -1
                    x.reset(recursive=True)
                    x.observedValueId = observedValueId
            self.knowing = knowing.copy()
        else:
            knowing = self.knowing

        if not resetAll and self.isCompiled is not None:
                return True

        # Compute first messages if needed
        msgs = []
        for n in self.varNodes + self.fctNodes:
            if resetAll:
                n.reset()

            computedMsgs = n.processLeafMsgs(knowing=knowing, method=method)
            for m in computedMsgs:
                msgs.append(m)

        loopCounter = 1
        while msgs:
            if verbose:
                print("")
                print(str(loopCounter)+":\t" + str(msgs[0]))
                for m in msgs[1:]:
                    print("\t"+str(m))

            # Send messages
            for m in msgs:
                m.send()

            # Compute new messages
            msgs = []
            for n in self.varNodes + self.fctNodes:
                computedMsgs = n.processMsgs(knowing=knowing, method=method)
                for m in computedMsgs:
                    msgs.append(m)
            loopCounter += 1

        if loopCounter!=1:
            for x in self.varNodes:
                x.finishJob(methodName=method)

        self.isCompiled = method
        return True

    def probaMarginal(self, var, value, knowing=None):
        if not self.compile(knowing=knowing):
            return float('nan')

        xNode = self.xNode(var)
        if isinstance(value, bool) or (not isinstance(value, int)):
            value = xNode.var.valInternalId[value]


        proba = xNode.spData[value]
        return proba

    def p(self, event, knowing=None):
        if knowing is None:
            knowing = pt.Event(subset=self.subset)
        if not event.isMarginal():
            print("Error: event must correspond to a marginal probability")
            return float('nan')
        e = event.export()
        var = e[0][0]
        value = e[0][1][0]
        return self.probaMarginal(var, value, knowing=knowing)



    def mostProbableConfiguration(self, knowing=None):
        self.compile(knowing=knowing)

        event = pt.Event(subset=self.subset)
        probability = []

        for xNode in self.varNodes:
            max = None
            e = None
            for valueId in xNode.valuesId():
                p = xNode.msData[valueId]
                if max is None or max < p:
                    max = p
                    e = (xNode.var==valueId)
            event &= e
            probability.append(max)

        return [event, probability]

    def MPC(self, knowing=None, eventFormat='shortStr', withProbabilities=False):
        mpc = self.mostProbableConfiguration(knowing)
        if eventFormat=='vec' or eventFormat=='value' or eventFormat=='shortStr' or eventFormat=='str':
            mpc[0] = [next(iter(set)) for set in mpc[0].space]
            if eventFormat != 'vec':
                mpc[0] = [self.subset.varList[varId].values[valId] for varId, valId in enumerate(mpc[0])]
                if eventFormat=='shortStr':
                    mpc[0] = [self.subset.varList[varId].printEqTo(val, shortStr=True) for varId, val in enumerate(mpc[0])]
                if eventFormat == 'str':
                    mpc[0] = [self.subset.varList[varId].printEqTo(val, shortStr=False) for varId, val in enumerate(mpc[0])]

        if withProbabilities:
            return mpc
        else:
            return mpc[0]

    def drawGraph(self, show=False, rotate=False, engine='dot', size='', mpc=False):
        # Init graph
        dot = Digraph(comment='Factor Graph', engine=engine)
        dot.edge_attr.update(arrowhead='none')
        dot.graph_attr.update(rotation='-90'if rotate else '0')

        # Update graph style
        pointSize = 0.8
        if engine=='sfdp':
            if size=='':
                size = '5,5'
            dot.graph_attr.update(size=size)
            pointSize = 4

        penwidth = str(int(pointSize * 2.2))
        dot.node_attr.update(penwidth=penwidth, fontsize=str(pointSize*18))
        dot.edge_attr.update(penwidth=penwidth)

        if mpc:
            mpc = self.MPC(eventFormat='str')

        # Draw variable nodes
        for node in self.varNodes:
            style = 'filled' if node.isObserved() else 'solid'
            if mpc:
                label = mpc[self.subset.varIdMap[node.var]]
            else:
                label = node.var.name if not node.isObserved() else node.var.printEqTo(node.observedValueId, shortStr=False)
            dot.node(str(node), label=label, style=style, width=str(pointSize), height=str(pointSize*0.6))
        # Draw function nodes
        for node in self.fctNodes:
            dot.node(str(node), shape="rect", style="filled", color="black", fixedsize="true", width=str(pointSize*0.7), height=str(pointSize*0.15), label="")

        # Draw edges
        currentNodes = [f for f in self.fctNodes if len(f.neighbours)==1]
        processedNodes = set([])
        while currentNodes:
            nextNodes = set([])
            for node in currentNodes:
                processedNodes.add(node)
                for child in node.neighbours:
                    if child not in processedNodes:
                        dot.edge(str(node), str(child))
                        nextNodes.add(child)
            currentNodes = list(nextNodes)

        dot.render('output/factorGraph.gv', view=show)
        return dot

    def drawMPC(self, show=False, rotate=False, engine='dot', size=''):
        return self.drawGraph(show=show, rotate=rotate, engine=engine, size=size, mpc=True)

def graphFromBayesNet(bayesNet):
    """
    Construct a Factor Graph from a Bayes Network
    :param bayesNet: the Bayes Network to convert
    :type bayesNet: bNet.BayesNetwork
    :return: the converted Factor Graph
    :rtype: FactorGraph
    """
    graph = Graph(bayesNet.subset)

    for n in bayesNet.nodes:
        varList = [e.var for e in n.parents] + [n.var]


        f = graph.addFct(n.proba, varList)
        # print(str(f) + " : " + str(n.proba))

    return graph
