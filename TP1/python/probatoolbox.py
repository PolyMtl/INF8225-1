import numpy as np
import collections as col

class RandomVariableSubSet(object):
    def __init__(self):
        self.varList = []
        self.varIdMap = col.Counter()
        self.space = []

    def addRndVar(self, rndVar):
        self.varList.append(rndVar)
        self.varIdMap[rndVar] = len(self.varList)-1
        self.space.append(set(range(0, len(rndVar.values))))

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.varIdMap[item]
        return self.varList[item]

    def __len__(self):
        return len(self.varList)

    def size(self):
        return len(self)

    def extractSpace(self, spaceValue, toVariables):
        space = []
        for v in toVariables:
            if isinstance(v, RandomVariable):
                v = self.varIdMap[v]
            try:
                space.append(spaceValue[v].copy())
            except AttributeError:
                space.append(spaceValue[v])
        return space

    def expandSpace(self, spaceValue, fromVariables):
        space = []
        for s in self.space:
            space.append(s.copy())
        for i, v in enumerate(fromVariables):
            if isinstance(v, RandomVariable):
                v = self.varIdMap[v]
            try:
                space[v] = spaceValue[i].copy()
            except AttributeError:
                space[v] = spaceValue[i]
        return space

    def emptySpace(self):
        space = []
        for i in range(len(self.space)):
            space.append(set([]))
        return space

defaultSubset = RandomVariableSubSet()

class RandomVariable(object):

    def __init__(self, name, subset, values=[True, False]):
        self.name = name
        self.values = values
        self.valInternalId = col.Counter()
        for i, v in enumerate(values):
            self.valInternalId[v] = i

        # Stockage et ajout au subset
        self.subset = subset
        subset.addRndVar(self)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def printEqTo(self, value, shortStr=True):
        if isinstance(value, int) and not isinstance(value, bool):
            value = self.values[value]

        if shortStr:
            if isinstance(value, bool):
                if value:
                    return self.name.upper()
                else:
                    return self.name.lower()
        return self.name+"="+str(value)

    def __eq__(self, other):
        if isinstance(other, RandomVariable):
            return self.subset is other.subset and self.values == other.values and self.name == other.name
        return Event.varEq(self, other)

    def __ne__(self, other):
        eq = (self==other)
        if isinstance(eq, bool):
            return not eq
        return ~eq


class Event(object):
    def __init__(self, space=[], subset=None):
        assert not isinstance(space, RandomVariableSubSet)

        if subset is None:
            subset=defaultSubset
        self.subset = subset

        if space is not None and not space:
            space = [0 for i in range(len(subset.space))]

        if space is not None and len(space)==len(subset.space):
            self.space = Event.readSpace(space.copy(), subset)
            self.isValid = True
        else:
            self.space = space
            self.isValid = False

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Event(self.space.copy(), self.subset)

    def varEq( var, value):
        space = []
        for v in var.subset.varList:
            if v is var:
                if isinstance(value, int) and not isinstance(value, bool):
                    value = set([value])
                space.append(value)
            else:
                space.append(0)
        return Event(space, var.subset)

    def eventFromSpace(space, base):
        if not base:
            return Event()
        subset = base[0].subset
        space = subset.expandSpace(space, base)
        return Event(space=space, subset=subset)

    @staticmethod
    def readSpace(space, rndVarSet):
        resultSpace = []
        for varId, s in enumerate(space):
            sSet = set()
            if isinstance(s, int) and not isinstance(s, bool):
                if s > 0:
                    sSet.add(s)
                elif s == 0:
                    sSet = set([])
            elif isinstance(s, list):
                for l in s:
                    if isinstance(l, int):
                        sSet.add(l)
                    else:
                        sSet.add(rndVarSet[varId].valInternalId[l])
            elif not isinstance(s, set):
                sSet.add(rndVarSet[varId].valInternalId[s])
            else:
                sSet = s
            resultSpace.append(sSet)
        return resultSpace

    def __str__(self):
        if not self.isValid:
            return "Event(invalid)"
        str = "Event(" + self.printVars() + ")"
        return str

    def printVars(self):
        r = ""
        compSpace = (~self).space
        for varId, var in enumerate(self.subset.varList):
            if self.space[varId]:
                if len(r)>0:
                    r += " & "

                r += var.name
                if len(self.space[varId]) <= len(compSpace[varId]):
                    r += "="
                    space = self.space[varId]
                else:
                    r += "!="
                    space = compSpace[varId]
                for i, valueId in enumerate(space):
                    if i != 0:
                        r += ","
                    r += str(var.values[valueId])
        return r

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.subset is other.subset and self.space == other.space

    @staticmethod
    def spaceUnion(to, other, default):
        for i, s in enumerate(other):
            to[i] = to[i] | s
        return to

    @staticmethod
    def spaceReplaceUnion(to, other, default):
        for i, s in enumerate(other):
            if s:
                to[i] = s
        return to

    @staticmethod
    def spaceSubstract(to, other, default):
        for i, s in enumerate(other):
            if len(s)>0 and not to[i]:
                to[i] = default[i]
            to[i] = to[i] - s
        return to

    @staticmethod
    def spaceIntersect(to, other, default):
        for i, s in enumerate(other):
            if not s:
                continue
            elif not to[i]:
                to[i] = s
                continue

            to[i] = to[i] & s
            if not to[i]:
                to[i] = None
                return None
        return to


    def __add__(self, other):
        if not self.checkOperation(other):
            return self.__copy__()
        return Event(Event.spaceUnion(self.space.copy(), other.space, self.subset.space), subset=self.subset)

    def __radd__(self, other):
        return self.__add__(other)

    def __or__(self, other):
        return self.__add__(other)

    def __and__(self, other):
        if not self.checkOperation(other):
            return self.__copy__()
        return Event(Event.spaceIntersect(self.space.copy(), other.space, self.subset.space), subset=self.subset)

    def __rand__(self, other):
        return self.__and__(other)

    def __iadd__(self, other):
        if not self.checkOperation(other):
            return
        if Event.spaceUnion(self.space, other.space, self.subset.space) is None:
            self.isValid = False

    def __sub__(self, other):
        if not self.checkOperation(other):
            return self.__copy__()
        return Event(Event.spaceSubstract(self.space.copy(), other.space, self.subset.space), subset=self.subset)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __isub__(self, other):
        if not self.checkOperation(other):
            self.__copy__()
        Event.spaceSubstract(self.space, other.space, self.subset.space)

    def replaceUnion(self, other):
        if not self.checkOperation(other):
            return self.__copy__()
        return Event(Event.spaceReplaceUnion(self.space.copy(), other.space, self.subset.space), subset=self.subset)

    def checkOperation(self, other):
        return self.subset is other.subset and self.isValid and other.isValid

    def __invert__(self):
        if not self.isValid:
            return self.__copy__()
        emptySpace = self.subset.emptySpace()
        Event.spaceSubstract(emptySpace, self.space, self.subset.space)
        return Event(emptySpace, subset=self.subset)

    def listVecEvents(self, base=None, replaceIdWithCounter=False):
        if base is None or not base:
            space = self.space
        else:
            # Generation du sous-espace space issue de self.space
            space = self.subset.extractSpace(self.space, base)

        end = True
        currentEvent = []
        varyingVar = []
        for varId, values in enumerate(space):
            if not values:
                values = self.subset.space[varId]
                space[varId] = values
            if len(values) > 1:
                varyingVar.append(varId)

            firstValue = 0 if replaceIdWithCounter else next(iter(values))
            currentEvent.append(firstValue)

        if not currentEvent:
            return []
        end = not varyingVar

        # Iteration a travers tous les elements
        rList = [currentEvent]
        while(not end):
            found = False;
            for varId in varyingVar:

                # On tente d'incrementer sa valeur actuelle
                currentV = currentEvent[varId] + 1

                if replaceIdWithCounter:
                    # Si le compteur actuel est inferieur a la taille de l'ensemble des valeurs possibles pour
                    # cette variable, on a trouve notre nouveau compteur
                    found = currentV < len(space[varId])

                else:
                    # On lit les valeurs que la variable actuelle peut prendre
                    values = space[varId]
                    # On recupere la plus grande valeur possible pour cette variable
                    for lastV in values:
                        continue

                    # Si la nouvelle valeur est egale a la plus grande valeur possible, on a trouve notre evenement suivant
                    found = currentV==lastV
                    # Sinon on verifie que la nouvelle valeur est acceptable, le cas echeant on cherche une nouvelle valeur acceptable
                    while lastV >= currentV and not found:
                        if values & set([currentV]):
                            found = True
                        else:
                            currentV+=1

                # Si l'evenement a ete trouve on le modifie selon la nouvelle valeur de la variable
                if found:
                    currentEvent[varId] = currentV
                    break
                # sinon on affecte la premiere valeur possible a cette variable et on passe a la suivante
                else:
                    currentEvent[varId] = 0 if replaceIdWithCounter else next(iter(space[varId]))

            if found:
                rList.append(list(currentEvent))
            else:
                end = True

        return rList

    def listEvents(self, base=[]):
        if not base:
            base = self.subset.varList
        list = []
        for event in self.listVecEvents(base):
            eventSet = [set([e]) for e in event]
            eventSet = self.subset.expandSpace(eventSet, base)
            list.append(Event(space=eventSet, subset=self.subset))
        return list

    def listMarginalUnionEvents(self):
        base = []
        for varId, varSet in enumerate(self.space):
            if varSet:
                base.append(varId)

        return self.listEvents(base=base)

    def listStrEvents(self, base=[]):
        if not base:
            base = self.subset.varList
        list = []
        for e in self.listVecEvents(base):
            list.append(self.eventToStr(e, base))
        return list

    def eventToStr(self, event=[], base=[]):
        if not base:
            base = self.subset.varList
        if not event:
            event = self

        events = [event]
        if isinstance(event, Event):
            events = event.listVecEvents(base=base)

        list =[]
        for e in events:
            definition = base[0].printEqTo(base[0].values[e[0]])
            for i, v in enumerate(base[1:]):
                definition += " " + v.printEqTo(v.values[e[i + 1]])
            list.append(definition)

        if len(list)==1:
            return list[0]
        return list

    def constraintOnVar(self, var):
        varId = self.subset.varIdMap[var]
        return self.space[varId]

    def isMarginal(self):
        constraintFound = False
        for s in self.space:
            if s is None or len(s)>1:
                return False
            elif s:
                if constraintFound:
                    return False
                constraintFound = True
        return constraintFound

    def isMarginalUnion(self):
        for s in self.space:
            if s is None or len(s) > 1:
                return False
        return True

    def isSingleEvent(self):
        for s in self.space:
            if s is None or not s or len(s) > 1:
                return False
        return True

    def isEmpty(self):
        if not self.isValid:
            return True
        for s in self.space:
            if s:
                return False
        return True

    def export(self):
        varList = []
        for varId, varSet in enumerate(self.space):
            if varSet:
                l = [self.subset.varList[varId], []]
                l[1] = list(varSet)
                varList.append(l)
        return varList