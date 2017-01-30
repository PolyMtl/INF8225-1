from tabulate import tabulate
import BayesNetwork as bn
import FactorGraph as fg

bnet = bn.Network()
bnet.defaultComputeMethod = "exp2"

C = bnet.createNode("C", 0.001)
T = bnet.createNode("T", 0.002)
A = bnet.createNode("A", [0.95, 0.94, 0.29, 0.001], parents=[T, C])
M = bnet.createNode("M", [0.90, 0.05], parents=A)
J = bnet.createNode("J", [0.70, 0.01], parents=A)

graph = fg.graphFromBayesNet(bnet)
graph.addObserved(A==True)
graph.compile(method='exp')

knowing = [(M==True)&(J==False), (M==False)&(J==True), (M==True)&(J==True), (M==False)&(J==False)]
knowing+= [M==True, J==True]

probas = []
for k in knowing:
    p = []
    p.append("p( C=True | "+k.printSimplified()+")")
    p.append(bnet.p(C==True, k))
    p.append(graph.p(C==True, k))
    probas.append(p)

print(tabulate(probas, headers=['Proba', 'Bayes', 'SumProduct']))