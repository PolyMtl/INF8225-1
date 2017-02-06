from tabulate import tabulate
import BayesNetwork as bn
import FactorGraph as fg

bnet = bn.Network()
bnet.defaultComputeMethod = "exp2"

R = bnet.createNode("R", 0.5)
C1 = bnet.createNode("C1", [0.99,0.001],parents=R)
C2 = bnet.createNode("C2", [0.001,0.99],parents=R)
T = bnet.createNode("T", [0.05, 0.99, 0.0001, 0.05], parents=[C1,C2])


C = bnet.createNode("C", 0.001)
T = bnet.createNode("T", 0.002)
A = bnet.createNode("A", [0.95, 0.94, 0.29, 0.001], parents=[T, C])
M = bnet.createNode("M", [0.90, 0.05], parents=A)
J = bnet.createNode("J", [0.70, 0.01], parents=A)

graph = fg.graphFromBayesNet(bnet)
graph.compile(method='exp')

graph.addObserved(A==True)


knowing = [(M==True)&(J==False), (M==False)&(J==True), (M==True)&(J==True), (M==False)&(J==False)]
knowing+= [M==True, J==True]
for _ in knowing:
    print("p(C=True|"+_.printVars()+") = " + str(round(graph.p(C==True, _)*100,2)) + "%")