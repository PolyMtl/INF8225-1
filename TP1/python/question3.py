from tabulate import tabulate
import BayesNetwork as bn
import FactorGraph as fg

bayesNet = bn.Network()
C = bayesNet.createNode("C", 0.001)
T = bayesNet.createNode("T", 0.002)
A = bayesNet.createNode("A", [0.95, 0.94, 0.29, 0.001], parents=[T, C])
M = bayesNet.createNode("M", [0.90, 0.05], parents=A)
J = bayesNet.createNode("J", [0.70, 0.01], parents=A)

graph = fg.graphFromBayesNet(bayesNet)
graph.drawGraph(show=True)

toCompute = [(M==True)&(J==False), (M==False)&(J==True), (M==True)&(J==True)]
toCompute+= [(M==False)&(J==False), M==True, J==True]
for knowing in toCompute:
    p = graph.p(C==True, knowing)
    print("p(C=True|"+knowing.printVars()+") = " + str(round(p*100,3)) + "%")
print("")
for node in graph.varNodes:
    var = node.var
    p = graph.p(var == True)
    print("p(" + var.name + "=True) = " + str(round(p * 100, 3)) + '%')