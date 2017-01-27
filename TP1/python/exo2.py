import BayesNetwork as bn
import FactorGraph as fg

bnet = bn.Network()

C = bnet.addNode("C", 0.001)
T = bnet.addNode("T", 0.002)
A = bnet.addNode("A", [0.95,0.94,0.29,0.001], parents=[T,C])
M = bnet.addNode("M", [0.90,0.05], parents=A)
J = bnet.addNode("J", [0.70,0.01], parents=A)

graph = fg.graphFromBayesNet(bnet)

graph.compile(knowing=(A==True), method='exp')



p1 = bnet.p(C==True, (M==True)&(J==False))
p2 = bnet.p(C==True, (M==False)&(J==True))
p3 = bnet.p(C==True, (M==True)&(J==True))
p4 = bnet.p(C==True, (M==False)&(J==False))

p5 = bnet.p(C==True, M==True)
p6 = bnet.p(C==True, J==True)
