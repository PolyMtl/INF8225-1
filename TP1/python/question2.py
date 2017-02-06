from tabulate import tabulate
import BayesNetwork as bn

bayesNet = bn.Network()
C = bayesNet.createNode("C", 0.001)
T = bayesNet.createNode("T", 0.002)
A = bayesNet.createNode("A", [0.95, 0.94, 0.29, 0.001], parents=[T, C])
M = bayesNet.createNode("M", [0.90, 0.05], parents=A)
J = bayesNet.createNode("J", [0.70, 0.01], parents=A)
bayesNet.drawGraph(show=True)
bayesNet.defaultComputeMethod = 'exp2'

p = bayesNet.p(A==True, (T==True)&(C==False))
print("P(A|T,!C)=" + str(round(p,4)))
p = bayesNet.p(J==True, A==True)
print("P(J|A)=" + str(round(p,4)))
p = bayesNet.p(C==True)
print("P(C)=" + str(round(p,4)))
p1 = round( bayesNet.p(C==True, A==True) * bayesNet.p(T==True, A==True), 5)
p2 = round( bayesNet.p((C==True)&(T==True), A==True), 5)
print("P(C|A)*P(T|A)=" + str(p1) + " differe de p(C,T|A)="+ str(p2))

print("")
print(25*'_')
# --- b. DISTRIBUTION ---
print([str(round(_*100,2))+'%' for _ in bayesNet.distribution()])
bayesNet.drawDistribution(show=True) # Affiche l'histogramme

print("")
print(25*'_')
# --- c. PROBABILITES MARGINALES Conditionelles ---
event = (C==True)&(A==False) # Impose C=0 et A=1 et retourne un objet Event
for _ in event.listVecEvents():
    print(_)                 # Les vecteurs ont comme base [C,T,A,M,J]

print("")
toCompute = [(M==True)&(J==False), (M==False)&(J==True), (M==True)&(J==True)]
toCompute+= [(M==False)&(J==False), M==True, J==True]
for knowing in toCompute:
    p = bayesNet.p(C==True, knowing)
    print("p(C=True|"+knowing.printVars()+") = " + str(round(p*100,3)) + "%")


print("")
print(25*'_')
# --- d. PROBABILITES MARGINALES Inconditionelles ---
for node in bayesNet.nodes:
    var = node.var
    p = bayesNet.p(var==True)
    print("p("+var.name+"=True) = "+str(round(p*100,3))+'%')

print("")
pC = 0.001
pT = 0.002
pA = 0.95*pT*pC + 0.94*(1-pT)*pC + 0.29*pT*(1-pC) + 0.001*(1-pT)*(1-pC)
pM = 0.90*pA + 0.05*(1-pA)
pJ = 0.70*pA + 0.01*(1-pA)
for _ in [[pC,'C'],[pT,'T'],[pA,'A'],[pM,'M'],[pJ,'J']]:
    print("p(" + _[1] + "=True) = " + str(round(_[0] * 100, 3)) + '%')