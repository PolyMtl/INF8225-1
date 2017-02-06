import BayesNetwork as bn

# --- EXPLAINING AWAY ---
explAwayNet = bn.Network("Explaining Away Net")
A = explAwayNet.createNode('A', 0.27)
B = explAwayNet.createNode('B', 0.11)
C = explAwayNet.createNode('C', [0.9,0.7,0.6,0.1], parents=[A, B])
explAwayNet.drawGraph(observed=[C])

# Pour l'explication de l'algorithme de calcul, voir la question 2
p = explAwayNet.p(A==True)
print("P(A)=" + str(round(p,4)*100) + '%')
p = explAwayNet.p(A==True, C==True)
print("P(A|C)=" + str(round(p,4)*100) + '%')
p = explAwayNet.p(A==True, (C==True)&(B==True))
print("P(A|C,B)=" + str(round(p,4)*100) + '%')
print("")
p = explAwayNet.p(A==True, C==True) * explAwayNet.p(B==True, C==True)
print("P(A|C)*P(B|C)=" + str(p))
p = explAwayNet.p((A==True)&(B==True), C==True)
print("P(A,B|C)=" + str(p))


print("")
print(25*'_')
# --- SERIAL BLOCKING ---
serialBlockNet = bn.Network("Serial Blocking Net")
A = serialBlockNet.createNode('A', 0.80)
B = serialBlockNet.createNode('B', [0.99, 0.13], parents=A)
C = serialBlockNet.createNode('C', [0.2, 0.7], parents=B)
serialBlockNet.drawGraph(observed=[B], engine='sfdp')

p = serialBlockNet.p(A==True, B==True)
print("P(A|B)=" + str(round(p,4)*100) + '%')
p = serialBlockNet.p(A==True, (C==True)&(B==True))
print("P(A|B,C)=" + str(round(p,4)*100) + '%')
print("")
p = serialBlockNet.p(A==True, B==True) * serialBlockNet.p(C==True, B==True)
print("P(A|B)*P(C|B)=" + str(p))
p = serialBlockNet.p((A==True)&(C==True), B==True)
print("P(A,C|B)=" + str(p))

print("")
print(25*'_')
# --- DIVERGENT BLOCKING ---
serialBlockNet = bn.Network("Divergent Blocking Net")
B = serialBlockNet.createNode('B', 0.50)
A = serialBlockNet.createNode('A', [0.89, 0.17], parents=B)
C = serialBlockNet.createNode('C', [0.23, 0.72], parents=B)
serialBlockNet.drawGraph(observed=[B])

p = serialBlockNet.p(A==True, B==True)
print("P(A|B)=" + str(round(p,4)*100) + '%')
p = serialBlockNet.p(A==True, (C==True)&(B==True))
print("P(A|B,C)=" + str(round(p,4)*100) + '%')
print("")
p = serialBlockNet.p(A==True, B==True) * serialBlockNet.p(C==True, B==True)
print("P(A|B)*P(C|B)=" + str(p))
p = serialBlockNet.p((A==True)&(C==True), B==True)
print("P(A,C|B)=" + str(p))