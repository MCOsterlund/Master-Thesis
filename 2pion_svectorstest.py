from svector_class import svector
import numpy as np
import matrix_elements as mat
import gauss_transformer as tr
import methods as met

b=3.9
S=41.5 #Coupling strength parameters
E0=0
mp=938.27 #Mev
mn=939.57
mpi0=134.98 #Mev
mpiC=139.57
mpi=(mpi0+mpiC)/2
mbare=(mp+mn)/2
hc=197.3 #hbar*c
dimP=1
bmax=5

mpi=(mpi0+mpiC)/2
mbare=(mp+mn)/2 ##Resetting the masses to their starting values.
masses=np.array([mbare,mpi])
params=np.array([b,S])
dimP=1
dimD=1
bmax=3
ngausP=2
ngausD=1
K=met.K_gen(masses[1],masses[1],masses[0])

bs=np.array([1])
Ap=[1/(bis**2) for bis in bs]

wlist=tr.w_gen_2pion(mpi,mpi)

A=np.array([[1,2],[1,4]])
AD=[]
AD.append(A)

O1,K1=mat.PionTwo(Ap,AD,params,masses,wlist,K)
O2,K2=mat.PionTwo_Test(Ap,AD,params,masses,wlist,K)

print(O1[-1,-1])
print(O2[-1,-1])
print(O1[-1,-1]-O2[-1,-1])
print(np.array_equal(O1,O2))


    




