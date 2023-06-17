import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import methods as met
import gauss_transformer as tr
import matrix_elements as mat
from scipy.optimize import minimize
from scipy.integrate import quad

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
ngausP=3
ngausD=1
dimP=1
dimG=3
bmax=5
params=np.array([b,S])
w=tr.w_gen_2pion(mpi,mpi)
alphaD=[]
alphaDCoul=[]

#---------------WITH COULOMB--------------------

dictDC=met.global_minDCoul(ngausP,ngausD,dimP,dimG,bmax,masses,params)

cDC=dictDC['eigenvectors'][-1]
coordsDC=dictDC['coords'][-1]

print('Fuld D coords:', coordsDC)
print('Full eigenvectors:', cDC)
masses_min=dictDC['masses']
w=tr.w_gen_2pion(masses_min[1],masses_min[1])
K=met.K_gen(masses_min[1],masses_min[1],masses_min[0])

bsP=coordsDC[:ngausP]
bsD=coordsDC[ngausP:]

for j in range(0,len(bsD),dimG):
        AD=tr.A_generate(bsD[j:j+dimG],w)
        alphaDCoul.append(AD)
alphaP=[1/(b**2) for b in bsP]
cPlotD=cDC[:ngausP+1]

NC,HC=mat.PionTwoCoulomb(alphaP,alphaDCoul,params,masses,w,K)

print('N Matrix:', NC, '\n')
print('H Matrix:', HC, '\n')

print('Contribution of of two-pion part with Coulomb:', cDC[-1]*NC[-1,-1]*cDC[-1])

print('Checking the energy of the wavefunction:', dictDC['E_list'][-1])
#print('Checking the norm is unity:', cDC.T@NC@cDC)

#------------------WITHOUT COULOMB----------------------------

dictD=met.global_minD(ngausP,ngausD,dimP,dimG,bmax,masses,params)

cD=dictD['eigenvectors'][-1]
coordsD=dictD['coords'][-1]
masses_min=dictD['masses']
w=tr.w_gen_2pion(masses_min[1],masses_min[1])
K=met.K_gen(masses_min[1],masses_min[1],masses_min[0])


print('Fuld D coords:', coordsD)
print('Full eigenvectors:', cD)

bsP=coordsD[:ngausP]
bsD=coordsD[ngausP:]

print('Supposed P-wave coords:', bsP)
print('Supposed D-wave coords:', bsD)

for j in range(0,len(bsD),dimG):
        AD=tr.A_generate(bsD[j:j+dimG],w)
        alphaD.append(AD)
alphaP=[1/(b**2) for b in bsP]
cPlotD=cD[:ngausP+1]

N,H=mat.PionTwo(alphaP,alphaD,params,masses,w,K)

print('N Matrix:', N, '\n')
print('H Matrix:', H, '\n')

print('Contribution of of two-pion part with Coulomb:', cD[-1]*N[-1,-1]*cD[-1])

print('Checking the energy of the wavefunction:', dictD['E_list'][-1])
#print('Checking the norm is unity:', cD.T@N@cD)