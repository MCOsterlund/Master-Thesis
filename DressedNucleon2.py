import numpy as np
import matplotlib.pyplot as plt
import gauss_transformer as tr
import methods as met
import matrix_elements as mat
from scipy.optimize import minimize
from scipy.linalg import eigh
from svector_class import svector

##Constants and parameters
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
ngausP=3
ngausD=3

bij=np.array([])
bs1=[]
wlist=tr.w_gen_2pion(mpi,mpi)

###---------------------JACOBI'S FORMULA----------------------------------


###----------------------MATRIX ELEMENT METHOD----------------------------

EnergyOnePion=met.global_minP(ngausP,1,bmax,masses,params)
EPi1=EnergyOnePion['E_list']
gaussians1=EnergyOnePion['gaussians']
coordsP=['coords'][-1]
eigP= ['eigenvectors'][-1]

print('Energies from one pion approximation:', EPi1)

dictD=met.global_minD(ngausP,ngausD,1,3,bmax,masses,params)
ED2=dictD['E_list']
gaussians2=dictD['gaussians']
gauss2_plot=gaussians2[3:]
E2_plot=ED2[3:]
coordsD=dictD['coords']
print('Energies from two pion approximation:', ED2)

format_EP1=np.around(EPi1[-1],1)
format_ED2=np.around(ED2[-1],)

SMALL_SIZE = 12
INTERMEDIATE_SIZE=14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=MEDIUM_SIZE
plt.rcParams['legend.fontsize']=INTERMEDIATE_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=SMALL_SIZE
plt.rcParams['ytick.labelsize']=SMALL_SIZE

plt.figure(0)
plt.plot(gauss2_plot,E2_plot, marker='.')
plt.plot(gaussians1,EPi1, marker='.')
plt.xticks(0,4,1)
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [MeV]')
plt.text(1.5,-586.4,'S=%s MeV, b=%s fm' %(S,b), fontsize=12)
plt.text(1.5,-586.7,'1 Pion energy: %s MeV' %(format_EP1), fontsize=12)
plt.text(1.5,-587,'2 Pion energy: %s MeV' %(format_ED2), fontsize=12)
plt.legend(['2 Pion system', '1 Pion system'])
plt.savefig('figures/2Pions_Energy_Convergence.pdf'.format(0),bbox_inches='tight')
plt.show()