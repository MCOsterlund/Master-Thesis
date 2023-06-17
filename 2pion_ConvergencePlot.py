import numpy as np
import matplotlib.pyplot as plt

##DATA RECIEVED BY RUNNING "DressedNucleon2.py"

gauss=[1,2,3]

E1=[-585.75517319, -585.7569118,  -585.75667412]
E2=[-589.94872089, -589.95038543 ,-589.95036204]

format_EP1=np.around(E1[-1],1)
format_ED2=np.around(E2[-1],)
b=3.9
S=41.5

SMALL_SIZE = 12
INTERMEDIATE_SIZE=14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=MEDIUM_SIZE
plt.rcParams['legend.fontsize']=INTERMEDIATE_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=SMALL_SIZE
plt.rcParams['ytick.labelsize']=SMALL_SIZE
ticksX=[1,2,3,4]
ticksY=[-585.6,-585.7,-585.8]

plt.figure(0)
plt.plot(gauss,E1, marker='.')
#plt.plot(gauss,E2, marker='.')
plt.xticks(np.arange(min(ticksX),max(ticksX),1))
plt.yticks(np.arange(min(ticksY),max(ticksY),0.05))
plt.ticklabel_format(style='scientific',axis='y')
plt.ylim([-585.7,-585.8])
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [MeV]')
plt.text(1.5,-586.4,'S=%s MeV, b=%s fm' %(S,b), fontsize=12)
plt.text(1.5,-586.7,'1 Pion energy: %s MeV' %(format_EP1), fontsize=12)
plt.text(1.5,-587,'2 Pion energy: %s MeV' %(format_ED2), fontsize=12)
plt.legend(['1 Pion system', '2 Pion system'])
plt.savefig('figures/2Pions_Energy_Convergence.pdf'.format(0),bbox_inches='tight')
plt.show()