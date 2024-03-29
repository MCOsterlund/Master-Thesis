import numpy as np
import matplotlib.pyplot as plt

E_list=[-85.52435455378392,
-73.48186910282541,
-99.65043660912295,
-116.23006012930506,
-158.55499591108358,
-216.78771021337062,
-398.43555053817914,
-585.5874604154711,
-65.6099883653138,
-398.43555053817914,
-65.60998836529822,
-513.8716589689977,
-462.069913906701,
-562.0718548805843,
-561.1823248987175,
-578.1916106562189,
-581.0846363175211,
-585.0376743936724,
-583.168072452526,
-585.7104207173535,
-585.0376743936724,
-585.7450137411092,
-585.5874604154711,
-585.7521563928191,
-585.7104207173533,
-585.7546420563301,
-585.7450137411092,
-585.7549200604045,
-585.7521563928191,
-585.7551604430217,
-585.7546420563301,
-585.7551352008683,
-585.7549959928658,
-585.7551715463566,
-585.7551352008685,
-585.7551719241827,
-585.7551604430215,
-585.7551732178445,
-585.7551715463566,
-585.7551729416318]
iterations=np.linspace(1,len(E_list)+1,len(E_list))
comp_final=[-585.7551729416318]*len(E_list)


SMALL_SIZE = 12
INTERMEDIATE_SIZE=14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=MEDIUM_SIZE
plt.rcParams['legend.fontsize']=INTERMEDIATE_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=SMALL_SIZE
plt.rcParams['ytick.labelsize']=SMALL_SIZE

ticksY=[0,-50,-100,-150,-200,-250,-300,-350,-400,-450,-500,-550,-600]
plt.plot(iterations,E_list, label='Minimizer energies')
plt.plot(iterations,comp_final, '--', label='Final energy')
plt.yticks(np.arange(min(ticksY),max(ticksY),50))
plt.ylim([-600,-50])
plt.legend()
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Energy [MeV]')
plt.savefig('figures/1pionIterations.pdf'.format(0),bbox_inches='tight')
plt.show()