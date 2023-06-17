import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=MEDIUM_SIZE
plt.rcParams['legend.fontsize']=SMALL_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=SMALL_SIZE
plt.rcParams['ytick.labelsize']=SMALL_SIZE

S_list2=np.linspace(0,80,100)
E_comp=np.linspace(0.1,0.6,100)

plt.figure(0)
plt.title('Energy comparison as a function of coupling strength')
plt.yticks(np.arange(min(E_comp),max(E_comp+0.1),0.1))
plt.xticks(np.arange(min(S_list2),max(S_list2+10),10))
plt.xlabel(r'$S_W$ [MeV]')
plt.ylabel(r'$|E_2-E_1| \quad [MeV]$')
plt.plot(S_list2,E_comp,marker='o')
plt.savefig('figures/PlotTest.pdf'.format(0),bbox_inches='tight')
plt.show()

S_list2=[0,1,2,3,4,5]
E_comp=[6,7,8,9,10]
    
with open('plottest.txt', 'w') as file:
    file.write('List 1: \n')
    for item in S_list2:
        file.write(str(item) + '\n')

    file.write('---\n')

    file.write('List 2: \n')
    for item in E_comp:
        file.write(str(item)+'\n')