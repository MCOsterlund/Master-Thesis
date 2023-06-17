import numpy as np
import matplotlib.pyplot as plt
import methods as met
import gauss_transformer as tr
import matrix_elements as mat
import math

ngausP=1
ngausD=1

S=20
E0=0
mp=938.27 #Mev
mn=939.57
mpi0=134.98 #Mev
mpiC=139.57
mpi=(mpi0+mpiC)/2
mbare=(mp+mn)/2
hc=197.3 #hbar*c
dimP=1
dimG=3
bmax=5
w=tr.w_gen_2pion(mpi,mpi)
b_list=[0.01,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

E_comp=[]
E_ref=[]
b_list2=[]

O1_list=[]
O2_list=[]
O3_list=[]

for i in range(len(b_list)):
    b=b_list[i]
    print('Status:', i,b)

    params=np.array([b,S])
    masses=np.array([mbare,mpi])

    dictD=met.global_minD(ngausP,ngausD,dimP,dimG,bmax,masses,params)

    EP=dictD['E_list'][ngausP-1]
    ED=dictD['E_list'][ngausP+ngausD-1]
    massesN=dictD['masses']
    print(massesN)

    coords=dictD['coords']
    AP=[1/(bs**2) for bs in coords[-1][:ngausP]]
    alphasD=[]
    for j in range(ngausD):
        bj=coords[-1][ngausP+3*j:ngausP+3*j+3]
        AD=tr.A_generate(bj,w)
        alphasD.append(AD)
    K=met.K_gen(massesN[1],massesN[1],massesN[0])
    N,H=mat.PionTwo(AP,alphasD,params,massesN,w,K)
    c=dictD['eigenvectors'][-1]

    N1=N[0,0]
    N2=N[1:ngausP+1,1:ngausP+1]
    N3=N[1+ngausP:ngausD+ngausP+1,1+ngausP:ngausD+ngausP+1]

    c1=c[0,0]
    c2=c[1:ngausP+1]
    c3=c[1+ngausP:ngausD+ngausP+1]

    O1_list.append(c1*N1*c1)
    O2_list.append((c2.T@N2@c2)[-1])
    O3_list.append((c3.T@N3@c3)[-1])

    print(O1_list)
    print(np.abs(ED-EP))

    if EP==0 and ED==0:
        continue
    else:
        E_ref.append(np.abs(ED))
        E_comp.append(np.abs(ED-EP))
        b_list2.append(b)

with open('Paramstest_rangeParam.txt','w') as file:
    file.write('S_list: \n')
    for item in b_list:
        file.write(str(item) + '\n')

    file.write('---\n')

    file.write('S_list2: \n')
    for item in b_list2:
        file.write(str(item) + '\n')

    file.write('---\n')

    file.write('O1_list: \n')
    for item in O1_list:
        file.write(str(item)+'\n')
    
    file.write('---\n')

    file.write('O2_list: \n')
    for item in O2_list:
        file.write(str(item)+'\n')
    
    file.write('---\n')

    file.write('O3_list: \n')
    for item in O3_list:
        file.write(str(item)+'\n')
    
    file.write('---\n')

    file.write('E_comp: \n')
    for item in E_comp:
        file.write(str(item)+'\n')
    
    file.write('---\n')

    file.write('E_ref: \n')
    for item in E_ref:
        file.write(str(item)+'\n')

#SMALL_SIZE = 12
#MEDIUM_SIZE = 16
#BIGGER_SIZE = 18
#
#plt.rcParams['axes.titlesize']=MEDIUM_SIZE
#plt.rcParams['legend.fontsize']=SMALL_SIZE
#plt.rcParams['axes.labelsize']=MEDIUM_SIZE
#plt.rcParams['xtick.labelsize']=SMALL_SIZE
#plt.rcParams['ytick.labelsize']=SMALL_SIZE
#
#plt.figure(0)
#plt.title('Energy difference as a function of range parameter')
#plt.xlabel(r'$b_W$ [fm]')
#plt.ylabel(r'$|E_2-E_1| \quad [MeV]$')
#plt.plot(b_list2,E_comp,marker='o')
#plt.savefig('figures/EnergyCompb.pdf'.format(0),bbox_inches='tight')
#
#plt.figure(1)
#plt.title('Overlap as a function of range parameter')
#plt.xlabel(r'$b_W$ [fm]')
#plt.ylabel('Contribution two overlap')
#plt.plot(b_list,O1_list, marker='o', color='red', label='Bare proton')
#plt.plot(b_list,O2_list, marker='o', color='blue', label='One pion contribution')
#plt.plot(b_list,O3_list, marker='o',color='green', label='Two pion contribution')
#plt.legend()
#plt.savefig('figures/OverlapCompb.pdf'.format(0),bbox_inches='tight')
#
#plt.figure(3)
#plt.title('Two pion overlap as a function of range parameter')
#plt.xlabel(r'$b_W$ [fm]')
#plt.ylabel('Contribution to overlap')
#plt.plot(b_list,O3_list, marker='o',color='green', label='Two pion contribution')
#plt.legend()
#plt.savefig('figures/OverlapComp2Pionb.pdf'.format(0),bbox_inches='tight')
#
#
#plt.figure(2)
#plt.title('Energy comparison')
#plt.xlabel(r'$|E_2|$ [MeV]')
#plt.ylabel(r'$|E_2-E_1| [MeV]$')
#plt.plot(E_ref,E_comp,marker='o')
#plt.savefig('figures/EnergyCompAbsE2b.pdf'.format(0),bbox_inches='tight')
#plt.show()