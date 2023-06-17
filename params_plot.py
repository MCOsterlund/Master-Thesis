import matplotlib.pyplot as plt
import numpy as np

#Coupling strength

S_list=[0,5,10,20,30,40,50,60,70,80]
S_list2=[5,10,20,30,40,50,60,70,80]
O1_list=[1.0,
0.9371487659220021,
0.8368601424474366,
0.7114282229436201,
0.6507409039894173,
0.6165812270965665,
0.5948757365748083,
0.57991844245808,
0.5689996389119805,
0.560684954568055]
O2_list=[0.,
0.06130316,
0.15991697,
0.2851186,
0.34660401,
0.38145429,
0.40364949,
0.41895037,
0.43011306,
0.43860483]
O3_list=[0.,
0.00154807,
0.00322289,
0.00345318,
0.00265509,
0.00196448,
0.00147477,
0.00113119,
0.0008873,
0.00071022]
E_comp=[0.5807021751407042,
1.3531592534124997,
1.8115287976527839,
1.6807502831952377,
1.4585973250503912,
1.25734946274315,
1.0888454629421176,
0.9517283180192635,
0.8400361463684476]
E_ref=[13.632415226218729,
46.90981674248056,
137.05626867894517,
237.96727093949605,
342.6698113370739,
449.122815007011,
556.5158017974333,
664.4700188199479,
772.7834109685072
]


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=MEDIUM_SIZE
plt.rcParams['legend.fontsize']=SMALL_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=MEDIUM_SIZE
plt.rcParams['ytick.labelsize']=MEDIUM_SIZE


ticksX=[0,5,10,20,30,40,50,60,70,80,90]

E_rel=[]
b_Erel=[]

for i in range(len(E_comp)):
    A=E_comp[i]/E_ref[i]
    E_rel.append(A)

plt.figure(0)
plt.xlabel(r'$S_W$ [MeV]')
plt.xticks(np.arange(min(ticksX),max(ticksX),10))
plt.ylabel(r'$\Delta E_{rel} \quad [MeV]$')
plt.plot(S_list2,E_rel,marker='o')
plt.savefig('figures/EnergyComp.pdf'.format(0),bbox_inches='tight')

plt.figure(1)
plt.xlabel(r'$S_W$ [MeV]')
plt.ylabel('Contribution to norm')
plt.xticks(np.arange(min(ticksX),max(ticksX),10))
plt.plot(S_list,O1_list, marker='o', color='red', label='Bare proton')
plt.plot(S_list,O2_list, marker='o', color='blue', label='One pion contribution')
plt.plot(S_list,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapComp.pdf'.format(0),bbox_inches='tight')

plt.figure(3)
plt.xlabel(r'$S_W$ [MeV]')
plt.ylabel('Contribution to norm')
plt.xticks(np.arange(min(ticksX),max(ticksX),10))
plt.plot(S_list,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapComp2Pion.pdf'.format(0),bbox_inches='tight')

plt.figure(2)
#plt.title('Energy comparison')
plt.xlabel(r'$|E_2| \quad [MeV]$')
plt.ylabel(r'$|E_2-E_1| \quad [MeV]$')
plt.plot(E_ref,E_comp,marker='o')
plt.savefig('figures/EnergyCompAbsE2.pdf'.format(0),bbox_inches='tight')

#Range parameter

b_list2=[0.01,
0.5,
1,
1.5,
2,
2.5,
3,
3.5,
4,
4.5,
5]
O1_list=[1.0,
0.9999230282603473,
0.995320086928436,
0.9627881000300982,
0.8844365424370244,
0.7902021861660417,
0.7114282229436201,
0.6526830939955633,
0.608823992946414,
0.5746374768340006,
0.5467624159720526]
O2_list=[2.9177571e-18,
7.69717364e-05,
0.00467987,
0.03720022,
0.11537143,
0.20865351,
0.2851186,
0.33990281,
0.37821139,
0.40542943,
0.42522184]
O3_list=[9.0345719e-36,
3.26219801e-12,
4.35841376e-08,
1.16784752e-05,
0.00019203,
0.0011443,
0.00345318,
0.0074141,
0.01296462,
0.01993309,
0.02801574]
E_comp=[
7.314073680600072e-15,
5.203025343902823e-09,
1.970779897408903e-05,
0.007108034759628623,
0.10131959763562293,
0.583778112080708,
1.8115287976527839,
4.140833702339364,
7.860513086249966,
13.243969002892811,
20.49707664264031,
]
E_ref=[4.2238628475940325e-14,
0.09947569353314886,
2.4847301090886273,
13.982576693775812,
41.10303429259527,
83.28266874899339,
135.7909280568651,
195.0946948123509,
259.5593070601055,
328.57848298116187,
401.9796022748209]

E_rel=[]
b_Erel=[]

for i in range(len(E_comp)):
    if i==0:
        continue
    else:
        b_Erel.append(b_list2[i])
        A=E_comp[i]/E_ref[i]
        E_rel.append(A)

ticksX=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5]
ticksY=[0,0.005,0.01,0.015,0.02,0.025,0.03,0.035]

plt.figure(4)
#plt.title(r'Relative energy difference as a function of $b_W$, $S_W=20 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel(r'$\Delta E_{rel} \quad [MeV]$')
plt.xticks(np.arange(min(ticksX),max(ticksX),0.5))
plt.plot(b_Erel,E_rel,marker='o')
plt.savefig('figures/EnergyCompb.pdf'.format(0),bbox_inches='tight')

plt.figure(5)
#plt.title(r'Overlap as a function of $b_W$, $S_W=20 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel('Contribution to norm')
plt.plot(b_list2,O1_list, marker='o', color='red', label='Bare proton')
plt.plot(b_list2,O2_list, marker='o', color='blue', label='One pion contribution')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapCompb.pdf'.format(0),bbox_inches='tight')

plt.figure(6)
#plt.title(r'Two pion overlap as a function of $b_W$, $S_W=20 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel('Contribution to norm')
plt.yticks(np.arange(min(ticksY),max(ticksY),0.005))
plt.ylim([-0.001,0.03])
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapComp2Pionb.pdf'.format(0),bbox_inches='tight')

plt.figure(7)
#plt.title('Energy comparison')
plt.xlabel(r'$|E_2|$ [MeV]')
plt.ylabel(r'$|E_2-E_1| [MeV]$')
plt.plot(E_ref,E_comp,marker='o')
plt.savefig('figures/EnergyCompAbsE2b.pdf'.format(0),bbox_inches='tight')

#Extreme parameters

b_list2=[0.01,
0.5,
1,
1.5,
2,
2.5,
3,
3.5,
4,
4.5,
5,
6,
7,
8,
9,
10
]

O1_list=[1.0,
0.9999230282603473,
0.995320086928436,
0.9627881000300982,
0.8844365424370244,
0.7902021861660417,
0.7114282229436201,
0.6526830939955633,
0.608823992946414,
0.5746374768340006,
0.5478357301515293,
0.5047475783385325,
0.4707880444464195,
0.44215435765164113,
0.4170117642794069,
0.39436045851858337
]

O2_list=[2.9177571e-18,
7.69717364e-05,
0.00467987,
0.03720022,
0.11537143,
0.20865351,
0.2851186,
0.33990281,
0.37821139,
0.40542943,
0.42496382,
0.45080841,
0.46633414,
0.47628219,
0.48297207,
0.48763969
]

O3_list=[9.0345719e-36,
3.26219801e-12,
4.35841376e-08,
1.16784752e-05,
0.00019203,
0.0011443,
0.00345318,
0.0074141,
0.01296462,
0.01993309,
0.02720045,
0.04444401,
0.06287782,
0.08156346,
0.10001616,
0.11799985]
E_comp=[
7.314073680600072e-15,
5.203025343902823e-09,
1.970779897408903e-05,
0.007108034759628623,
0.10131959763562293,
0.583778112080708,
1.8115287976527839,
4.140833702339364,
7.860513086249966,
13.243969002892811,
20.070758647177513,
40.052390190429264,
68.88939489464042,
107.66671742640312,
157.4823144903803,
219.47568073598677
]
E_ref=[
4.2238628475940325e-14,
0.09947569353314886,
2.4847301090886273,
13.982576693775812,
41.10303429259527,
83.28266874899339,
135.7909280568651,
195.0946948123509,
259.5593070601055,
328.57848298116187,    
414.1709854877807,
584.4833925067879,
775.782871229584,
988.4731986311206,
1223.0589111986976,
1480.149044865742
]

ticksY=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]

plt.figure(8)
#plt.title(r'Energy difference as a function of $b_W$, $S_W=20MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel(r'$|E_2-E_1| \quad [MeV]$')
plt.plot(b_list2,E_comp,marker='o')
plt.savefig('figures/EnergyCompb_Extreme.pdf'.format(0),bbox_inches='tight')

plt.figure(9)
#plt.title(r'Overlap as a function of $b_W$, $S_W=20MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel('Contribution to norm')
plt.yticks(np.arange(min(ticksY),max(ticksY),0.1))
plt.plot(b_list2,O1_list, marker='o', color='red', label='Bare proton')
plt.plot(b_list2,O2_list, marker='o', color='blue', label='One pion contribution')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapCompb_Extreme.pdf'.format(0),bbox_inches='tight')

plt.figure(10)
#plt.title(r'Two pion overlap as a function of $b_W$, $S_W=20MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel('Contribution to norm')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapComp2Pionb_Extreme.pdf'.format(0),bbox_inches='tight')


plt.figure(11)
#plt.title('Energy comparison')
plt.xlabel(r'$|E_2|$ [MeV]')
plt.ylabel(r'$|E_2-E_1| [MeV]$')
plt.plot(E_ref,E_comp,marker='o')
plt.savefig('figures/EnergyCompAbsE2b_Extreme.pdf'.format(0),bbox_inches='tight')

##Range parameter, low S_W

b_list2=[0.01,
0.5,
1,
1.5,
2,
2.5,
3,
3.5,
4,
4.5,
5
]

O1_list=[1.0,
0.9999964286668739,
0.9997337269753349,
0.9974912211707222,
0.9894889100792217,
0.9710721546375144,
0.9375712444948093,
0.8832281908426485,
0.8000240070182107,
0.6820537502219712,
0.5312703851671033]

O2_list=[1.59935224e-20,
3.57133313e-06,
0.00026627,
0.00250796,
0.01048837,
0.02868229,
0.06092703,
0.11048584,
0.18003771,
0.26782402,
0.36450896]

O3_list=[
6.51795747e-39,
4.47825444e-16,
4.52500034e-09,
8.23437204e-07,
2.27222013e-05,
0.00024556,
0.00150173,
0.00628597,
0.01993828,
0.05012223,
0.10422065
]

E_comp=[
    1.6464315367573146e-14,
3.4073291063640987e-12,
4.295200497911322e-06,
0.0005094923690946995,
0.011113743933293563,
0.10357577214806746,
0.5694028310144326,
2.188786458873121,
6.502307904070122,
15.821173449649855,
32.61783167024834,
]

E_ref=[
    2.5378534896231397e-14,
0.005870488168175974,
0.1521598884088857,
0.8969320498886113,
2.9142945014943744,
6.913267275501353,
13.610908518490426,
23.914746269956595,
39.23363017884954,
61.606498700275765,
93.20595897523955,
]

ticksX=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6]

plt.figure(12)
#plt.title(r'Energy difference as a function of $b_W$, $S_W=5 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.xticks(np.arange(min(ticksX),max(ticksX),1))
plt.ylabel(r'$|E_2-E_1| \quad [MeV]$')
plt.plot(b_list2,E_comp,marker='o')
plt.savefig('figures/EnergyCompb_LowS.pdf'.format(0),bbox_inches='tight')

plt.figure(13)
#plt.title(r'Overlap as a function of $b_W$, $S_W=5 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.xticks(np.arange(min(ticksX),max(ticksX),1))
plt.ylabel('Contribution to norm')
plt.plot(b_list2,O1_list, marker='o', color='red', label='Bare proton')
plt.plot(b_list2,O2_list, marker='o', color='blue', label='One pion contribution')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapCompb_LowS.pdf'.format(0),bbox_inches='tight')

plt.figure(14)
#plt.title(r'Two pion overlap as a function of $b_W$, $S_W=5 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.xticks(np.arange(min(ticksX),max(ticksX),1))
plt.ylabel('Contribution to norm')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapComp2Pionb_LowS.pdf'.format(0),bbox_inches='tight')

plt.figure(15)
#plt.title('Energy comparison')
plt.xlabel(r'$|E_2|$ [MeV]')
plt.ylabel(r'$|E_2-E_1| [MeV]$')
plt.plot(E_ref,E_comp,marker='o')
plt.savefig('figures/EnergyCompAbsE2b_LowS.pdf'.format(0),bbox_inches='tight')

##Range parameter, large S_W

b_list2=[0.01,
0.5,
1,
1.5,
2,
2.5,
3,
3.5,
4,
4.5,
5]

O1_list=[
    0.9999999999999998,
0.9987745370873047,
0.9395618136834075,
0.769060106197178,
0.6503269301832257,
0.5913289355967302,
0.560684954568055,
0.5431664345503171,
0.5322125730932245,
0.5248202473550426,
0.5195077111033334
]

O2_list=[
    4.95667191e-17,
0.00122546,
0.06043707,
0.23090277,
0.34950223,
0.40826653,
0.43860483,
0.45576762,
0.46632792,
0.47329172,
0.47814491
]

O3_list=[ 
9.33664815e-35,
7.03232948e-14,
1.11669681e-06,
3.71236626e-05,
0.00017084,
0.00040453,
0.00071022,
0.00106595,
0.00145951,
0.00188803,
0.00234738
]

E_comp=[7.318652714021359e-15,
9.17786736387427e-07,
0.0010612261834879178,
0.028965363529550814,
0.14964973319570163,
0.40806305713977054,
0.8400361463684476,
1.4787521985216472,
2.3594981319195085,
3.522639385937282,
5.008956446685033]

E_ref=[
   6.960367263885563e-14,
1.5900659948173161,
37.45277062406363,
162.05774414615126,
344.6198693092541,
550.8516637779045,
772.7834109685072,
1009.3999749937434,
1260.5932992462208,
1526.187618788911,
1805.8814972108273 
]

plt.figure(16)
#plt.title(r'Energy difference as a function of $b_W$, $S_W=80 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel(r'$|E_2-E_1| \quad [MeV]$')
plt.plot(b_list2,E_comp,marker='o')
plt.savefig('figures/EnergyCompb_LargeS.pdf'.format(0),bbox_inches='tight')

plt.figure(17)
#plt.title(r'Overlap as a function of $b_W$, $S_W=80 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel('Contribution to norm')
plt.plot(b_list2,O1_list, marker='o', color='red', label='Bare proton')
plt.plot(b_list2,O2_list, marker='o', color='blue', label='One pion contribution')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapCompb_LargeS.pdf'.format(0),bbox_inches='tight')

plt.figure(18)
#plt.title(r'Two pion overlap as a function of $b_W$, $S_W=80 MeV$')
plt.xlabel(r'$b_W$ [fm]')
plt.ylabel('Contribution to norm')
plt.plot(b_list2,O3_list, marker='o',color='green', label='Two pion contribution')
plt.legend()
plt.savefig('figures/OverlapComp2Pionb_LargeS.pdf'.format(0),bbox_inches='tight')

plt.figure(19)
#plt.title('Energy comparison')
plt.xlabel(r'$|E_2|$ [MeV]')
plt.ylabel(r'$|E_2-E_1| [MeV]$')
plt.plot(E_ref,E_comp,marker='o')
plt.savefig('figures/EnergyCompAbsE2b_LargeS.pdf'.format(0),bbox_inches='tight')