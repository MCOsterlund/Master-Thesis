import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import methods as met
import gauss_transformer as tr
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

##SOLUTION TO DIFFERENTIAL EQUATION IS FOUND AT: https://github.com/MartinMikkelsen

##-----------------------CONSTANTS----------------------------------------------------------------

b=3.9
S=41.5 #Coupling strength parameters
mp=938.27 #Mev
mn=939.57
mbare=(mp+mn)/2
mpi0=134.98 
mpiC=139.57#Mev
mpi=(mpi0+mpiC)/2
mu=mbare*mpi/(mbare+mpi) #Reduced mass
hc=197.3 #hbar*c

##---------------------FUNCTIONS FOR SOLVING THE DIFFERENTIAL EQUATION-----------------------------

#Function for calculating f(r)
def f(r):
    return S/b*np.exp(-r**2/(b**2))

#Function for setting up the system of differential equations.
def coupled(r,u,p):
    y,v,I=u
    E=p[0]
    dy=v
    dv=2*mu/(hc**2)*((mpi - E)*y + f(r))- 4/r*v
    dI=12*np.pi*f(r)*r**4 * y
    return dy,dv,dI

#Function for setting up the boundaries of the function.
def boundaries(ua,ub,p):
    ya,va,Ia=ua
    yb,vb,Ib=ub
    E=p[0]
    return va,vb+(2*mu*(mpi+np.abs(E)))**0.5*yb, Ia, Ib-E

rmax=5*b
rmin=0.01*b
start=np.log(rmin)
stop=np.log(rmax)
r=np.logspace(start,stop,num=3000,base=np.exp(1)) #Logarithmically spaced numbers
E=-2

u=[0*r,0*r,E*r/r[-1]] #Starting values

res1=solve_bvp(coupled,boundaries,r,u,p=[E],tol=1e-7,max_nodes=100000)
mbare=mbare-res1.p[0]
mu=mbare*mpi/(mbare+mpi)
res=solve_bvp(coupled,boundaries,r,u,p=[E],tol=1e-7,max_nodes=100000)
format_E=np.around(res.p[0],1)

##------------------------FUNCTIONS FOR SOLVING WITH GAUSSIAN METHOD-------------------------

ngaus=1
dim=1
bmax=b
bs1=[]
wave_tot=np.zeros((res.x.size,ngaus))
params=np.zeros(2)
masses=np.zeros(2)
params[0]=b
params[1]=S
masses[0]=(mp+mn)/2
masses[1]=mpi

dim=1
bmax=rmax

E_dict=met.global_minP(ngaus,dim,bmax,masses,params)
coords=E_dict['coords'] #Optimized coordinates.
format_EG=np.around(E_dict['E_list'][-1],1)
eigenvectors=E_dict['eigenvectors']
print(E_dict['masses'])
print("Energy from Differential:", format_E)
print("Energy from Gaussians:", format_EG)

wave_tot=np.zeros((len(res.x),len(coords)))

maxdiff=np.max(np.abs(res.y[0]))

for i in range(len(coords)):
    wavefunc=np.zeros(res.x.shape)
    wave_sum=np.zeros(res.x.shape)
    coords_actual=coords[i]
    c=eigenvectors[i]
    A=[1/(b**2) for b in coords_actual]
    for j in range(1,c.shape[0]):  #Start from index 1, since the first element is always hardcoded
        wave_sum+=c[j]*np.exp(-(A[j-1])*res.x**2)
    wavefunc+=wave_sum #I've multiplied by three in the matrix elements, so don't do it here.
    wave_tot[:,i]=wavefunc/c[0]

#print('Difference after:', np.abs(maxG-maxdiff))

SMALL_SIZE = 12
INTERMEDIATE_SIZE=14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=MEDIUM_SIZE
plt.rcParams['legend.fontsize']=INTERMEDIATE_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=SMALL_SIZE
plt.rcParams['ytick.labelsize']=SMALL_SIZE

##Pure wavefunction
plt.figure(0)
plt.xlabel('r [fm]')
plt.ylabel(r'$|r\phi(r)|^2 \quad [fm^{-5/2}]$')
plt.plot(res.x,res.y[0], label=r'$\phi(r)$')
plt.plot(res.x,res.yp[0], label=r'$\phi´(r)$')
plt.text(5,-0.009,'S=%s, b=%s' %(S,b))
plt.text(5,-0.012,'E=%s MeV' %(format_E))
plt.legend()
plt.savefig('figures/PionPhoto_Wavefunc_differential.pdf'.format(0),bbox_inches='tight')

##Gaussian comparison
fig, ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})
ax[0].set_xlabel('r [fm]')
ax[0].set_ylabel(r'$|r\phi(r)| \quad [fm^{-3/2}]$')
ax[0].plot(res.x,res.x*np.abs(res.y[0]), label=r'$|r\phi(r)|$')
ax[0].plot(res.x,res.x*np.abs(wave_tot[:,0]), '--', color="green",label='1 Gaussians')
ax[0].plot(res.x,res.x*np.abs(wave_tot[:,1]), '--', color="red",label='2 Gaussians')
#ax[0].plot(res.x,res.x*np.abs(wave_tot[:,2]), '--', color="blue",label='3 Gaussians')
#ax[0].plot(res.x,res.x*np.abs(wave_tot[:,3]), '--', color="red", label='4 Gaussians')
ax[0].text(7,0.01,'S=%s, b=%s' %(S,b))
ax[0].text(7,0.0087,'Differential method: E=%s MeV' %(format_E))
ax[0].text(7,0.0071,'Gaussian Method: E=%s MeV' %(format_EG))
ax[0].legend()

zoom_xlim=(2.3,3.4)
zoom_ylim=(0.0216,0.0224)

##---------------------ZOOMED SECTION---------------------------
ax[1].set_xlabel('r [fm]')
ax[1].set_ylabel(r'$|r\phi(r)| \quad [fm^{-3/2}]$')
ax[1].plot(res.x,res.x*np.abs(res.y[0]), label=r'$|r\phi(r)|$')
ax[1].plot(res.x,res.x*np.abs(wave_tot[:,0]), '--',color="green", label='1 Gaussians')
ax[1].plot(res.x,res.x*np.abs(wave_tot[:,1]), '--',color="red", label='2 Gaussians')
#ax[1].plot(res.x,res.x*np.abs(wave_tot[:,2]), '--',color="blue", label='3 Gaussians')
#ax[1].plot(res.x,res.x*np.abs(wave_tot[:,3]), '--',color="red", label='4 Gaussians')
ax[1].set_xlim(zoom_xlim)
ax[1].set_ylim(zoom_ylim)

plt.tight_layout()
plt.savefig('figures/Dressing1_Wavefunc.pdf'.format(0),bbox_inches='tight')
plt.show()

###-------------------------HIGH PARAMS SETTINGS------------------------

#plt.text(7,0.01,'S=%s, b=%s' %(S,b))
#plt.text(7,0.009,'Differential method: E=%s MeV' %(format_E))
#plt.text(7,0.008,'Gaussian Method: E=%s MeV' %(format_EG))

###-------------------------NORMAL PARAMS SETTINGS------------------------

#plt.text(2.5,0.0020,'S=%s, b=%s' %(S,b))
#plt.text(2.5, 0.0018,'Differential method: E=%s MeV' %(format_E))
#plt.text(2.5, 0.0016,'Gaussian Method: E=%s MeV' %(format_EG))




