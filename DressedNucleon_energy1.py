import numpy as np
import matplotlib.pyplot as plt
import gauss_transformer as tr
import methods as met
import matrix_elements as mat
from scipy.optimize import minimize
from scipy.linalg import eigh, det
from svector_class import svector
 
params=np.zeros(2)
masses=np.zeros(2)
params[0]=3.9
params[1]=41.5
masses[0]=1524
masses[1]=134.98

ngaus=2
dim=1
bmax=1
 #Roughly the diameter of the proton in fm.

#hal=np.random.rand(100)
#bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.

#met.ParamMin(params,bs,masses,w=None)

E_dict=met.global_minP(ngaus,dim,bmax,-586,masses,params)

coords=E_dict['coords']
coordsLast=coords[-1]

alphas=[1/(b**2) for b in coordsLast]
print(coordsLast)
print(alphas)
print(E_dict['E_list'])

#E_dict_test=met.SVM_pseudo(ngaus,400,dim,bmax,Et,params)
#print('Test dictionary:', E_dict_test['E_list'])

met.plot_convergence(E_dict,0,'Convergence for Pion Photoproduction: Global Minimizer', 'Number of Gaussians', 'Energy [MeV]', ['Numerical values', 'Target value'], 'figures/PionPhoto_Global.pdf'.format(0))
#
#met.plot_convergence(E_dict_SVM, 1, 'Convergence for Pion Photoproduction: SVM','Number of Gaussians', 'Energy [MeV]', ['Numerical values', 'Target value'], 'figures/PionPhoto_SVM.pdf'.format(1))





