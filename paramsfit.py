import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

S_list=[0,5,10,20,30,40,50,60,70,80]
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

def fitfunc(x,A,B):
    return A*x*np.exp(-B*x**2)

popt, pcov=curve_fit(fitfunc,S_list,O3_list)
print(popt)

S_fit=np.linspace(0,80,200)

plt.plot(S_list,O3_list, '.')
plt.plot(S_fit,fitfunc(S_fit,popt[0],popt[1]))
plt.show()