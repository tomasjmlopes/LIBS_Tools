#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from LIBS_SPEC.voigt_fit import *
from LIBS_SPEC.fundamental_constants import *
import numpy as np
import matplotlib.pyplot as plt

def dlambda_doppler(lambda_0, T, m):
    
    return np.sqrt(2*kb_si*T/(m*c**2))*lambda_0

def spectrometer_broadening(ritz_l, wavelengths, resolution):
    
    fwhm_instrument = ritz_l/resolution
    #fwhm_instrument = 500/resolution
    sigma_instrument = fwhm_instrument/np.sqrt(8*np.log(2)) 
    
    value_instrument = 1./np.sqrt(2*np.pi)/sigma_instrument*np.exp (-(wavelengths-ritz_l)**2/(2*sigma_instrument**2))
    
    return value_instrument

def stark_broadening(ritz_l, wavelengths, params):
    
    stark = voigt_profile_binder(wavelengths, params[0], params[1], params[2],params[3])
    
    return stark


def line_shape(ritz_l,wl,dl, resolution = 3000, params_voigt = []):
    
    #value = (wl-ritz_l>=0)*(wl-(ritz_l+2*dl)<0)
    #value_doppler = 1./(sqrt(pi)*dlambda_doppler(ritz_l))*exp(-(wl-ritz_l)**2/dlambda_doppler(ritz_l)**2)
    #value_doppler1 = c/ritz_l * sqrt(m/(2*pi*kb_si*T)) * exp (-m/(2*kb_si*T)*c**2*(1-wl/ritz_l)**2)
    
    #spectrometer resolution broadening
    
    new_wavelengths = np.linspace(wl,np.roll(wl,-1),5)
    dh = np.diff(new_wavelengths,axis=0)
    
    value_instrument = spectrometer_broadening(ritz_l, new_wavelengths, resolution)
    
    broad=np.trapz(value_instrument, dx = dh, axis = 0)/(np.roll(wl,-1)-wl)
    broad[-1] = 0
    
   # broad1 = spectrometer_broadening(ritz_l, wl, resolution)
    
    if params_voigt!=[]:
        value_instrument = stark_broadening(ritz_l, wl, params_voigt)
        
    return broad#, broad1

"""

ee=200
x = arange(499-ee,501-ee,0.03)
r = 500.06-ee
r2= 499.2-ee
c,d = line_shape(r,x,0.1)
c1,d1 = line_shape(r2,x,0.1)
c+=c1
d+=d1
plt.plot(c,label='int')
plt.plot(d)
plt.legend(loc=1)

dh = np.diff(x,axis=0)
print(trapz(c,dx = dh))
print(trapz(d,dx=dh))

"""