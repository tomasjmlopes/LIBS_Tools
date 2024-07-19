# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:53:12 2021

@author: nunoa
"""

from scipy import special
from scipy import optimize 
import numpy as np


def voigt_profile_binder(x,x0, A, sigma, gamma):
    return A*special.voigt_profile(x-x0,sigma, gamma)
    
    
def voigt_fit(x,y,initial_guess_x0 = 1,initial_guess_A=1):
    
    return optimize.curve_fit(voigt_profile_binder, x, y,p0=[initial_guess_x0,initial_guess_A,1,1])


def voigt_fwhm(x,params):
    y = voigt_profile_binder(x,params[0],params[1],params[2],params[3])
    d = y - (max(y) / 2) 
    indexes = np.where(d > 0)[0] 
    return abs(x[indexes[-1]] - x[indexes[0]]),[x[indexes[0]],x[indexes[-1]]],[y[indexes[0]],y[indexes[-1]]]

def lorentzian_profile_binder(x,x0, A, sigma, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return A* gamma / np.pi / ((x-x0)**2 + gamma**2)

def gaussian_profile_binder(x,x0, A, sigma, gamma):
    """ Return Gaussian line shape at x with HWHM sigma """
    return A*np.sqrt(np.log(2) / np.pi) / sigma* np.exp(-((x-x0) / sigma)**2 * np.log(2))


def binder_fit(x,y, binder, initial_guess_x0 = 1, initial_guess_A = 1):

    return optimize.curve_fit(binder, x, y,p0=[initial_guess_x0,initial_guess_A,1,1])

def fit_fwhm(x,params,binder):
    y = binder(x,params[0],params[1],params[2],params[3])
    d = y - (max(y) / 2) 
    indexes = np.where(d > 0)[0] 
    return abs(x[indexes[-1]] - x[indexes[0]]),[x[indexes[0]],x[indexes[-1]]],[y[indexes[0]],y[indexes[-1]]]


"""
from matplotlib import *
x=[-1,0,1,2,3,4,5,6]
y=[0,0,1,2,1,0,0,0]
xx=arange(-1,6,0.1)
p1,p2 = binder_fit(x,y,gaussian_profile_binder)

plot(x,y,'o')
plot(xx,gaussian_profile_binder(xx,p1[0], p1[1], p1[2],p1[3]))
"""