#!/usr/bin/env python
# coding: utf-8

"""
fundamental_constants.py
    Defines some useful fundamental constants
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"



import numpy as np

#####################################################
#fundamental constants

#Boltzmann constant
kb = 8.617333262145*10**-5 # ev K^-1
kb_si = 1.380649*10**-23

#atomic mass unit
ua =  1.66053904*10**-27 #kg

#velocity of light
c = 299792458 # m/s

#electron mass
m_e = 9.10938356*10**-31

#plancks constant
h = 6.62607004 * 10**-34

#electron charge
e_c = 1.60217662 * 10**-19 #C

#epsilon_0
eps_0 = 8.8541878128* 10**-12 #F m^-1

#reference plasma temperature
T_ref =  1*11604.45 #in K

#reference electron density
n_e_ref = 1.*10**17 * (10**2)**3 #cm^-3 to m^-3

#####################################################

def plasma_reduction_energy_factor(n_e, T):
    return (e_c**2/(4*np.pi*eps_0))*np.sqrt(e_c**2*n_e/(eps_0*kb_si*T))*6.24150913*10**18



