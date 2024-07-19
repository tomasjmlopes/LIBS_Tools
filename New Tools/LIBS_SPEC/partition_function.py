#!/usr/bin/env python
# coding: utf-8
"""
partition_function.py
    Implements the necessary functions to compute the partition function values.
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"


import pandas as pd
from LIBS_SPEC.fundamental_constants import *
import numpy as np

def partition_function(element, ion_state,Tev):
    """

    Parameters
    ----------
    element : string
        Chemical symbol of the element.
    ion_state : int
        Ion state for which you want to compute the partition function.
    Tev : float
        Plasma temperature.

    Returns
    -------
    ZZ : float
        Result of the partition function

    """
    
    #read from data
    df = pd.read_csv('LIBS_SPEC/Databases/partition_data/' +element+" "+ ion_state+'.txt',delimiter=',',index_col=False)
        
    energies = df['LeveleV']
    
    #clean bad data
    energies = pd.to_numeric(energies, errors='coerce')
    energies.fillna(1e6,inplace=True)
    
    degeneracies= df['g']
    
    #clean bad data
    degeneracies = pd.to_numeric(degeneracies, errors='coerce')
    degeneracies.fillna(0,inplace=True)
    
    #compute the partition function
    ZZ=0
    for i in range(0,len(energies)):

        #print(str(i) + ' - '+str(degeneracies[i])+' - '+str(energies[i]))
        ZZ+=degeneracies[i]*np.exp(-energies[i]/(kb*Tev*T_ref))
        #print(ZZ)

    return ZZ




def test_partition_function():
    #import database with elements
    df = pd.read_csv('LIBS_SPEC/Databases/element_data/ion_data.csv',delimiter=';',index_col=False)

    #two lists, one with the element symbols and other with atomic number
    element_list=np.array(df['Element'])
    element_number = np.arange(1,len(np.array(df['Element']))+1)

    #set the ionized state for each element - max_ion_state defines a maximum
    max_ion_state = 4
    element_ions=[]
    for i in range(0,len(element_number)):
        element_ions.append(np.arange(0,min([element_number[i],max_ion_state])))
    
    for i1 in range(0,len(element_list)):
        element = str(element_list[i1])
        for j1 in range(0,len(element_ions[i1])):
            ion_state = str(int(element_ions[i1][j1]))
            current_file = open('LIBS_SPEC/Databases/partition_data/' +element+" "+ ion_state+'.txt','r')
            Z=partition_function(element,ion_state, 1)
            reference = float(current_file.readlines()[-1].split("#")[2])
            if (abs(Z-reference)>0.01):
                print("Warning -- Failed test at "+element+ " "+ ion_state + ' by ' + str(abs(Z-reference)))
                print("Z = "+str(Z)+ " ref = "+str(reference))
            current_file.close()
            
            
#partition_function('Ca','0',1.0)
#test_partition_function()


