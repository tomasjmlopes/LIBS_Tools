#!/usr/bin/env python
# coding: utf-8

"""
scrap_databases.py
    Implements the necessary functions to read data from databases
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"



import pandas as pd
from LIBS_SPEC.line import *
import numpy as np

def read_lines_NIST(element):
    """
    Reads the data from the NIST database for a given element
    """
        
    #gather data
    #read data from database
    filename = 'LIBS_SPEC/Databases/libs_data_NIST/'+element+'.txt'
    lines_df = pd.read_csv(filename,sep='\t')
    
    #H does not contain ion_state info
    if element == 'H': 
        ion_state=np.ones(lines_df['ritz_wl_air(nm)'].shape)
    else:
        ion_state = np.array(lines_df['sp_num'])
    
    center = np.array(lines_df['obs_wl_air(nm)'])
    ritz = np.array(lines_df['ritz_wl_air(nm)'])
    e_upper = np.array(lines_df['Ek(eV)'])
    e_lower = np.array(lines_df['Ei(eV)'])
    A_ji = np.array(lines_df['Aki(s^-1)'])
    g_j = np.array(lines_df['g_k'])
    rel_intens = np.array(lines_df['intens'])
            
        
    list_lines=[]
    
    for i in range(0,len(center)):
        #try:
            list_lines.append(line(ritz[i], e_upper[i], e_lower[i], A_ji[i], g_j[i], ion_state[i], element))
        #except:
         #   None
    return list_lines

def read_lines_Kurucz(element):
    """
    Reads the data from the Kurucz database for a given element
    """
    
    #gather data
    #read data from database
    filename = 'LIBS_SPEC/Databases/libs_data_Kurucz/by_element/'+element+'.txt'
    lines_df = pd.read_csv(filename,sep=',')
    
    ion_state = np.array(lines_df['ion state num'])
    center = np.array(lines_df['Wl/nm'])
    ritz = np.array(lines_df['Wl/nm'])
    e_upper = np.array(lines_df['E_upper_lev.'])
    e_lower = np.array(lines_df['E_lower_lev.'])
    A_ji = np.array(lines_df['A-Value'])
    g_j = np.array(lines_df['g_j'])
    rel_intens = 0
            
        
    list_lines=[]
    
    for i in range(0,len(center)):
        list_lines.append(line(ritz[i], e_upper[i], e_lower[i], A_ji[i], g_j[i], ion_state[i], element))
    
    return list_lines

def read_ion_properties(element_label):
    
    """
    Reads the ion properties data from the NIST database for a given element
    """
    
    #gather data
    #read data from database
    filename = 'LIBS_SPEC/Databases/element_data/ion_Data.csv'
    df = pd.read_csv(filename,delimiter = ';')
    at_mass = float(df[df['Element']==element_label]['Mass'].iloc[0])
    element_row = df.loc[df['Element']==element_label]
    ion_en = element_row.to_numpy()[0][2:]
    
    return at_mass, ion_en

