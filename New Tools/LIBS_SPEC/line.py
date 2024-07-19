#!/usr/bin/env python
# coding: utf-8

"""
line.py
    Implements a class to store information for a given line.
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"



import re

class line:
    
    """
    
    A class to store information for a given line
    
    Attributes:
        ritz: float
            Value of Ritz wavelength
        e_upper: float
            Value of the energy of the upper level
        e_lower: float
            Value of the energye of the lower level
        A_ji: float
            Transition probability associated to the line
        g_j: float
            Degeneracy of the upper level
        ion_state: float
            Ionized state
        
    
    """
    
    def __init__(self, ritz, e_upper, e_lower, A_ji, g_j, ion_state,label):
        
        """
        creates a line object with the necessary attributes
        """
        
        #obs: we use re.sub() to clean the data (some of the data contains numbers and letters which are ) 
        self.ritz = float(re.sub('[^0-9.]','', str(ritz)))            
        self.e_upper = float(re.sub('[^0-9.]','', str(e_upper)))
        self.e_lower = float(re.sub('[^0-9.]','', str(e_lower)))
        self.A_ji = float(re.sub('[^0-9.]','', str(A_ji)))
        self.g_j = float(re.sub('[^0-9.]','', str(g_j)))
        self.ion_state = float(re.sub('[^0-9.]','', str(ion_state)))
        self.label = str(label)
        
    def __str__(self):
        """
        Returns a string representation
        """
        return ('Line of Wavelength '+ str(self.ritz)+ str('\n'))
    
    def __repr__(self):
        """
        Returns the shell representation
        """
        return ('Wavelength - ' + str(self.ritz) + ' - ion '+ str(self.ion_state)+ str('\n'))

