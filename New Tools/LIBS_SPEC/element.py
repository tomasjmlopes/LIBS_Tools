#!/usr/bin/env python
# coding: utf-8
"""
element.py
    Implements the class element that contains information for each element,
    generates the synthetic spectrum and retrives the most relevant lines.
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"


import numpy as np
from matplotlib.pyplot import *
import pandas as pd
from LIBS_SPEC.fundamental_constants import *
from LIBS_SPEC.scrap_databases import *
from LIBS_SPEC.line_shapes import *
from LIBS_SPEC.partition_function import *
from LIBS_SPEC.line import *

class element:
    
    """
    A class that contains information for each element
    
    Attributes
    
        lines_NIST: list
            Contains the lines information gathered from NIST database
            
        lines_Kurucz: list
            Contains the lines information gathered from Kurucz database
        
        ion_energies: array
            Contains the ion energy for each ion state
            
        mass: float
            Atomic mass
        
    
    Methods
        
        synthetic_spectrum_NIST
        
        synthetic_spectrum_Kurucz
        
        get_most_relevant_lines_ion_state
        
    
    """
    
    
    def __init__(self, element, ratio = 1):
        """
        Given an element symbol creates an instance of class element to contain
        the information for that element.

        Parameters
        ----------
        element : string
            Chemical symbol as string
        ratio : float, optional
            Element concentration, the default is 1.

        """
        #read the information from the database of lines
        try:
            self.lines_NIST=read_lines_NIST(element)
        except:
            print('**** Warning - NIST data not loaded for '+element)
            self.lines_NIST=[]
        try:
            self.lines_Kurucz=read_lines_Kurucz(element)
        except:
            print('**** Warning - Kurucz data not loaded for '+element)
            self.lines_Kurucz=[]
        
        #read the information for mass and ion energies from the database of ion properties
        at_mass, ion_energ = read_ion_properties(element)
        
        self.ion_energies = ion_energ
        self.mass = at_mass*ua
        
        self.label = element
        
        self.ratio = ratio
            
    def __repr__(self):
        
        return " [ "+ self.label + " ratio " + str(self.ratio) + " ] "
    
            
    def __str__(self):
        
        return " [ "+ self.label + " ratio " + str(self.ratio) + " ] "


        
    def spectrum_NIST(self, max_ion_state=3,wl=[], lower_limit = 200., upper_limit =900., d_lambda = .1,
                           electron_density = n_e_ref, electron_temperature=T_ref,
                           params_voigt=[],resolution=3000,
                           normalize = False, Plot = False):
        """
       
        For a given temperature and electron density, the method
        returns a synthetic spectrum generated using data from the NIST database based on
        the Saha-Boltzmann equation.
        Uses a wl array with the wavelengths given or defines one from lower to upper
        limits with d_lambda. The spectral broadening is obtained from that contained in the 
        line_shapes.py 
        
        Parameters
        ----------
        max_ion_state : int, optional
            Maximum ionized state to be taken in consideration. The default is 3.
        wl : array, optional
            Optional array of wavelengths values for a specific spectrometer. The default is None.
        lower_limit : float, optional
            Lower limit for the wavelength. The default is 200..
        upper_limit : float, optional
            Upper limit for the wavelength. The default is 900..
        d_lambda : float, optional
            If wl is not provided, corresponds to the interval between consecutive wavelengths. The default is .1.
        electron_density : float, optional
            Electron density. The default is n_e_ref.
        electron_temperature : float, optional
            Electron temperature. The default is T_ref.
        normalize : Bool, optional
            If you want to normalize the spectrum. The default is False.
        plot : Bool, optional
            If you want to plot the computed spectrum. The default is False.

        Returns
        -------
        wavelength : array
            Array continaing the wavelengths array used for the simulation fo the spectrum.
        intensity : array of arrays
            Array containing each array corresponding to the simulated signal of each ionized state.
        labels : list of strings
            Labels for each array in intensity.
        n_ion: list of floats
            Relative concentration of each ionized state.

        """
        
        
        #if wavelegths not provided create an array
        if len(wl) == 0:
            wavelength = np.arange(lower_limit, upper_limit, d_lambda)
        else:
            wavelength = wl
        
        #to avoid errors
        if self.label == 'H':
            max_ion_state=1
        if self.label == 'He':
            max_ion_state=2
        if self.label == 'Li':
            max_ion_state=3
            
                    
        #initialize the signal
        intensity = np.zeros((max_ion_state,wavelength.shape[0]))
        
        #auxiliar values
        T=electron_temperature
        n_e=electron_density
        debroglie = np.sqrt(h**2/(2*np.pi*m_e*kb_si*electron_temperature))
        
        ####Partition function calculations to compute ion state concentrations####
        
        #Partition function for ion state 0
        Z_0 = partition_function(self.label,str(0),T/T_ref)
        factor_ion_list = [1]
        Zs=[Z_0]
        n_ion = []
        
        #for each ionized sate
        for i in range(2, max_ion_state+1):
            try:
                #compute the new partiion function
                Z_ion = partition_function(self.label,str(int(i)-1),T/T_ref)

                f = 1
                
                #multiply by previous values, saha eq. is a recurrent equation
                for j in range(0,len(factor_ion_list)):
                    f*=factor_ion_list[j]
                
                #ionized state energy - lacks the correction 
                E_ion=self.ion_energies[int(i)-2]
                
                #apply the saha formula
                f*=2*np.exp(-E_ion/(kb*T))*(debroglie**-3/n_e)*Z_ion/Zs[-1]
                
                #append to the lists
                factor_ion_list.append(f)
                Zs.append(Z_ion)
                
            
            except:
                print("Warning - Skipped a line at " +str(i)+" ion state, not found for element " + self.label)
            
        
        ff =0
        
        for i in range(0,len(factor_ion_list)):
            ff+=factor_ion_list[i]
        
        ns=1./ff
        
        for i in range(1,len(factor_ion_list)+1):
            n_ion.append(ns*factor_ion_list[i-1])

        

        for i in range(0,len(self.lines_NIST)):
            line_0 = self.lines_NIST[i]
            ion_num = line_0.ion_state
            if ion_num<max_ion_state+1:
                index = int(ion_num)

                #try:
                n_i = n_ion[index-1]
                lshape = line_shape(line_0.ritz,wavelength,d_lambda,resolution = resolution, params_voigt = params_voigt)
                intensity[index-1] += (1./Zs[index-1])*self.ratio*n_i*line_0.A_ji*line_0.g_j*lshape*np.exp(-line_0.e_upper/(kb*electron_temperature))
                 
                #except:
                #    print("Warning - Skipped " +str(i)+" line")
                    
        labels=[]
        for i in range(1,len(intensity)+1):
            labels_element = self.label + ' ' +str(i)
            labels.append(labels_element)
        
        #if you want to plot
        if Plot == True:
            figure(1)
            for i in range(0, len(labels)):
                fill_between(wavelength, intensity[i], label = labels[i] + " (" + str(n_ion[i]*100)[:5] +"%)", lw=2., ls='-')
            
            plot(wavelength, intensity.sum(axis = 0), label = "Sum", color = 'k', lw =1., ls=':')
            legend()
            
        return wavelength, intensity, labels, n_ion
    
    
    def get_most_relevant_lines_ion_state(self, n_lines = 10, ion_state=1, max_ion_state=3, lower_limit = 200., upper_limit =900., 
                                electron_density = n_e_ref, electron_temperature=T_ref):
        """
        
        For a given temperature and electron density, the method
        returns the n_lines(default 10) most relevant lines of ion_state, 
        within the defines limits.
        Obs: this method simulates the spectrum using NIST database 
        to obtain the information required, taking the predicted line intensity 
        from the Saha-Boltzmann equation without broadening.

        Parameters
        ----------
        n_lines : int, optional
            Number of lines to obtain. The default is 10.
        ion_state : int, optional
            Ion state you want to obtain the lines. The default is 1.
        max_ion_state : int, optional
            Maximum ionized state take into account in the simulation. The default is 3.
        lower_limit : float, optional
            Lower limit for the wavelength. The default is 200..
        upper_limit : float, optional
            Upper limit for the wavelength. The default is 900..
        electron_density : float, optional
            Electron density. The default is n_e_ref.
        electron_temperature : float, optional
            Electron temperature. The default is T_ref.

        Returns
        -------
        sorted_array: List of arrays
            List of 
        sorted_lines: list of lines
            Contains a number n_lines of elements of class line, 
            sorted from the most relevant one.       
        """

        lines_wavelength = []
        lines_intensity = []
        lines_list=[]
        
        T=electron_temperature
        n_e=n_e_ref
        debroglie = np.sqrt(h**2/(2*np.pi*m_e*kb_si*electron_temperature))
        
        #Partition function for ion state 0
        Z_0 = partition_function(self.label,str(0),T/T_ref)
        factor_ion_list = [1]
        Zs=[Z_0]
        n_ion = []
        
        #for each ionized sate
        for i in range(2, max_ion_state+1):
            try:
                #compute the new partiion function
                Z_ion = partition_function(self.label,str(int(i)-1),T/T_ref)

                f = 1
                
                #multiply by previous values, saha eq. is a recurrent equation
                for j in range(0,len(factor_ion_list)):
                    f*=factor_ion_list[j]
                
                #ionized state energy - lacks the correction 
                E_ion=self.ion_energies[int(i)-2]
                
                #apply the saha formula
                f*=2*np.exp(-E_ion/(kb*T))*(debroglie**-3/n_e)*Z_ion/Zs[-1]
                
                #append to the lists
                factor_ion_list.append(f)
                Zs.append(Z_ion)
                
            
            except:
                print("Warning - Skipped a line at " +str(i)+" ion state, not found for element " + self.label)
            
        
        ff =0
        
        for i in range(0,len(factor_ion_list)):
            ff+=factor_ion_list[i]
        
        ns=1./ff
        
        for i in range(1,len(factor_ion_list)+1):
            n_ion.append(ns*factor_ion_list[i-1])
        
        """
        Z_0 = partition_function(self.label,str(0),T/T_ref)
        factor_ion_list = [1]
        Zs=[Z_0]
        n_ion = []
        
        for i in range(2, max_ion_state+1):
            Z_ion = partition_function(self.label,str(int(i)-1),T/T_ref)
            
            f = 1
            for j in range(0,len(factor_ion_list)):
                f*=factor_ion_list[j]
            
            E_ion=self.ion_energies[int(i)-2]
            
            f*=2*np.exp(-E_ion/(kb*T))*(debroglie**-3/n_e)*Z_ion/Zs[-1]
            
            factor_ion_list.append(f)
            Zs.append(Z_ion)
            
        
        ff =0
        
        for i in range(0,len(factor_ion_list)):
            ff+=factor_ion_list[i]
        
        ns=1./ff
        
        for i in range(1,max_ion_state+1):
            n_ion.append(ns*factor_ion_list[i-1])
        """
        
        for i in range(0,len(self.lines_NIST)):
            line_0 = self.lines_NIST[i]
            if line_0.ritz>lower_limit and line_0.ritz<upper_limit:
                ion_num = line_0.ion_state
                if ion_num==ion_state:

                        index = int(ion_num)
                        n_i = n_ion[index-1]

                        intensity = (1./Zs[index-1])*self.ratio*n_i*line_0.A_ji*line_0.g_j*np.exp(-line_0.e_upper/(kb*electron_temperature))

                        lines_list.append(line_0)
                        lines_wavelength.append(line_0.ritz)
                        lines_intensity.append(intensity)

        lines_wavelength = np.array(lines_wavelength)
        lines_intensity = np.array(lines_intensity)
        lines_list = np.array(lines_list)
        
        sorted_array = np.array([np.array([xx,yy]) for yy,xx in sorted(zip(lines_intensity,lines_wavelength), key = lambda x: x[0],reverse=True)])
        sorted_lines = np.array([xx for yy,xx in sorted(zip(lines_intensity,lines_list), key = lambda x: x[0],reverse=True)])
        
        if n_lines < len(sorted_array):
            return sorted_array[:n_lines],sorted_lines[:n_lines]
        else:
            print( " Warning, asked for " + str(n_lines) + ' only got ' + str(len(sorted_array)) )
            return sorted_array, sorted_lines




    def generate_lines_database(self, ion_state=1, 
                                max_ion_state=3, lower_limit = 200., upper_limit =900., 
                                    electron_density = n_e_ref, electron_temperature=T_ref):
            
            """
            
            For a given temperature and electron density, the method
            returns the n_lines(default 10) most relevant lines of ion_state, 
            within the defines limits.
            Obs: this method simulates the spectrum using NIST database 
            to obtain the information required, taking the predicted line intensity 
            from the Saha-Boltzmann equation without broadening.
    
            Parameters
            ----------
            n_lines : int, optional
                Number of lines to obtain. The default is 10.
            ion_state : int, optional
                Ion state you want to obtain the lines. The default is 1.
            max_ion_state : int, optional
                Maximum ionized state take into account in the simulation. The default is 3.
            lower_limit : float, optional
                Lower limit for the wavelength. The default is 200..
            upper_limit : float, optional
                Upper limit for the wavelength. The default is 900..
            electron_density : float, optional
                Electron density. The default is n_e_ref.
            electron_temperature : float, optional
                Electron temperature. The default is T_ref.
    
            Returns
            -------
            sorted_array: List of arrays
                List of 
            sorted_lines: list of lines
                Contains a number n_lines of elements of class line, 
                sorted from the most relevant one.       
            """
    
            lines_wavelength = []
            lines_intensity = []
            lines_list=[]
            
            T=electron_temperature
            n_e=electron_density
            debroglie = np.sqrt(h**2/(2*np.pi*m_e*kb_si*electron_temperature))
            
            #Partition function for ion state 0
            Z_0 = partition_function(self.label,str(0),T/T_ref)
            factor_ion_list = [1]
            Zs=[Z_0]
            n_ion = []
            
            #for each ionized sate
            for i in range(2, max_ion_state+1):
                try:
                    #compute the new partiion function
                    Z_ion = partition_function(self.label,str(int(i)-1),T/T_ref)
    
                    f = 1
                    
                    #multiply by previous values, saha eq. is a recurrent equation
                    for j in range(0,len(factor_ion_list)):
                        f*=factor_ion_list[j]
                    
                    #ionized state energy - lacks the correction 
                    E_ion=self.ion_energies[int(i)-2]
                    
                    #apply the saha formula
                    f*=2*np.exp(-E_ion/(kb*T))*(debroglie**-3/n_e)*Z_ion/Zs[-1]
                    
                    #append to the lists
                    factor_ion_list.append(f)
                    Zs.append(Z_ion)
                    
                
                except:
                    print("Warning - Skipped a line at " +str(i)+" ion state, not found for element " + self.label)
                
            
            ff =0
            
            for i in range(0,len(factor_ion_list)):
                ff+=factor_ion_list[i]
            
            ns=1./ff
            
            for i in range(1,len(factor_ion_list)+1):
                n_ion.append(ns*factor_ion_list[i-1])
            

            for i in range(0,len(self.lines_NIST)):
                line_0 = self.lines_NIST[i]
                if line_0.ritz>lower_limit and line_0.ritz<upper_limit:
                    ion_num = line_0.ion_state
                    if ion_num==ion_state:
    
                            index = int(ion_num)
                            n_i = n_ion[index-1]
    
                            intensity = (1./Zs[index-1])*self.ratio*n_i*line_0.A_ji*line_0.g_j*np.exp(-line_0.e_upper/(kb*electron_temperature))
    
                            lines_list.append(line_0)
                            lines_wavelength.append(line_0.ritz)
                            lines_intensity.append(intensity)
                            
                            
    
            lines_wavelength = np.array(lines_wavelength)
            lines_intensity = np.array(lines_intensity)
            lines_list = np.array(lines_list)
            
            lista=[]
            try:
                maxim = max(lines_intensity)
            except:
                maxim = 1
            for i in range(0,len(lines_wavelength)):
                lista.append(self.label+";"+str(ion_state)+';'+str(lines_wavelength[i])+";"+str(lines_intensity[i]/maxim))
                #lista.append([self.label,str(lines_wavelength[i]),str(lines_intensity[i]/maxim)])
            
            lista=np.array(lista)
            #file1 = open(folder+"database_lines.txt","w")
            #f_handle = file(filename)
            #np.save(f_handle, lista)
            #f_handle.close()
            
            return lista
        
        

def get_database_lines(lower_limit = 250., upper_limit =900., 
                                    electron_density = n_e_ref, electron_temperature=T_ref):
    #import database with elements
    df = pd.read_csv('core/database/element_data/ion_data.csv',delimiter=';',index_col=False)

    #two lists, one with the element symbols and other with atomic number
    element_list=np.array(df['Element'])
    element_number =np.arange(1,len(np.array(df['Element']))+1)

    #set the ionized state for each element - max_ion_state defines a maximum
    max_ion_state = 2
    element_ions=[]
    for i in range(0,len(element_number)):
        element_ions.append(np.arange(0,min([element_number[i],max_ion_state])))
    
    lista_final=['Element;Ion;Line;Relative Intensity\n']
    for i1 in range(0,len(element_list)):
        element_l = str(element_list[i1])
        el = element(element_l)
        print(el)
        
        ll1 = el.generate_lines_database( ion_state=1, 
                                max_ion_state=3, lower_limit = lower_limit, upper_limit = upper_limit, 
                                    electron_density = electron_density, 
                                    electron_temperature=electron_temperature)
        
        ll2 = el.generate_lines_database( ion_state=2, 
                                max_ion_state=3, lower_limit = lower_limit, upper_limit = upper_limit, 
                                    electron_density = electron_density, 
                                    electron_temperature=electron_temperature)
        
        lista_final = np.concatenate([lista_final, ll1, ll2])
    
    np.savetxt("d_lines.txt",lista_final, fmt="%s")
    
    return lista_final
            
#al = element("Al")
#ll = al.generate_lines_database( ion_state=1, 
#                                max_ion_state=3, lower_limit = 200., upper_limit =900., 
#                                    electron_density = n_e_ref, electron_temperature=T_ref)
#ll = get_database_lines()
#Example
#al = element("Al")
#al.spectrum_NIST(d_lambda=0.01,Plot=True)
#print(al.get_most_relevant_lines_ion_state(ion_state=1))