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

class element_information:
    """
    A class that contains information for multiple elements
    
    Attributes
    ----------
    elements : list of str
        List of chemical symbols
    ratios : list of float
        List of element concentrations
    lines_NIST : list of lists
        Contains the lines information gathered from NIST database for each element
    lines_Kurucz : list of lists
        Contains the lines information gathered from Kurucz database for each element
    ion_energies : list of arrays
        Contains the ion energy for each ion state for each element
    masses : list of float
        Atomic masses for each element
    
    Methods
    -------
    spectrum_NIST
    spectrum_Kurucz
    get_most_relevant_lines_ion_state
    generate_lines_database
    """
    
    def __init__(self, elements, ratios=None):
        """
        Given a list of element symbols, creates an instance of class element to contain
        the information for those elements.

        Parameters
        ----------
        elements : list of str
            List of chemical symbols as strings
        ratios : list of float, optional
            List of element concentrations, defaults to 1 for each element if not provided.
        """
        self.elements = elements
        self.ratios = ratios if ratios is not None else [1] * len(elements)
        
        if len(self.elements) != len(self.ratios):
            raise ValueError("The number of elements and ratios must be the same.")
        
        self.lines_NIST = []
        self.lines_Kurucz = []
        self.ion_energies = []
        self.masses = []
        
        for element in self.elements:
            try:
                self.lines_NIST.append(read_lines_NIST(element))
            except:
                print(f'**** Warning - NIST data not loaded for {element}')
                self.lines_NIST.append([])
            
            try:
                self.lines_Kurucz.append(read_lines_Kurucz(element))
            except:
                print(f'**** Warning - Kurucz data not loaded for {element}')
                self.lines_Kurucz.append([])
            
            at_mass, ion_energ = read_ion_properties(element)
            self.ion_energies.append(ion_energ)
            self.masses.append(at_mass * ua)
    
    def __repr__(self):
        return " ".join([f"[{element} ratio {ratio}]" for element, ratio in zip(self.elements, self.ratios)])
    
    def __str__(self):
        return self.__repr__()

    def spectrum_NIST(self, max_ion_state=3, wl=[], lower_limit=200., upper_limit=900., d_lambda=.1,
                      electron_density=n_e_ref, electron_temperature=T_ref,
                      params_voigt=[], resolution=3000,
                      normalize=False, Plot=False):
        """
        For given temperature and electron density, returns a synthetic spectrum 
        generated using data from the NIST database based on the Saha-Boltzmann equation.
        
        Parameters
        ----------
        (Same as before, no changes)
        
        Returns
        -------
        wavelength : array
            Array containing the wavelengths used for the simulation of the spectrum.
        intensity : array of arrays
            Array containing each array corresponding to the simulated signal of each ionized state for each element.
        labels : list of strings
            Labels for each array in intensity.
        n_ion : list of lists of floats
            Relative concentration of each ionized state for each element.
        """
        if len(wl) == 0:
            wavelength = np.arange(lower_limit, upper_limit, d_lambda)
        else:
            wavelength = wl
        
        intensity = np.zeros((len(self.elements), max_ion_state, wavelength.shape[0]))
        
        T = electron_temperature
        n_e = electron_density
        debroglie = np.sqrt(h**2 / (2 * np.pi * m_e * kb_si * T))
        
        labels = []
        n_ion_all = []
        
        for elem_idx, (element, ratio) in enumerate(zip(self.elements, self.ratios)):
            Z_0 = partition_function(element, '0', T/T_ref)
            factor_ion_list = [1]
            Zs = [Z_0]
            n_ion = []
            
            for i in range(2, max_ion_state + 1):
                try:
                    Z_ion = partition_function(element, str(i-1), T/T_ref)
                    f = np.prod(factor_ion_list)
                    E_ion = self.ion_energies[elem_idx][i-2]
                    f *= 2 * np.exp(-E_ion / (kb * T)) * (debroglie**-3 / n_e) * Z_ion / Zs[-1]
                    factor_ion_list.append(f)
                    Zs.append(Z_ion)
                except:
                    print(f"Warning - Skipped a line at {i} ion state, not found for element {element}")
            
            ns = 1 / sum(factor_ion_list)
            n_ion = [ns * f for f in factor_ion_list]
            n_ion_all.append(n_ion)
            
            for line in self.lines_NIST[elem_idx]:
                ion_num = line.ion_state
                if ion_num < max_ion_state + 1:
                    index = int(ion_num)
                    n_i = n_ion[index-1]
                    lshape = line_shape(line.ritz, wavelength, d_lambda, resolution=resolution, params_voigt=params_voigt)
                    intensity[elem_idx, index-1] += (1/Zs[index-1]) * ratio * n_i * line.A_ji * line.g_j * lshape * np.exp(-line.e_upper / (kb * T))
            
            for i in range(1, max_ion_state + 1):
                labels.append(f'{element} {i}')
        
        if Plot:
            figure(1)
            for i, label in enumerate(labels):
                elem_idx, ion_idx = divmod(i, max_ion_state)
                fill_between(wavelength, intensity[elem_idx, ion_idx], 
                             label=f"{label} ({n_ion_all[elem_idx][ion_idx]*100:.2f}%)", 
                             lw=2., ls='-')
            
            plot(wavelength, intensity.sum(axis=(0,1)), label="Sum", color='k', lw=1., ls=':')
            legend()
        
        return wavelength, intensity, labels, n_ion_all

    def get_most_relevant_lines_ion_state(self, n_lines=10, ion_state=1, max_ion_state=3, 
                                          lower_limit=200., upper_limit=900., 
                                          electron_density=n_e_ref, electron_temperature=T_ref):
        """
        For given temperature and electron density, returns the n_lines most relevant lines 
        of ion_state for each element, within the defined limits.
        
        Parameters
        ----------
        (Same as before, no changes)
        
        Returns
        -------
        sorted_arrays : list of arrays
            List of arrays containing wavelength and intensity for each element
        sorted_lines : list of lists
            List of lists containing Line objects for each element, sorted by relevance
        """
        sorted_arrays = []
        sorted_lines = []
        
        for elem_idx, (element, ratio) in enumerate(zip(self.elements, self.ratios)):
            lines_wavelength = []
            lines_intensity = []
            lines_list = []
            
            T = electron_temperature
            n_e = electron_density
            debroglie = np.sqrt(h**2 / (2 * np.pi * m_e * kb_si * T))
            
            Z_0 = partition_function(element, '0', T/T_ref)
            factor_ion_list = [1]
            Zs = [Z_0]
            n_ion = []
            
            for i in range(2, max_ion_state + 1):
                try:
                    Z_ion = partition_function(element, str(i-1), T/T_ref)
                    f = np.prod(factor_ion_list)
                    E_ion = self.ion_energies[elem_idx][i-2]
                    f *= 2 * np.exp(-E_ion / (kb * T)) * (debroglie**-3 / n_e) * Z_ion / Zs[-1]
                    factor_ion_list.append(f)
                    Zs.append(Z_ion)
                except:
                    print(f"Warning - Skipped a line at {i} ion state, not found for element {element}")
            
            ns = 1 / sum(factor_ion_list)
            n_ion = [ns * f for f in factor_ion_list]
            
            for line in self.lines_NIST[elem_idx]:
                if lower_limit < line.ritz < upper_limit and line.ion_state == ion_state:
                    index = int(line.ion_state)
                    n_i = n_ion[index-1]
                    intensity = (1/Zs[index-1]) * ratio * n_i * line.A_ji * line.g_j * np.exp(-line.e_upper / (kb * T))
                    
                    lines_list.append(line)
                    lines_wavelength.append(line.ritz)
                    lines_intensity.append(intensity)
            
            lines_wavelength = np.array(lines_wavelength)
            lines_intensity = np.array(lines_intensity)
            lines_list = np.array(lines_list)
            
            sorted_indices = np.argsort(lines_intensity)[::-1]
            sorted_array = np.array([[lines_wavelength[i], lines_intensity[i]] for i in sorted_indices])
            sorted_lines_elem = lines_list[sorted_indices]
            
            if n_lines < len(sorted_array):
                sorted_arrays.append(sorted_array[:n_lines])
                sorted_lines.append(sorted_lines_elem[:n_lines])
            else:
                print(f"Warning, asked for {n_lines} but only got {len(sorted_array)} for element {element}")
                sorted_arrays.append(sorted_array)
                sorted_lines.append(sorted_lines_elem)
        
        return sorted_arrays, sorted_lines

    def generate_lines_database(self, max_ion_state=3, lower_limit=200., upper_limit=900., 
                                electron_density=n_e_ref, electron_temperature=T_ref):
        """
        Generates a database of spectral lines for all elements in the instance,
        considering multiple ionization states.
        
        Parameters
        ----------
        max_ion_state : int, optional
            Maximum ionization state to consider. Default is 3.
        lower_limit : float, optional
            Lower wavelength limit in nm. Default is 200.
        upper_limit : float, optional
            Upper wavelength limit in nm. Default is 900.
        electron_density : float, optional
            Electron density. Default is n_e_ref.
        electron_temperature : float, optional
            Electron temperature. Default is T_ref.
        
        Returns
        -------
        lista_final : list of str
            List of strings containing element, ion state, wavelength, and relative intensity for each line
        """
        lista_final = ['Element;Ion;Line;Relative Intensity\n']
        
        T = electron_temperature
        n_e = electron_density
        debroglie = np.sqrt(h**2 / (2 * np.pi * m_e * kb_si * T))
        
        for elem_idx, (element, ratio) in enumerate(zip(self.elements, self.ratios)):
            Z_0 = partition_function(element, '0', T/T_ref)
            factor_ion_list = [1]
            Zs = [Z_0]
            n_ion = []
            
            for i in range(2, max_ion_state + 1):
                try:
                    Z_ion = partition_function(element, str(i-1), T/T_ref)
                    f = np.prod(factor_ion_list)
                    E_ion = self.ion_energies[elem_idx][i-2]
                    f *= 2 * np.exp(-E_ion / (kb * T)) * (debroglie**-3 / n_e) * Z_ion / Zs[-1]
                    factor_ion_list.append(f)
                    Zs.append(Z_ion)
                except:
                    print(f"Warning - Skipped a line at {i} ion state, not found for element {element}")
            
            ns = 1 / sum(factor_ion_list)
            n_ion = [ns * f for f in factor_ion_list]
            
            for ion_state in range(1, max_ion_state + 1):
                lines_wavelength = []
                lines_intensity = []
                
                for line in self.lines_NIST[elem_idx]:
                    if lower_limit < line.ritz < upper_limit and line.ion_state == ion_state:
                        index = int(line.ion_state)
                        n_i = n_ion[index-1]
                        intensity = (1/Zs[index-1]) * ratio * n_i * line.A_ji * line.g_j * np.exp(-line.e_upper / (kb * T))
                        
                        lines_wavelength.append(line.ritz)
                        lines_intensity.append(intensity)
                
                lines_wavelength = np.array(lines_wavelength)
                lines_intensity = np.array(lines_intensity)
                
                max_intensity = np.max(lines_intensity) if len(lines_intensity) > 0 else 1
                
                for wavelength, intensity in zip(lines_wavelength, lines_intensity):
                    lista_final.append(f"{element};{ion_state};{wavelength};{intensity/max_intensity}")
        
        return lista_final

def get_database_lines(lower_limit=250., upper_limit=900., 
                       electron_density=n_e_ref, electron_temperature=T_ref):
    df = pd.read_csv('LIBS_SPEC/Databases/element_data/ion_data.csv', delimiter=';', index_col=False)
    element_list = np.array(df['Element'])
    
    max_ion_state = 2
    
    el = element(element_list)
    lista_final = ['Element;Ion;Line;Relative Intensity\n']
    
    for ion_state in [1, 2]:
        lines = el.generate_lines_database(
            ion_state=ion_state, 
            max_ion_state=max_ion_state,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            electron_density=electron_density,
            electron_temperature=electron_temperature
        )
        lista_final.extend(lines[1:])  # Skip the header
    
    np.savetxt("d_lines.txt", lista_final, fmt="%s")
    
    return lista_final