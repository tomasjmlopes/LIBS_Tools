from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from LIBS_SPEC.element_new import element_information

class EmissionToolkit:
    """
    A toolkit for handling and analyzing emission spectra data.

    This class provides methods for loading element data, identifying spectral lines,
    and classifying emission intensities.

    Attributes:
        element_list (List[str]): List of elements to consider in the analysis.
        lower_limit (float): Lower wavelength limit for spectral analysis.
        upper_limit (float): Upper wavelength limit for spectral analysis.
        element_manipulator (element_information): Object for handling element data.
        dbase (Optional[pd.DataFrame]): Processed database of spectral lines.
        id (Optional[Dict[str, List[Tuple[float, float, float, str]]]]): Identified elements and their spectral lines.
    """

    def __init__(self, 
                 element_list: List[str] = ["Co", "Si", "K", "O", "Fe", "Rb", "Li", "Mn", "Mg", "P", "Al", "Cu", "Pb", "Cr", "Ti", "C", "Na", "Ca", "Zn", "V"],
                 lower_limit: float = 200,
                 upper_limit: float = 900):
        """
        Initialize the EmissionToolkit.

        Args:
            element_list (List[str], optional): List of elements to consider. Defaults to a predefined list.
            lower_limit (float, optional): Lower wavelength limit. Defaults to 200 nm.
            upper_limit (float, optional): Upper wavelength limit. Defaults to 900 nm.
        """
        self.element_list = element_list
        self.element_manipulator = element_information(element_list)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.dbase: Optional[pd.DataFrame] = None
        self.id: Optional[Dict[str, List[Tuple[float, float, float, str]]]] = None

    @staticmethod
    def _classify_intensity(value: float) -> str:
        """
        Classify the intensity of a spectral line.

        Args:
            value (float): Relative intensity value.

        Returns:
            str: Classification of the intensity.
        """
        if 0.05 < value < 0.3:
            return "Low Intensity"
        elif 0.3 <= value < 0.5:
            return "Medium Intensity"
        elif 0.5 <= value <= 1:
            return "High Intensity"
        elif 0 <= value < 0.05:
            return "Ultra low Intensity"
        else:
            return "NA"

    def generate_database(self) -> pd.DataFrame:
        """
        Generate a database of spectral lines for the specified elements.

        Returns:
            pd.DataFrame: DataFrame containing spectral line information.
        """
        element_data = self.element_manipulator.generate_lines_database(max_ion_state=3, lower_limit=self.lower_limit, upper_limit=self.upper_limit)
        header = element_data[0].strip().split(';')
        data_rows = [row.split(';') for row in element_data[1:]]

        df = pd.DataFrame(data_rows, columns=header)
        df['Ion'] = df['Ion'].astype(int)
        df['Line'] = df['Line'].astype(float)
        df['Relative Intensity'] = df['Relative Intensity'].astype(float)
        return df

    def process_database(self, lower_limit: float = 250, upper_limit: float = 800, ion_num: int = 1, min_intensity: float = 0.01) -> np.ndarray:
        """
        Process the spectral line database with specified filters.

        Args:
            lower_limit (float, optional): Lower wavelength limit. Defaults to 250 nm.
            upper_limit (float, optional): Upper wavelength limit. Defaults to 800 nm.
            ion_num (int, optional): Ion state to filter. Defaults to 1.

        Returns:
            np.ndarray: Processed database as a numpy array.
        """
        df = self.generate_database()
        df['Line'] = df['Line'].round(2)
        df = df[(df['Relative Intensity'] > min_intensity) & 
                (df['Line'].between(lower_limit, upper_limit)) & 
                (df['Ion'] <= ion_num)]
        
        df['Relative Intensity'] = df['Relative Intensity'].round(3)
        df['Class'] = df['Relative Intensity'].apply(self._classify_intensity)
        df = df.sort_values('Relative Intensity', ascending=False)
        
        self.dbase = df
        return df.to_numpy()

    def identify_elements(self, x_features: List[float], wavelength_tolerance: float = 0.5, ion_num: int = 2, min_intensity: float = 0.01) -> Dict[str, List[Tuple[float, float, float, str]]]:
        """
        Identify elements based on the emission lines present in x_features.

        Args:
            x_features (List[float]): List of feature wavelengths to analyze.
            wavelength_tolerance (float, optional): Wavelength tolerance for matching. Defaults to 0.5 nm.

        Returns:
            Dict[str, List[Tuple[float, float, float, str]]]: Identified elements with their spectral lines.
        """
        emission_lines = self.process_database(min_intensity = min_intensity, ion_num = ion_num)
        identified_elements = {}

        for feature in x_features:
            matches = [
                (element, feature, wavelength, intensity, category)
                for element, _, wavelength, intensity, category in emission_lines
                if abs(feature - wavelength) <= wavelength_tolerance
            ]
            
            if matches:
                best_match = max(matches, key=lambda x: x[3])
                element, feature, wavelength, intensity, category = best_match
                if element not in identified_elements:
                    identified_elements[element] = []
                identified_elements[element].append((feature, wavelength, intensity, category))

        self.id = identified_elements
        return identified_elements

    def print_identified_elements(self) -> None:
        """
        Print the identified elements in a formatted way.
        """
        if self.id is None:
            print("No elements have been identified yet. Run 'identify_elements' first.")
            return

        print("Identified Elements:")
        for element, matches in self.id.items():
            print(f"{element}:")
            for feature, reference, intensity, category in matches:
                print(f"  - Feature at {feature:.2f} nm matches reference line at {reference:.2f} nm")
                print(f"    Relative Intensity: {intensity:.3f}, Category: {category}")