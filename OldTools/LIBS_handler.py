import os
from typing import List, Union, Optional, Tuple
import numpy as np
import h5py
import yaml
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
import pandas as pd

class LIBS_Toolkit:
    """
    Python Toolkit designed to handle LIBS datasets.

    This class provides tools for loading data and performing various manipulations,
    including feature extraction (manual and automatic), baseline removal, and normalization.

    Attributes:
        fname (str): The filename of the LIBS dataset.
        config (dict): Configuration parameters.
        dataset (np.ndarray): The loaded LIBS dataset.
        wavelengths (np.ndarray): The wavelengths corresponding to the spectral dimension.
        positions (np.ndarray): The positions of each spectrum.
        x_size (int): The size of the x dimension.
        y_size (int): The size of the y dimension.
        spectral_size (int): The size of the spectral dimension.
        features (np.ndarray): Extracted features.
        x_features (List[float]): Wavelengths of extracted features.
    """

    def __init__(self, fname: str, config_file: Optional[str] = None, overwrite: bool = False):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"The file {fname} does not exist.")
        self.fname = fname
        self.config = self._load_config(config_file)
        self.dataset = None
        self.wavelengths = None
        self.positions = None
        self.x_size = None
        self.y_size = None
        self.spectral_size = None
        self.features = None
        self.x_features = None
        self.id_features = None
        self.classifier = None
        self.scaler = None
        self._overwrite = overwrite

    def _load_config(self, config_file: Optional[str]) -> dict:
        if config_file is None:
            return {'resolution': 0.5}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _classification_line(self, value):
        if value>0.05 and value<0.3:
            return "Low Intensity"
        elif value>=0.3 and value<0.5:
            return "Medium Intensity"
        elif value>=0.5 and value<=1:
            return "High Intensity"
        elif value>=0.0 and value<0.05:
            return "Ultra low Intensity"
        else:
            return "NA"

    def _element_lines(self, element, lower_limit = 250, upper_limit = 800, ion_num = 1):
        df = pd.read_csv("d_lines.txt", sep = ';')
        df['Line'] = [round(f, 2) for f in df['Line']]
        df.sort_values(by = ['Line','Relative Intensity'], ascending = [True, False], inplace = True)
        df1 = df[df['Relative Intensity'] > 0.01]
        df1 = df1[df1['Line'] > lower_limit]
        df1 = df1[df1['Line'] < upper_limit]
        df0 = df1.copy(deep = True)
        df0['Relative Intensity'] = [round(f, 3) for f in df1['Relative Intensity']]
        df0['Class'] = [self._classification_line(f) for f in df1['Relative Intensity']]
        df2 = df0[df0['Class'] == "High Intensity"]
        df3 = df0[(df0["Element"] == element)]
        df4 = df3.sort_values(by = ['Relative Intensity'], ascending = [False])
        df4 = df4[df4['Ion'] == ion_num]
        return df4.to_numpy()
    
    def get_dframe(self, emission_lines):
        return np.concatenate([self._element_lines(i) for i in emission_lines])

    def load_dataset(self, init_wv: Optional[int] = None, final_wv: Optional[int] = None, 
                     baseline_corrected: bool = True, return_pos: bool = False) -> None:
        """
        Load the entire dataset (spectrums and wavelengths).

        Args:
            init_wv (int, optional): Initial wavelength index. Defaults to None.
            final_wv (int, optional): Final wavelength index. Defaults to None.
            baseline_corrected (bool): Whether to load baseline-corrected data. Defaults to True.
            return_pos (bool): Whether to load spectrum positions. Defaults to False.

        Raises:
            IOError: If there's an error loading the dataset.
        """
        try:
            with h5py.File(self.fname, 'r') as hf:
                sample = list(hf.keys())[0].split(' ')[-1]
                baseline = 'Pro' if baseline_corrected else "raw_spectrum"

                spectrums = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/{baseline}']) for i in range(len(hf[f'Sample_ID: {sample}']))]
                positions = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/position']) for i in range(len(hf[f'Sample_ID: {sample}']))]

                self.wavelengths = np.array(hf['System properties']['wavelengths']).flatten()
                
                if init_wv is not None and final_wv is not None:
                    spectrums = [s[init_wv:final_wv] for s in spectrums]
                    self.wavelengths = self.wavelengths[init_wv:final_wv]

                self.x_size = len(np.unique([p[1] for p in positions]))
                self.y_size = len(np.unique([p[0] for p in positions]))
                self.spectral_size = len(self.wavelengths)

                # Sort spectrums and positions
                sorted_indices = np.lexsort(([p[0] for p in positions], [p[1] for p in positions]))
                spectrums = [spectrums[i] for i in sorted_indices]
                positions = [positions[i] for i in sorted_indices]

                self.dataset = np.array(spectrums).reshape(self.x_size, self.y_size, self.spectral_size)
                if return_pos:
                    self.positions = np.array(positions)

        except Exception as e:
            raise IOError(f"Error loading dataset: {str(e)}")

    def wavelength_to_index(self, WoI: float) -> int:
        """
        Find index closest to Wavelength of Interest "WoI"

        Args:
            WoI (float): Wavelength of interest

        Returns:
            int: Index of closest wavelength
        """
        return np.argmin(np.abs(self.wavelengths - WoI))

    def normalize_to_sum(self) -> np.ndarray:
        """
        Normalize each spectrum to its sum.

        Returns:
            np.ndarray: The normalized dataset.
        """
        normalized = self.dataset / np.sum(self.dataset, axis=2)[:,:,np.newaxis]
        if self._overwrite:
            self.dataset = normalized
        return normalized
    
    def perform_kmeans_clustering(self, n_clusters: int = 3, random_state = None):
        """
        Perform k-means clustering on the extracted features.

        Args:
            n_clusters (int): Number of clusters to form.

        Returns:
            np.ndarray: Labels of the clusters for each position.
        """
        if self.features is None:
            raise ValueError("Features not set. Please extract features before clustering.")

        reshaped_features = self.features.reshape(self.features.shape[0], -1).T
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(reshaped_features)

        if random_state:
            self.classifier = KMeans(n_clusters=n_clusters, random_state=random_state)
        else:
            self.classifier = KMeans(n_clusters=n_clusters)
        self.classifier.fit(scaled_features)

        # Reshape labels back to the original spatial dimensions
        labels = self.classifier.labels_.reshape(self.x_size, self.y_size)
        
        return labels
    
    def rocchio_classify(self, new_data):
        """
        Use pretrained K-Means to perform classification
        """
        return self.classifier.predict(new_data).reshape(self.y_size, self.x_size)

    def baseline_correct(self) -> np.ndarray:
        """
        Subtracts the baselines from the spectra.

        Returns:
            np.ndarray: Baseline-corrected dataset
        """
        flat_spectra = self.dataset.reshape(-1, self.spectral_size)
        baselines = self._get_baseline(flat_spectra)
        baselines = baselines[:, :self.spectral_size]  # Align baselines with spectra
        corrected_spectra = (flat_spectra - baselines).reshape(self.x_size, self.y_size, self.spectral_size)

        if self._overwrite:
            self.dataset = corrected_spectra
        return corrected_spectra

    def manual_features(self, list_of_wavelengths: List[float], sigma: Optional[float] = None) -> np.ndarray:
        """
        Extract the wavelengths provided in list of wavelengths

        Args:
            list_of_wavelengths (List[float]): List of wavelengths to extract
            sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

        Returns:
            np.ndarray: Extracted features
        """
        features = np.array([self.dataset[:, :, self.wavelength_to_index(wl)] for wl in list_of_wavelengths])
        if sigma is not None:
            features = np.array([gaussian_filter(f, sigma=sigma) for f in features])
        self.features = features
        self.x_features = list_of_wavelengths
        return features

    def automatic_feature_extraction(self, n_features: int = 20, smallest_dim_pixels: int = 5, 
                                     prominence: Union[float, str] = 'auto', sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically extract N features from dataset using the FT Feature Finder.

        Args:
            n_features (int): Number of features to extract
            smallest_dim_pixels (int): Smallest pixel dimension
            prominence (float or 'auto'): Prominence for peak finding. If 'auto', it's calculated from the data.
            sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and feature metric
        """
        freqs_x = 2*np.pi*np.fft.fftfreq(self.y_size, self.config['resolution'])
        freqs_y = 2*np.pi*np.fft.fftfreq(self.x_size, self.config['resolution'])

        fft_map = np.array([np.fft.fftshift(np.fft.fft2(self.dataset[:, :, i])) for i in range(self.spectral_size)])
        fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0  # Remove DC Component
    
        kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))
    
        object_size_small = smallest_dim_pixels * self.config['resolution']
        size_kspace_small = np.pi / object_size_small
    
        R = np.sqrt(kxx**2 + kyy**2)

        sum1 = np.sum(np.abs(fft_map[:, (R < size_kspace_small)]), axis=(1))
        max1 = np.sum(np.abs(fft_map), axis=(1, 2))
    
        sums = np.divide(sum1, max1, out=np.zeros(sum1.shape, dtype=float))
        sums = np.nan_to_num((sums - np.nanmin(sums))/(np.nanmax(sums) - np.nanmin(sums)), nan=0.0)

        if prominence == 'auto':
            prominence = np.mean(sums) + np.std(sums)

        inds, _ = find_peaks(sums, distance=5, prominence=prominence)
        self.x_features = self.wavelengths[inds]

        # Sort peaks in decreasing order and select first n features
        inds = inds[np.argsort(sums[inds])[::-1][:n_features]]

        # Extract Features
        self.x_features = self.wavelengths[inds]
        self.features = self.manual_features(self.x_features, sigma=sigma)
        return sums

    def _get_baseline(self, dataset: np.ndarray, min_window_size: int = 50, smooth_window_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate baseline using rolling window method.

        Args:
            dataset (np.ndarray): Input dataset
            min_window_size (int): Minimum window size for rolling minimum
            smooth_window_size (int, optional): Window size for smoothing

        Returns:
            np.ndarray: Calculated baselines
        """
        if smooth_window_size is None:
            smooth_window_size = 2 * min_window_size

        local_minima = self._rolling_min(
            arr = np.hstack(
                [dataset[:, 0][:, np.newaxis]] *
                ((min_window_size + smooth_window_size) // 2)
                + [dataset]
                + [dataset[:, -1][:, np.newaxis]] *
                ((min_window_size + smooth_window_size) // 2)
            ),
            window_width = min_window_size
        )
        return np.apply_along_axis(arr = local_minima, func1d = np.convolve, axis = 1,
                                   v = self._get_smoothing_kernel(smooth_window_size), mode = 'valid')

    @staticmethod
    def _rolling_min(arr: np.ndarray, window_width: int) -> np.ndarray:
        """
        Calculates the moving minima in each row of the provided array.

        Args:
            arr (np.ndarray): Input array
            window_width (int): Width of the rolling window

        Returns:
            np.ndarray: Array of rolling minimums
        """
        window = sliding_window_view(arr, (window_width,), axis = len(arr.shape) - 1)
        return np.amin(window, axis = len(arr.shape))

    @staticmethod
    def _get_smoothing_kernel(window_width: int) -> np.ndarray:
        """
        Generates a Gaussian smoothing kernel of the desired width.

        Args:
            window_width (int): Width of the smoothing window

        Returns:
            np.ndarray: Gaussian smoothing kernel
        """
        kernel = np.arange(-window_width//2, window_width//2 + 1, 1)
        sigma = window_width // 4
        kernel = np.exp(-(kernel ** 2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    

    def plot_spectrum(self, x: int, y: int) -> None:
        """
        Plot the spectrum at the given x, y coordinates.

        Args:
            x (int): x-coordinate
            y (int): y-coordinate
        """
        spectrum = self.dataset[x, y, :]
        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spectrum)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(f'Spectrum at position ({x}, {y})')
        plt.show()

    def spectrum_generator(self):
        """
        Generator for loading spectra one at a time.

        Yields:
            np.ndarray: Individual spectrum
        """
        with h5py.File(self.fname, 'r') as hf:
            sample = list(hf.keys())[0].split(' ')[-1]
            for i in range(self.x_size * self.y_size):
                yield np.array(hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/Pro'])

    def load_single_wavelength(self, wavelength: float, plot: bool = False) -> None:
        """
        Load a single wavelength from the dataset and display it as an image.

        This function efficiently loads only the requested wavelength data
        without loading the entire dataset into memory.

        Args:
            wavelength (float): The wavelength to load and display.

        Raises:
            ValueError: If the wavelength is not found in the dataset.
        """
        # Find the index of the closest wavelength
        wavelength_index = self.wavelength_to_index(wavelength)

        try:
            with h5py.File(self.fname, 'r') as hf:
                sample = list(hf.keys())[0].split(' ')[-1]
                
                image = np.zeros((self.y_size, self.x_size))

                sub_array_index = wavelength_index // 2048
                sub_wavelength_index = wavelength_index % 2048
                
                for i in range(self.y_size * self.x_size):
                    shot_i = hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/Pro'][sub_array_index][sub_wavelength_index]
                    y = i  % self.y_size
                    x = i // self.y_size
                    image[y, x] = shot_i
                image[:, ::2] = image[:, ::2][::-1]
                image = image.T

            if plot:
                plt.figure(figsize=(10, 8))
                plt.imshow(image, cmap='inferno')
                plt.colorbar(label='Intensity')
                plt.title(f'Image at wavelength {self.wavelengths[wavelength_index]:.2f} nm')
                plt.xlabel('X position')
                plt.ylabel('Y position')

        except Exception as e:
            raise IOError(f"Error loading wavelength data: {str(e)}")
        
        return image
    
    def basic_analysis(self, wavelength):
        fig, axs = plt.subplots(1, 2, figsize = (10, 4))
        ax = axs[0]
        ax.imshow(self.dataset[:, :, self.wavelength_to_index(wavelength)])
        ax.axis('off')

        ax = axs[1]
        ax.plot(self.wavelengths, self.dataset[self.x_size//2, self.y_size//2, :], 
                color = 'r', 
                label = 'LIBS Data')
        ax.legend()
        fig.tight_layout()

    def identify_elements(self, lines: List[List] = False, wavelength_tolerance: float = 0.5) -> Dict[str, List[Tuple[float, float, float, str]]]:
        """
        Identify elements based on the emission lines present in self.x_features.

        Args:
            emission_lines (List[List]): List of emission lines data.
            wavelength_tolerance (float): The wavelength tolerance in nm for matching emission lines.

        Returns:
            Dict[str, List[Tuple[float, float, float, str]]]: A dictionary where keys are element names and 
            values are lists of tuples. Each tuple contains the matched feature wavelength, 
            the corresponding reference wavelength, the relative intensity, and the intensity category.

        Raises:
            ValueError: If self.x_features is not set.
        """

        if lines:
            emission_lines = self.get_dframe(lines)
        else:
            emission_lines = self.get_dframe(["Co", "Si", "K", "O", "Fe", "Rb", "Li", "Mn", "Mg", "P", "Al", "Cu", "Pb", "Cr", "Ti", "C", "Na", "Ca", "Zn", "V"])

        if self.x_features is None:
            raise ValueError("No features have been extracted. Please run feature extraction first.")

        identified_elements = {}
        self.id_features = []

        for feature in self.x_features:
            matches_within_threshold = []
            for line in emission_lines:
                element, _, wavelength, intensity, intensity_category = line
                if abs(feature - wavelength) <= wavelength_tolerance:
                    matches_within_threshold.append((element, feature, wavelength, intensity, intensity_category))

            best_match = None
            if matches_within_threshold:
                best_match = max(matches_within_threshold, key=lambda x: x[3])  # x[3] is the intensity

            if best_match:
                element, feature, wavelength, intensity, intensity_category = best_match
                if element not in identified_elements:
                    identified_elements[element] = []
                identified_elements[element].append((feature, wavelength, intensity, intensity_category))
                self.id_features.append(element)
            else:
                self.id_features.append(None)  # No element identified for this feature

        return identified_elements


    def print_identified_elements(self, identified_elements: Dict[str, List[Tuple[float, float, float, str]]]) -> None:
        """
        Print the identified elements in a formatted way.

        Args:
            identified_elements (Dict[str, List[Tuple[float, float, float, str]]]): The dictionary of identified elements.
        """
        print("Identified Elements:")
        for element, matches in identified_elements.items():
            print(f"{element}:")
            for feature, reference, intensity, category in matches:
                print(f"  - Feature at {feature:.2f} nm matches reference line at {reference:.2f} nm")
                print(f"    Relative Intensity: {intensity:.3f}, Category: {category}")

    def calculate_element_probabilities(self, identified_elements: Dict[str, List[Tuple[float, float, float, str]]]) -> Dict[str, float]:
        """
        Calculate the probability of each element's presence based on the number and intensity of matched lines.

        Args:
            identified_elements (Dict[str, List[Tuple[float, float, float, str]]]): The dictionary of identified elements.

        Returns:
            Dict[str, float]: A dictionary where keys are element names and values are their calculated probabilities.
        """
        probabilities = {}
        total_intensity = sum(sum(intensity for _, _, intensity, _ in matches) for matches in identified_elements.values())

        for element, matches in identified_elements.items():
            element_intensity = sum(intensity for _, _, intensity, _ in matches)
            probabilities[element] = element_intensity / total_intensity

        return probabilities

    def print_element_probabilities(self, probabilities: Dict[str, float]) -> None:
        """
        Print the calculated probabilities of elements in a formatted way.

        Args:
            probabilities (Dict[str, float]): The dictionary of element probabilities.
        """
        print("Element Probabilities:")
        for element, probability in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"{element}: {probability:.2%}")

# import os
# from typing import List, Union, Optional, Tuple
# import numpy as np
# import h5py
# import yaml
# import dask.array as da
# from scipy.ndimage import gaussian_filter
# from scipy.signal import find_peaks
# from numpy.lib.stride_tricks import sliding_window_view
# import matplotlib.pyplot as plt

# class LIBS_Toolkit:
#     """
#     Python Toolkit designed to handle LIBS datasets.

#     This class provides tools for loading data and performing various manipulations,
#     including feature extraction (manual and automatic), baseline removal, and normalization.

#     Attributes:
#         fname (str): The filename of the LIBS dataset.
#         config (dict): Configuration parameters.
#         dataset (da.Array): The loaded LIBS dataset.
#         wavelengths (np.ndarray): The wavelengths corresponding to the spectral dimension.
#         positions (np.ndarray): The positions of each spectrum.
#         x_size (int): The size of the x dimension.
#         y_size (int): The size of the y dimension.
#         spectral_size (int): The size of the spectral dimension.
#         features (np.ndarray): Extracted features.
#         x_features (List[float]): Wavelengths of extracted features.
#     """

#     def __init__(self, fname: str, config_file: Optional[str] = None):
#         if not os.path.exists(fname):
#             raise FileNotFoundError(f"The file {fname} does not exist.")
#         self.fname = fname
#         self.config = self._load_config(config_file)
#         self.dataset = None
#         self.wavelengths = None
#         self.positions = None
#         self.x_size = None
#         self.y_size = None
#         self.spectral_size = None
#         self.features = None
#         self.x_features = None

#     def _load_config(self, config_file: Optional[str]) -> dict:
#         if config_file is None:
#             return {'resolution': 0.5}
#         with open(config_file, 'r') as f:
#             return yaml.safe_load(f)

#     def load_dataset(self, init_wv: Optional[int] = None, final_wv: Optional[int] = None, 
#                      baseline_corrected: bool = True, return_pos: bool = False) -> None:
#         """
#         Load the entire dataset (spectrums and wavelengths).

#         Args:
#             init_wv (int, optional): Initial wavelength index. Defaults to None.
#             final_wv (int, optional): Final wavelength index. Defaults to None.
#             baseline_corrected (bool): Whether to load baseline-corrected data. Defaults to True.
#             return_pos (bool): Whether to load spectrum positions. Defaults to False.

#         Raises:
#             IOError: If there's an error loading the dataset.
#         """
#         try:
#             with h5py.File(self.fname, 'r') as hf:
#                 sample = list(hf.keys())[0].split(' ')[-1]
#                 baseline = 'Pro' if baseline_corrected else "raw_spectrum"

#                 spectrums = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/{baseline}']) for i in range(len(hf[f'Sample_ID: {sample}']))]
#                 positions = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/position']) for i in range(len(hf[f'Sample_ID: {sample}']))]

#                 self.wavelengths = np.array(hf['System properties']['wavelengths']).flatten()
                
#                 if init_wv is not None and final_wv is not None:
#                     spectrums = [s[init_wv:final_wv] for s in spectrums]
#                     self.wavelengths = self.wavelengths[init_wv:final_wv]

#                 self.x_size = len(np.unique([p[1] for p in positions]))
#                 self.y_size = len(np.unique([p[0] for p in positions]))
#                 self.spectral_size = len(self.wavelengths)

#                 # Sort spectrums and positions
#                 sorted_indices = np.lexsort(([p[0] for p in positions], [p[1] for p in positions]))
#                 spectrums = [spectrums[i] for i in sorted_indices]
#                 positions = [positions[i] for i in sorted_indices]

#                 self.dataset = da.from_array(np.array(spectrums).reshape(self.x_size, self.y_size, self.spectral_size), chunks=('auto', -1, -1))
#                 if return_pos:
#                     self.positions = np.array(positions)

#         except Exception as e:
#             raise IOError(f"Error loading dataset: {str(e)}")

#     def wavelength_to_index(self, WoI: float) -> int:
#         """
#         Find index closest to Wavelength of Interest "WoI"

#         Args:
#             WoI (float): Wavelength of interest

#         Returns:
#             int: Index of closest wavelength
#         """
#         return np.argmin(np.abs(self.wavelengths - WoI))

#     def normalize_to_sum(self) -> da.Array:
#         """
#         Normalize each spectrum to its sum.

#         Returns:
#             da.Array: The normalized dataset.
#         """
#         return self.dataset / self.dataset.sum(axis=2)[:,:,np.newaxis]

#     def _get_baseline(self, dataset: da.Array, min_window_size: int = 50, smooth_window_size: Optional[int] = None) -> np.ndarray:
#         """
#         Calculate baseline using rolling window method.

#         Args:
#             dataset (da.Array): Input dataset
#             min_window_size (int): Minimum window size for rolling minimum
#             smooth_window_size (int, optional): Window size for smoothing

#         Returns:
#             np.ndarray: Calculated baselines
#         """
#         if smooth_window_size is None:
#             smooth_window_size = 2 * min_window_size

#         local_minima = self._rolling_min(
#             arr = np.hstack(
#                 [dataset[:, 0][:, np.newaxis]] *
#                 ((min_window_size + smooth_window_size) // 2)
#                 + [dataset]
#                 + [dataset[:, -1][:, np.newaxis]] *
#                 ((min_window_size + smooth_window_size) // 2)
#             ),
#             window_width = min_window_size
#         )
#         return np.apply_along_axis(arr = local_minima, func1d = np.convolve, axis = 1,
#                                    v = self._get_smoothing_kernel(smooth_window_size), mode = 'valid')

#     @staticmethod
#     def _rolling_min(arr: np.ndarray, window_width: int) -> np.ndarray:
#         """
#         Calculates the moving minima in each row of the provided array.

#         Args:
#             arr (np.ndarray): Input array
#             window_width (int): Width of the rolling window

#         Returns:
#             np.ndarray: Array of rolling minimums
#         """
#         window = sliding_window_view(arr, (window_width,), axis = len(arr.shape) - 1)
#         return np.amin(window, axis = len(arr.shape))

#     @staticmethod
#     def _get_smoothing_kernel(window_width: int) -> np.ndarray:
#         """
#         Generates a Gaussian smoothing kernel of the desired width.

#         Args:
#             window_width (int): Width of the smoothing window

#         Returns:
#             np.ndarray: Gaussian smoothing kernel
#         """
#         kernel = np.arange(-window_width//2, window_width//2 + 1, 1)
#         sigma = window_width // 4
#         kernel = np.exp(-(kernel ** 2) / (2 * sigma**2))
#         return kernel / kernel.sum()

#     def baseline_correct(self) -> da.Array:
#         """
#         Subtracts the baselines from the spectra.

#         Returns:
#             da.Array: Baseline-corrected dataset
#         """
#         flat_spectra = self.dataset.reshape(-1, self.spectral_size)
#         baselines = self._get_baseline(flat_spectra.compute())
#         baselines = baselines[:, :self.spectral_size]  # Align baselines with spectra
#         corrected_spectra = flat_spectra - baselines
#         return corrected_spectra.reshape(self.x_size, self.y_size, self.spectral_size)

#     def manual_features(self, list_of_wavelengths: List[float], sigma: Optional[float] = None) -> np.ndarray:
#         """
#         Extract the wavelengths provided in list of wavelengths

#         Args:
#             list_of_wavelengths (List[float]): List of wavelengths to extract
#             sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

#         Returns:
#             np.ndarray: Extracted features
#         """
#         features = np.array([self.dataset[:, :, self.wavelength_to_index(wl)].compute() for wl in list_of_wavelengths])
#         if sigma is not None:
#             features = np.array([gaussian_filter(f, sigma=sigma) for f in features])
#         self.features = features
#         self.x_features = list_of_wavelengths
#         return features

#     def automatic_feature_extraction(self, n_features: int = 20, smallest_dim_pixels: int = 5, 
#                                      prominence: Union[float, str] = 'auto', sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Automatically extract N features from dataset using the FT Feature Finder.

#         Args:
#             n_features (int): Number of features to extract
#             smallest_dim_pixels (int): Smallest pixel dimension
#             prominence (float or 'auto'): Prominence for peak finding. If 'auto', it's calculated from the data.
#             sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

#         Returns:
#             Tuple[np.ndarray, np.ndarray]: Extracted features and feature metric
#         """
#         freqs_x = 2*np.pi*np.fft.fftfreq(self.y_size, self.config['resolution'])
#         freqs_y = 2*np.pi*np.fft.fftfreq(self.x_size, self.config['resolution'])

#         fft_map = np.array([np.fft.fftshift(np.fft.fft2(self.dataset[:, :, i])) for i in range(self.spectral_size)])
#         fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0  # Remove DC Component
    
#         kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))
    
#         object_size_small = smallest_dim_pixels * self.config['resolution']
#         size_kspace_small = np.pi / object_size_small
    
#         R = np.sqrt(kxx**2 + kyy**2)

#         sum1 = np.sum(np.abs(fft_map[:, (R < size_kspace_small)]), axis=(1))
#         max1 = np.sum(np.abs(fft_map), axis=(1, 2))
    
#         sums = np.divide(sum1, max1, out=np.zeros(sum1.shape, dtype=float))
#         sums = np.nan_to_num((sums - np.nanmin(sums))/(np.nanmax(sums) - np.nanmin(sums)), nan=0.0)

#         if prominence == 'auto':
#             prominence = np.mean(sums) + np.std(sums)

#         inds, _ = find_peaks(sums, distance=5, prominence=prominence)
#         self.x_features = self.wavelengths[inds]

#         # Sort peaks in decreasing order and select first n features
#         inds = inds[np.argsort(sums[inds])[::-1][:n_features]]

#         # Extract Features
#         self.x_features = self.wavelengths[inds]
#         self.features = self.manual_features(self.x_features, sigma=sigma)
#         return sums

#     def plot_spectrum(self, x: int, y: int) -> None:
#         """
#         Plot the spectrum at the given x, y coordinates.

#         Args:
#             x (int): x-coordinate
#             y (int): y-coordinate
#         """
#         spectrum = self.dataset[x, y, :].compute()
#         plt.figure(figsize=(10, 5))
#         plt.plot(self.wavelengths, spectrum)
#         plt.xlabel('Wavelength (nm)')
#         plt.ylabel('Intensity')
#         plt.title(f'Spectrum at position ({x}, {y})')
#         plt.show()

#     def spectrum_generator(self):
#         """
#         Generator for loading spectra one at a time.

#         Yields:
#             np.ndarray: Individual spectrum
#         """
#         with h5py.File(self.fname, 'r') as hf:
#             sample = list(hf.keys())[0].split(' ')[-1]
#             for i in range(self.x_size * self.y_size):
#                 yield np.array(hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/Pro'])