import os
from typing import List, Union, Optional, Tuple
from collections import defaultdict
import numpy as np
import h5py
import yaml
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
import pandas as pd
from LIBS_SPEC.line_handler import EmissionToolkit

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
        self._overwrite = overwrite
        self.config = self._load_config(config_file)
        self.emission_tkit = EmissionToolkit()
        self.dataset = None
        self.wavelengths = None
        self.positions = None
        self.x_size = None
        self.y_size = None
        self.spectral_size = None
        self.features = None
        self.x_features = None
        self.ids_features = None
        self.classifier = None
        self.scaler = None
        self.debug = None

    def _load_config(self, config_file: Optional[str]) -> dict:
        if config_file is None:
            return {'resolution': 0.5}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def update_elements(self, element_list: List[str]) -> None:
        self.emission_tkit = EmissionToolkit(element_list)

    def load_dataset(self, init_wv: Optional[int] = None, final_wv: Optional[int] = None, 
                     baseline_corrected: bool = True, return_pos: bool = False) -> None:
        """
        Loads the dataset from the file, optionally focusing on a specified wavelength range.

        Args:
            init_wv (int, optional): Initial wavelength index.
            final_wv (int, optional): Final wavelength index.
            baseline_corrected (bool): If True, applies baseline correction.
            return_pos (bool): If True, loads and stores spatial positions of spectra.
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
    
    def perform_kmeans_clustering(self, n_clusters: int = 3, random_state = None, get_features = False):
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
        if get_features:
            return labels, scaled_features.reshape(self.x_size, self.y_size, -1)
        else:
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


    def get_cluster_colors(self, n_clusters: int):
        """
        Generate a color map for the clusters.

        Args:
            n_clusters (int): Number of clusters.

        Returns:
            ListedColormap: Color map for the clusters.
        """
        colors = plt.cm.get_cmap('inferno')(np.linspace(0, 1, n_clusters))
        return ListedColormap(colors)
    
    def analyze_clusters(self, n_clusters: int = 3, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        if random_state:
            labels, int_features = self.perform_kmeans_clustering(n_clusters, random_state = random_state, get_features = True)
        else:
            labels, int_features = self.perform_kmeans_clustering(n_clusters, get_features = True)
        self.debug = int_features
        
        # Reshape int_features back to (x_size, y_size, spectral_size)

        # Calculate cluster mean intensities using the reshaped features
        cluster_mean_intensities = []
        for i in range(n_clusters):
            cluster_mask = (labels == i)
            cluster_spectra = int_features[cluster_mask]
            mean_intensities = np.mean(cluster_spectra, axis=0)
            cluster_mean_intensities.append(mean_intensities)

        cluster_colors = self.get_cluster_colors(n_clusters)
        
        # Use the updated plot_clustering_results method
        self.plot_clustering_results(labels, cluster_colors, cluster_mean_intensities)

        return labels, cluster_mean_intensities

    def plot_clustering_results(self, labels: np.ndarray, cluster_colors: ListedColormap, cluster_mean_intensities: List[np.ndarray]):
        """
        Plot the clustering results alongside average element intensities.

        Args:
            labels (np.ndarray): Cluster labels for each spectrum.
            cluster_colors (ListedColormap): Color map for the clusters.
            cluster_mean_intensities (List[np.ndarray]): Mean intensities for each cluster.
        """
        n_clusters = len(np.unique(labels))

        # Create a figure with subplots
        fig = plt.figure(figsize = (14, 8))
        gs = fig.add_gridspec(n_clusters + 1, 2)

        # Plot clustering results
        ax_cluster = fig.add_subplot(gs[:, 0])
        im = ax_cluster.imshow(labels.reshape(self.x_size, self.y_size), cmap=cluster_colors, interpolation='nearest')
        plt.colorbar(im, ax=ax_cluster, label='Cluster', ticks=range(n_clusters), fraction=0.046, pad=0.04)
        ax_cluster.set_title('Clustering Results')
        ax_cluster.set_xlabel('X position')
        ax_cluster.set_ylabel('Y position')

        # Plot average element intensities for each cluster
        for i in range(n_clusters):
            ax = fig.add_subplot(gs[i, 1])
            self._plot_average_element_intensities_single_cluster(cluster_mean_intensities[i], cluster_colors(i/n_clusters), ax, i)

        fig.tight_layout()

    def _plot_average_element_intensities_single_cluster(self, cluster_mean_intensities: np.ndarray, 
                                                         color: tuple, ax: plt.Axes, cluster_id: int):
        """
        Plot average intensities of elements for a single cluster.

        Args:
            cluster_mean_intensities (np.ndarray): Mean intensities for the cluster.
            color (tuple): Color for the cluster.
            ax (plt.Axes): The axes to plot on.
            cluster_id (int): The ID of the cluster.
        """
        if self.ids_features is None:
            raise ValueError("Element identification has not been performed. Call id_features() first.")

        element_intensities = defaultdict(list)
        for i, wavelength in enumerate(self.x_features):
            for element, lines in self.ids_features.items():
                if any(abs(wavelength - line[1]) < 0.2 for line in lines):
                    element_intensities[element].append(cluster_mean_intensities[i])
                    break
            else:
                element_intensities['Unknown'].append(cluster_mean_intensities[i])

        element_avg_intensities = {element: np.mean(intensities) for element, intensities in element_intensities.items()}

        elements = list(element_avg_intensities.keys())
        intensities = [element_avg_intensities[element] for element in elements]

        ax.bar(elements, intensities, color=color, width = 0.4)
        ax.set_ylabel('Average Intensity')
        ax.set_title(f'Cluster {cluster_id + 1}')
        ax.set_xticks(np.arange(len(elements)))
        ax.set_xticklabels(elements, rotation=45, ha='right')
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim(0, 1)

    def fft_features(self, n_features: int = 20, smallest_dim_pixels: int = 5, 
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

        fft_wavelengths = self.wavelengths[inds]
        fft_wavelengths = fft_wavelengths[np.argsort(sums[inds])[::-1][:n_features]]
        return fft_wavelengths

    def intensity_features(self, n_features: int = 20, prominence: Union[float, str] = 'auto'):
        mean_spec = self.calculate_average_spectrum()
        if prominence == 'auto':
            prominence = np.mean(mean_spec) + np.std(mean_spec)
        inds, _ = find_peaks(mean_spec, distance = 5, prominence = prominence)

        mean_intensities = mean_spec[inds]
        mean_features = self.wavelengths[inds]
        mean_features = mean_features[np.argsort(mean_intensities)[::-1][:n_features]]
        return mean_features

    def automatic_feature_extraction(self, fft_features: int = 20, fft_prominence: float = 'auto', min_pixel_dist: float = 5, 
                                           int_features: int = 20, int_prominence: float = 'auto', sigma: float = 0.5, perform_fft = True):

        if perform_fft:
            fft_features = self.fft_features(n_features = fft_features, smallest_dim_pixels = min_pixel_dist, prominence = fft_prominence)
        else:
            fft_features = np.array([])
        intensity_features = self.intensity_features(n_features = int_features, prominence = int_prominence)

        tolerance = 0.25
        result_array = fft_features.copy()
        for value in intensity_features:
            if np.all(np.abs(result_array - value) > tolerance):
                result_array = np.append(result_array, value)

        self.x_features = result_array
        self.features = self.manual_features(self.x_features, sigma = sigma)


    def id_features(self, wavelength_tolerance: float = 0.2, ion_num: int = 2, min_intensity: float = 0.01):
        self.ids_features = self.emission_tkit.identify_elements(self.x_features, wavelength_tolerance = wavelength_tolerance, ion_num = ion_num, min_intensity = min_intensity)

    def detailed_ids(self):
        return self.emission_tkit.print_identified_elements()

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

    def calculate_average_spectrum(self) -> np.ndarray:
        """
        Calculate the average spectrum of the whole dataset.

        Returns:
            np.ndarray: The average spectrum
        """
        return np.mean(self.dataset, axis = (0, 1))

    def estimate_element_probabilities(self) -> Dict[str, float]:
        """
        Estimate the probability of each identified element being present.

        Returns:
            Dict[str, float]: A dictionary with element symbols as keys and estimated probabilities as values
        """
        if self.ids_features is None:
            raise ValueError("Element identification has not been performed. Call id_features() first.")

        avg_spectrum = self.calculate_average_spectrum()
        total_intensity = 0
        element_intensities = {}

        for element, lines in self.ids_features.items():
            element_intensity = 0
            for line in lines:
                wavelength, _, _, _ = line
                index = self.wavelength_to_index(wavelength)
                element_intensity += avg_spectrum[index]
            element_intensities[element] = element_intensity
            total_intensity += element_intensity

        probabilities = {element: intensity / total_intensity 
                         for element, intensity in element_intensities.items()}

        return probabilities

    def print_element_probabilities(self):
        """
        Print the estimated probabilities of identified elements.
        """
        probabilities = self.estimate_element_probabilities()
        print("Estimated Element Probabilities:")
        for element, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"{element}: {prob*100:.2f}%")

    def standard_analysis(self, radius = 3, cmap = 'inferno'):
        mean_signal = np.mean(self.dataset, axis = (0, 1))
        min_signal = np.min(self.dataset, axis = (0, 1))
        max_signal = np.max(self.dataset, axis = (0, 1))


        fig = plt.figure(tight_layout = True, figsize = (10, 5))
        gs = gridspec.GridSpec(1, 2)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        x_center, y_center = self.y_size//2, self.x_size//2

        fig.suptitle('LIBS Signal Analysis', fontsize = 20)

        #######################################################
        #                                                     #
        #       Plot Spectrum (Average, MinMax, Point)        #
        #                                                     #
        #######################################################

        axs = ax1
        axs.plot(self.wavelengths, mean_signal, lw = 2, ls = '-', color = 'lightblue', label = 'Mean')
        meanr, = axs.plot(self.wavelengths, self.dataset[x_center - radius:x_center + radius, y_center - radius:y_center + radius].mean(axis = (0, 1)),
                        color = 'darkblue',
                        label = 'Point Mean',
                        lw = 2)
        axs.fill_between(self.wavelengths, min_signal, max_signal, color = 'steelblue', alpha = 0.2)

        wn = 120
        line = axs.axvline(self.wavelengths[wn], lw = '1', alpha = 0.5, color = 'red', label = 'Mapped Wavelength')
        axs.set_xlabel(r'Wavelength $(nm)$')
        axs.set_ylabel(r'Intensity (arb.un.)')
        axs.legend(fancybox = True, shadow = True)
        axs.grid(False)

        #######################################################
        #                                                     #
        #  Spatial Distribuition of selected emission line    #
        #                                                     #
        #######################################################

        axs = ax2
        axs.set_title('Spatial Distribuition')
        spatial_dist = axs.imshow(self.dataset[:, :, wn], cmap = cmap, interpolation = 'gaussian')
        sca = axs.scatter(x_center, y_center, color = 'k', s = 40)
        axs.set_xlabel(r'$x(mm)$')
        axs.set_ylabel(r'$y(mm)$')
        axs.grid(False)

        #######################################################
        #                                                     #
        #             Functions for Interaction               #
        #                                                     #
        #######################################################

        def update_map(wn):
            spatial_dist.set_data(self.dataset[:, :, wn]) 
            spatial_dist.set_clim(vmin = self.dataset[:, :, wn].min(), vmax = self.dataset[:, :, wn].max())
            line.set_xdata(self.wavelengths[wn])

        def onclick(event):
            if event.dblclick:
                if event.inaxes == ax1:
                    ix, _ = event.xdata, event.ydata
                    wn = self.wavelength_to_index(ix)
                    update_map(wn)
                    fig.canvas.draw_idle()
                elif event.inaxes == ax2:
                    xx, yy = int(event.xdata), int(event.ydata)
                    sca.set_offsets([xx, yy])
                    data_region = self.dataset[yy - radius:yy + radius, xx - radius:xx + radius]
                    if data_region.shape == (2*radius, 2*radius, self.spectral_size) and radius != 0:
                        meanr.set_data(self.wavelengths, data_region.mean(axis = (0, 1)))
                    else:
                        meanr.set_data(self.wavelengths, self.dataset[yy, xx])
                    fig.canvas.draw_idle()
                
        cid = fig.canvas.mpl_connect('button_press_event', onclick)


        fig.tight_layout()