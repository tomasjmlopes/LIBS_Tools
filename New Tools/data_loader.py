import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view

class LIBS_Toolkit:
    """
    Python Toolkit designed to handle LIBS datasets.
    It provides tools for loading data and performing 
    manipulations, namely:
        - Feature extraction (manual and automatic)
        - Baseline removal
        - Normalization
    """
    def __init__(
        self,
        fname,
        resolution = 0.5
    ):  
        self.fname = fname
        self.dataset = None
        self.wavelengths = None
        self.positions = None
        self.x_size = None
        self.y_size = None
        self.spectral_size = None
        self.normalized_dataset = None
        self.baseline_corrected_dataset = None
        self.features = None
        self.x_features = None
        self.resolution = resolution
    
    def load_dataset(self, init_wv = False, final_wv = False, baseline_corrected = True, return_pos = False):
        """
        Load Entire Dataset (spectrums and wavelengths).
        By setting positions to True it also load all the
        coordiantes of each spectrum.
        """
        hf = h5py.File(self.fname, 'r')
        sample = self.fname.split("\\")[-1]
        keys = [key for key in hf.keys()]
        sample = keys[0].split(' ')[-1]

        baseline = 'Pro' if baseline_corrected else "raw_spectrum"

        spectrums = np.array([np.ndarray.flatten(np.array(list(hf['Sample_ID: ' + sample]['Spot_' + str(i)]['Shot_0'][baseline]))) for i in range(0,
                                                            len(list(hf['Sample_ID: ' + sample])))])
        positions = np.array([np.ndarray.flatten(np.array(list(hf['Sample_ID: ' + sample]['Spot_' + str(i)]['position']))) for i in range(0,
                                                            len(list(hf['Sample_ID: ' + sample])))])
        unique_x = np.unique(positions[:, 1])
        unique_y = np.unique(positions[:, 0])
        
        self.x_size = len(unique_x)
        self.y_size = len(unique_y)
        indexes = np.lexsort((positions[:, 0], positions[:, 1]))

        if init_wv & final_wv:
            spectrums = spectrums[indexes, init_wv : final_wv]
            self.wavelengths = np.array(hf['System properties']['wavelengths']).flatten()[init_wv : final_wv]
        else:
            spectrums = spectrums[indexes, :]
            self.wavelengths = np.array(hf['System properties']['wavelengths']).flatten()

        hf.close()
        self.spectral_size = len(self.wavelengths)
        spectrums = self._map_shape(spectrums)
        positions = np.array(positions[indexes])
        self.dataset = spectrums
        if return_pos:
            self.positions = positions

    def load_wavelengths(self, file):
        """
        Load Wavelengths
        """
        hf = h5py.File(file, 'r')
        wavel = np.array(hf['System properties']['wavelengths'])
        hf.close()
        return wavel
    
    def wavelength_to_index(self, WoI):
        """
        Find index closest to Wavelength of Interest "WoI"
        """
        return np.argmin(np.abs(self.wavelengths - WoI))
    
    def normalize_to_sum(self, overwrite = False):
        """
        Normalize each sepectrum to its sum
        """
        spe = self.dataset.reshape(self.x_size*self.y_size, self.spectral_size)
        norm = np.array([spe[i, :]/np.sum(spe[i, :]) for i in range(0, spe.shape[0])]).reshape(self.x_size, self.y_size, self.spectral_size)
        if overwrite:
            self.dataset = norm
        else:
            self.normalized_dataset = norm
    
    def _get_baseline(self, dataset, min_window_size = 50, smooth_window_size = None):
        """
        Less accurate but faster baseline removal.  
        Uses a rolling window method and determines the minimum
        in each window.
        """
        print('Calculating baselines')

        if smooth_window_size is None:
            smooth_window_size = 2*min_window_size
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
    def _rolling_min(arr, window_width):
        """
        Calculates the moving minima in each row of the provided array.
        """
        window = sliding_window_view(arr, (window_width,), axis = len(arr.shape) - 1
        )

        return np.amin(window, axis = len(arr.shape))

    @staticmethod
    def _get_smoothing_kernel(window_width):
        """
        Generates a Gaussian smoothin kernel of the desired width.
        """
        kernel = np.arange(-window_width//2, window_width//2 + 1, 1)
        sigma = window_width // 4
        kernel = np.exp(-(kernel ** 2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel
    
    def baseline_correct(self, keep_baselines = False, overwrite = True):
        """
        Subtracts the baselines from the spectra.
        """
        self.dataset = self._flat_spec(self.dataset)
        baselines = self._get_baseline(self.dataset)
        baselines = self._align_baselines_with_spectra(baselines)
        if overwrite:
            self.dataset = np.subtract(
                self.dataset,
                baselines)
        else:
            self.baseline_corrected_dataset = np.subtract(
                self.dataset,
                baselines)
            self.baseline_corrected_dataset = self._map_shape(self.baseline_corrected_dataset)
            
        self.dataset = self._map_shape(self.dataset)

    def _align_baselines_with_spectra(self, baselines):
        """
        Discards the last few pixels of the determined baselines if they are longer than the corresponding spectra.
        """
        return baselines[:, :-(baselines.shape[1] - self.spectral_size)]
    
    def manual_features(self, list_of_wavelengths, sigma = False):
        """
        Extract the wavelengths provided in list of wavelengths
        """
        if sigma:
            self.features = np.array([gaussian_filter(self.dataset[:, :, self.wavelength_to_index(wl)], sigma = sigma) for wl in list_of_wavelengths])
        else:
            self.features = np.array([self.dataset[:, :, self.wavelength_to_index(wl)] for wl in list_of_wavelengths])
        self.x_features = list_of_wavelengths

    def automatic_feature_extraction(self, n_features = 20, smallest_dim_pixels = 5, prominence = 0.1, sigma = False):
        """
        Automatically extract N features from dataset using the FT Feature Finder. 
        It requires the smallest pixel dimension to given along with a prominence 
        for a find peaks algorithm applied to FT metric.
        """
        freqs_x = 2*np.pi*np.fft.fftfreq(self.y_size, self.resolution)
        freqs_y = 2*np.pi*np.fft.fftfreq(self.x_size, self.resolution)

        fft_map = np.array([np.fft.fftshift(np.fft.fft2(self.dataset[:, :, i])) for i in range(0, self.spectral_size)])
        fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0 # Remove DC Component
    
        kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))
    
        object_size_small = smallest_dim_pixels*self.resolution
        size_kspace_small = np.pi/object_size_small
    
        R = abs(np.sqrt(kxx**2 + kyy**2))

        sum1 = np.sum(np.abs(fft_map[:, (R < size_kspace_small)]), axis = (1))
        max1 = np.sum(np.abs(fft_map), axis = (1, 2))
    
        sums = np.divide(sum1, max1, out = np.zeros(sum1.shape, dtype = float))
        sums = np.nan_to_num((sums - np.nanmin(sums))/(np.nanmax(sums) - np.nanmin(sums)), nan = 0.0)

        inds, _ = find_peaks(sums, distance = 5, prominence = prominence)
        self.x_features = self.wavelengths[inds]

        # Sort peaks in decresing order and select first n features
        inds = np.argsort(sums[inds])[::-1][:n_features]

        # Extract Features
        self.x_features = self.x_features[inds]
        self.manual_features(self.x_features, sigma = sigma)
        return sums

    def _flat_spec(self, dataset):
        return dataset.reshape(self.x_size*self.y_size, self.spectral_size)
    
    def _map_shape(self, dataset):
        return dataset.reshape(self.x_size, self.y_size, self.spectral_size)