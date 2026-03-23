import numpy as np
from .power_spectral_density import PSDFeature
from .fast_fourier_transform import FFTFeature


class FeatureExtractor:
    def __init__(self, fs=250):
        self.fs = fs
        self.psd = PSDFeature(fs=fs)
        self.fft = FFTFeature(fs=fs)

    def extract(self, X):
        """
        X: (n_trials, channels, time)
        """

        psd_feat = self.psd.extract(X)
        fft_feat = self.fft.extract(X)

        features = np.hstack([psd_feat, fft_feat])

        return features