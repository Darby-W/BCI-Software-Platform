import numpy as np
from scipy.signal import butter, filtfilt


class BandpassFilter:
    def __init__(self, lowcut=8, highcut=30, fs=250, order=4):
        """
        lowcut : 低截止频率
        highcut: 高截止频率
        fs     : 采样率
        order  : 滤波器阶数
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def _design_filter(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def apply(self, data):
        """
        data: shape (n_samples, n_channels)
        """
        b, a = self._design_filter()

        filtered = filtfilt(b, a, data, axis=1)

        return filtered