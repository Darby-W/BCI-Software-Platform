from .band_pass_filter import BandpassFilter
from .Notch_filter import NotchFilter

class Preprocessing:
    def __init__(self, fs):
        self.fs = fs

    def apply(self, X):
        # Notch滤波（类名正确+调用apply方法）
        notch = NotchFilter(freq=50, fs=self.fs)
        X = notch.apply(X)

        # 带通滤波（假设BandpassFilter的apply方法正常）
        bandpass = BandpassFilter(
            lowcut=8,
            highcut=30,
            fs=self.fs
        )
        X = bandpass.apply(X)

        return X