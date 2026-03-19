from .Notch_filter import NotchFilter
from .band_pass_filter import BandpassFilter
import numpy as np

class Preprocessing:
    def __init__(self, fs):
        self.fs = fs

    def apply(self, X):

        notch = NotchFilter(freq=50, fs=self.fs)
        bandpass = BandpassFilter(
            lowcut=8,
            highcut=30,
            fs=self.fs
        )

        # 🔥 支持3D数据（trial级）
        if X.ndim == 3:
            print("检测到3D数据，逐trial进行预处理")

            X_processed = []

            for trial in X:
                # trial: (channels, time)
                trial = notch.apply(trial)
                trial = bandpass.apply(trial)
                X_processed.append(trial)

            return np.array(X_processed)

        # 🔥 原始2D数据（兼容）
        elif X.ndim == 2:
            X = notch.apply(X)
            X = bandpass.apply(X)
            return X

        else:
            raise ValueError(f"不支持的维度: {X.ndim}")