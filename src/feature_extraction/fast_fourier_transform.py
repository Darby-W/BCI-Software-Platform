import numpy as np


class FFTFeature:
    def __init__(self, fs=250):
        self.fs = fs

    def extract(self, X):
        """
        终极兼容：无论输入2维/3维，都能正确解包
        """
        # 第一步：强制适配为3维
        if len(X.shape) == 2:
            samples, channels = X.shape
            X = X.T.reshape(1, channels, samples)
        elif len(X.shape) != 3:
            raise ValueError(f"X维度必须是2或3维，当前是{len(X.shape)}维")

        # 第二步：解包
        trials, channels, samples = X.shape
        features = []

        for t in range(trials):
            trial_feature = []
            for c in range(channels):
                signal = X[t, c, :]
                fft_vals = np.fft.rfft(signal)
                fft_power = np.abs(fft_vals)
                trial_feature.append(np.mean(fft_power))
            features.append(trial_feature)

        return np.array(features)  # 输出：(trials, channels)