import numpy as np
from scipy.signal import welch


class PSDFeature:
    def __init__(self, fs=250):
        self.fs = fs

    def extract(self, X):
        """
        终极兼容：无论输入2维/3维，都能正确解包
        输入：
          - 2维：(samples, channels) → 时间点×通道
          - 3维：(trials, channels, samples) → 试次×通道×时间点
        输出：(trials, channels) → 试次×通道的PSD特征
        """
        # 第一步：强制适配为3维（核心！解决解包错误）
        if len(X.shape) == 2:
            # 2维→3维：(samples, channels) → (1, channels, samples)
            samples, channels = X.shape
            X = X.T.reshape(1, channels, samples)
        elif len(X.shape) != 3:
            raise ValueError(f"X维度必须是2或3维，当前是{len(X.shape)}维")

        # 第二步：解包（此时必为3维，不会报错）
        trials, channels, samples = X.shape
        features = []

        for t in range(trials):
            trial_feature = []
            for c in range(channels):
                signal = X[t, c, :]
                # 自适应nperseg，避免警告
                nperseg = min(256, len(signal))
                freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
                # 取8-30Hz功率
                band_mask = (freqs >= 8) & (freqs <= 30)
                band_power = np.mean(psd[band_mask])
                trial_feature.append(band_power)
            features.append(trial_feature)

        return np.array(features)  # 输出：(trials, channels)