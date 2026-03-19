import numpy as np
from .power_spectral_density import PSDFeature  # 你的PSD类
from .fast_fourier_transform import FFTFeature  # 你的FFT类


class FeatureExtractor:
    def __init__(self, fs=250):
        self.fs = fs
        self.psd = PSDFeature(fs=fs)  # 初始化PSD提取器
        self.fft = FFTFeature(fs=fs)  # 初始化FFT提取器

    def extract(self, X):
        """
        提取PSD+FFT特征并拼接（保证试次数不变）
        输入: X (n_trials, n_channels, n_samples) → 切分后的3维试次数据
        输出: features (n_trials, total_features) → 每个试次的拼接特征
        """
        # 1. 提取PSD特征（n_trials, channels）
        psd_feat = self.psd.extract(X)
        print(f"PSD特征形状: {psd_feat.shape}")  # 应该是(1374, 6)

        # 2. 提取FFT特征（n_trials, channels）
        fft_feat = self.fft.extract(X)
        print(f"FFT特征形状: {fft_feat.shape}")  # 应该是(1374, 6)

        # 3. 拼接特征（按列拼接，试次数不变）
        features = np.hstack([psd_feat, fft_feat])
        print(f"特征提取完成: {features.shape}")  # 应该是(1374, 12)

        return features