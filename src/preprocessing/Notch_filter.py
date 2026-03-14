from scipy.signal import iirnotch, filtfilt
import numpy as np


class NotchFilter:
    """
    陷波滤波器类（Notch Filter）
    用于去除EEG信号中的工频噪声（默认50Hz）
    """

    def __init__(self, fs=250, freq=50, Q=30):
        """
        初始化陷波滤波器
        参数:
            fs: 采样率，默认250Hz
            freq: 要去除的噪声频率，默认50Hz（工频）
            Q: 品质因数（决定滤波器带宽），默认30
        """
        self.fs = fs  # 采样率
        self.freq = freq  # 陷波频率
        self.Q = Q  # 品质因数

        # 错误修正：分开初始化b和a，而非解包None
        self.b = None
        self.a = None

        # 初始化时直接计算滤波器系数
        self._design_filter()

    def _design_filter(self):
        """
        私有方法：设计陷波滤波器，计算滤波器系数b和a
        内部调用，用户无需手动调用
        """
        # 确保iirnotch参数正确（freq, Q, fs），并返回b和a
        self.b, self.a = iirnotch(self.freq, self.Q, self.fs)

    def filter(self, signal):
        """
        对EEG信号进行陷波滤波（核心方法）
        参数:
            signal: EEG信号，支持两种格式：
                    - 单通道：1维数组 (n_samples,)
                    - 多通道：2维数组 (n_samples, n_channels)
        返回:
            filtered_signal: 滤波后的EEG信号，形状与输入一致
        """
        # 输入数据类型检查（可选，但增加鲁棒性）
        if not isinstance(signal, (np.ndarray, list)):
            raise TypeError("输入信号必须是numpy数组或列表")

        # 转换为numpy数组（兼容列表输入）
        signal = np.array(signal)

        # 处理单通道信号
        if signal.ndim == 1:
            filtered_signal = filtfilt(self.b, self.a, signal)

        # 处理多通道信号（逐通道滤波）
        elif signal.ndim == 2:
            filtered_signal = np.zeros_like(signal)
            for ch in range(signal.shape[1]):
                filtered_signal[:, ch] = filtfilt(self.b, self.a, signal[:, ch])

        # 不支持的维度
        else:
            raise ValueError(f"不支持的信号维度：{signal.ndim}，仅支持1维（单通道）或2维（多通道）")

        return filtered_signal

    # 新增apply方法，和BandpassFilter保持一致（适配你的预处理调用）
    def apply(self, signal):
        return self.filter(signal)

    # 可选：提供修改参数的方法（如果需要动态调整）
    def update_params(self, fs=None, freq=None, Q=None):
        """
        更新滤波器参数，并重新设计滤波器
        参数:
            fs: 新的采样率（可选，不传入则保持原值）
            freq: 新的陷波频率（可选）
            Q: 新的品质因数（可选）
        """
        if fs is not None:
            self.fs = fs
        if freq is not None:
            self.freq = freq
        if Q is not None:
            self.Q = Q

        # 重新计算滤波器系数
        self._design_filter()