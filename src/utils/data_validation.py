"""
数据验证模块
验证EEG数据格式和质量
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DataQualityReport:
    """数据质量报告"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    channel_quality: Dict[str, float]
    snr_estimate: float
    artifact_ratio: float


class DataValidator:
    """
    数据验证器
    检查EEG数据的质量和完整性
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def validate_eeg_data(
        self,
        data: np.ndarray,
        sample_rate: float,
        channel_names: List[str]
    ) -> DataQualityReport:
        """
        验证EEG数据
        
        Args:
            data: EEG数据 (n_channels, n_samples)
            sample_rate: 采样率 (Hz)
            channel_names: 通道名称列表
        
        Returns:
            数据质量报告
        """
        self.issues = []
        self.warnings = []
        
        # 1. 检查数据形状
        self._check_shape(data, channel_names)
        
        # 2. 检查数值范围
        self._check_range(data)
        
        # 3. 检查采样率
        self._check_sample_rate(sample_rate)
        
        # 4. 检查通道质量
        channel_quality = self._check_channel_quality(data, channel_names)
        
        # 5. 估计信噪比
        snr = self._estimate_snr(data)
        
        # 6. 估计伪迹比例
        artifact_ratio = self._estimate_artifact_ratio(data)
        
        return DataQualityReport(
            valid=len(self.issues) == 0,
            issues=self.issues,
            warnings=self.warnings,
            channel_quality=channel_quality,
            snr_estimate=snr,
            artifact_ratio=artifact_ratio
        )
    
    def _check_shape(self, data: np.ndarray, channel_names: List[str]):
        """检查数据形状"""
        if data.ndim != 2:
            self.issues.append(f"数据维度应为2维，实际为{data.ndim}维")
            return
        
        n_channels, n_samples = data.shape
        
        if n_channels != len(channel_names):
            self.issues.append(
                f"通道数不匹配: 数据有{n_channels}通道，通道名有{len(channel_names)}个"
            )
        
        if n_samples == 0:
            self.issues.append("数据为空")
        elif n_samples < 100:
            self.warnings.append(f"数据点过少: {n_samples}个采样点")
    
    def _check_range(self, data: np.ndarray):
        """检查数值范围"""
        if np.any(np.isnan(data)):
            self.issues.append("数据包含NaN值")
        
        if np.any(np.isinf(data)):
            self.issues.append("数据包含无穷值")
        
        max_val = np.max(np.abs(data))
        if max_val > 1e3:
            self.warnings.append(f"数据幅值过大: {max_val:.2f} µV")
        elif max_val < 1:
            self.warnings.append(f"数据幅值过小: {max_val:.2f} µV")
    
    def _check_sample_rate(self, sample_rate: float):
        """检查采样率"""
        if sample_rate <= 0:
            self.issues.append(f"采样率无效: {sample_rate}")
        elif sample_rate < 100:
            self.warnings.append(f"采样率过低: {sample_rate} Hz")
        elif sample_rate > 2000:
            self.warnings.append(f"采样率过高: {sample_rate} Hz")
    
    def _check_channel_quality(
        self,
        data: np.ndarray,
        channel_names: List[str]
    ) -> Dict[str, float]:
        """检查各通道质量"""
        quality = {}
        
        for i, name in enumerate(channel_names[:data.shape[0]]):
            channel_data = data[i]
            
            # 计算方差（信号变异度）
            variance = np.var(channel_data)
            
            # 计算平坦度（是否有死电极）
            flat_ratio = np.sum(np.abs(channel_data) < 1e-6) / len(channel_data)
            
            if flat_ratio > 0.5:
                quality[name] = 0.0  # 坏通道
                self.warnings.append(f"通道 {name} 可能为死电极")
            elif variance < 1:
                quality[name] = 0.5  # 低质量
                self.warnings.append(f"通道 {name} 信号变异度过低")
            else:
                quality[name] = 1.0  # 良好
        
        return quality
    
    def _estimate_snr(self, data: np.ndarray) -> float:
        """估计信噪比"""
        # 简化计算：使用信号均方根与估计噪声的比值
        signal_rms = np.sqrt(np.mean(data**2))
        
        # 估计噪声（高频成分）
        from scipy import signal
        b, a = signal.butter(4, 0.5, 'high', analog=False)
        noise = signal.filtfilt(b, a, data)
        noise_rms = np.sqrt(np.mean(noise**2))
        
        if noise_rms > 0:
            snr = 20 * np.log10(signal_rms / noise_rms)
        else:
            snr = np.inf
        
        return snr
    
    def _estimate_artifact_ratio(self, data: np.ndarray) -> float:
        """估计伪迹比例"""
        # 使用峰峰值检测伪迹
        peak_to_peak = np.max(data, axis=1) - np.min(data, axis=1)
        
        # 超过阈值的视为伪迹
        threshold = np.percentile(peak_to_peak, 95)
        artifact_channels = np.sum(peak_to_peak > threshold * 1.5)
        
        ratio = artifact_channels / data.shape[0]
        
        return ratio