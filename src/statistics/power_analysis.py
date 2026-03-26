"""
功效分析模块
计算所需样本量、统计功效等
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    from statsmodels.stats.power import (
        TTestPower, TTestIndPower, FTestAnovaPower,
        GofChisquarePower, NormalIndPower
    )
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Power analysis will be limited.")


class PowerAnalyzer:
    """
    统计功效分析器
    用于计算样本量、功效值等
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        self.alpha = alpha
        self.power = power
        
    def sample_size_for_ttest(
        self,
        effect_size: float,
        test_type: str = 'paired',
        alpha: float = None,
        power: float = None
    ) -> int:
        """
        计算t检验所需样本量
        
        Args:
            effect_size: 效应量 (Cohen's d)
            test_type: 检验类型 ('paired', 'independent')
            alpha: 显著性水平（默认使用实例设置）
            power: 统计功效（默认使用实例设置）
        
        Returns:
            所需样本量
        """
        if not STATSMODELS_AVAILABLE:
            return self._estimate_sample_size(effect_size)
        
        alpha = alpha or self.alpha
        power = power or self.power
        
        if test_type == 'paired':
            power_analysis = TTestPower()
        else:
            power_analysis = TTestIndPower()
        
        n = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(n))
    
    def sample_size_for_anova(
        self,
        effect_size: float,
        n_groups: int,
        alpha: float = None,
        power: float = None
    ) -> int:
        """
        计算ANOVA所需样本量
        
        Args:
            effect_size: 效应量 (Cohen's f)
            n_groups: 组数
            alpha: 显著性水平
            power: 统计功效
        
        Returns:
            每组所需样本量
        """
        if not STATSMODELS_AVAILABLE:
            return self._estimate_sample_size(effect_size)
        
        alpha = alpha or self.alpha
        power = power or self.power
        
        power_analysis = FTestAnovaPower()
        n = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            k_groups=n_groups,
            nobs=None
        )
        
        return int(np.ceil(n))
    
    def achieved_power(
        self,
        effect_size: float,
        n_samples: int,
        test_type: str = 'paired',
        alpha: float = None
    ) -> float:
        """
        计算当前样本量能达到的统计功效
        
        Args:
            effect_size: 效应量
            n_samples: 样本量
            test_type: 检验类型
            alpha: 显著性水平
        
        Returns:
            统计功效值
        """
        if not STATSMODELS_AVAILABLE:
            return 0.8  # 估计值
        
        alpha = alpha or self.alpha
        
        if test_type == 'paired':
            power_analysis = TTestPower()
        else:
            power_analysis = TTestIndPower()
        
        power = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            nobs=n_samples,
            power=None,
            alternative='two-sided'
        )
        
        return power
    
    def _estimate_sample_size(self, effect_size: float) -> int:
        """
        粗略估计样本量（无statsmodels时使用）
        
        Args:
            effect_size: 效应量
        
        Returns:
            估计样本量
        """
        # 经验法则
        if effect_size >= 0.8:  # 大效应
            return 26
        elif effect_size >= 0.5:  # 中效应
            return 64
        else:  # 小效应
            return 200
    
    def effect_size_interpretation(self, effect_size: float, test_type: str = 'cohen_d') -> str:
        """
        解释效应量大小
        
        Args:
            effect_size: 效应量值
            test_type: 检验类型
        
        Returns:
            解释文字
        """
        effect_size = abs(effect_size)
        
        if test_type == 'cohen_d':
            if effect_size >= 0.8:
                return "大效应 (Large effect)"
            elif effect_size >= 0.5:
                return "中等效应 (Medium effect)"
            elif effect_size >= 0.2:
                return "小效应 (Small effect)"
            else:
                return "可忽略效应 (Negligible effect)"
        
        elif test_type == 'eta_squared':
            if effect_size >= 0.14:
                return "大效应 (Large effect)"
            elif effect_size >= 0.06:
                return "中等效应 (Medium effect)"
            elif effect_size >= 0.01:
                return "小效应 (Small effect)"
            else:
                return "可忽略效应 (Negligible effect)"
        
        else:
            return f"效应量 = {effect_size:.3f}"
    
    def generate_power_report(
        self,
        effect_size: float,
        n_samples: int,
        test_type: str = 'paired'
    ) -> str:
        """
        生成功效分析报告
        
        Args:
            effect_size: 效应量
            n_samples: 当前样本量
            test_type: 检验类型
        
        Returns:
            格式化的报告文本
        """
        power = self.achieved_power(effect_size, n_samples, test_type)
        required_n = self.sample_size_for_ttest(effect_size, test_type)
        
        report = f"""
### 功效分析报告

**效应量**: {effect_size:.3f} ({self.effect_size_interpretation(effect_size)})

**当前样本量**: N = {n_samples}
**当前统计功效**: {power:.3f} ({'✓ 充足' if power >= 0.8 else '✗ 不足'})

**达到功效0.8所需样本量**: N = {required_n}

**建议**: 
"""
        
        if power >= 0.8:
            report += "当前样本量充足，统计功效良好。\n"
        else:
            report += f"当前样本量不足，建议增加至至少 {required_n} 个样本以获得可靠的统计结论。\n"
        
        return report