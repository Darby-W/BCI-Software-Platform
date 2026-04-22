"""
配置管理模块
管理实验配置和系统设置
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SystemConfig:
    """系统配置"""
    debug: bool = False
    log_level: str = "INFO"
    output_dir: str = str(PROJECT_ROOT / "results")
    cache_dir: str = str(PROJECT_ROOT / "cache")
    max_workers: int = 4
    gpu_enabled: bool = False
    gpu_id: int = 0


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_id: str
    experiment_name: str
    subject_id: str
    paradigm: str = "motor_imagery"
    n_trials: int = 100
    trial_duration: float = 4.0
    inter_trial_interval: float = 2.0
    
    # 数据采集配置
    device: str = "BrainProducts"
    n_channels: int = 32
    sample_rate: int = 500
    
    # 预处理配置
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch_freq: float = 50.0
    
    # 特征提取配置
    feature_method: str = "CSP"
    time_window: tuple = (0.5, 2.5)
    
    # 分类配置
    algorithm: str = "EEGNet"
    cv_folds: int = 10
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class ConfigManager:
    """
    配置管理器
    管理实验配置和系统设置
    """
    
    def __init__(self, config_dir: str = None):
        default_config_dir = PROJECT_ROOT / "config"
        self.config_dir = Path(config_dir) if config_dir else default_config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.system_config = SystemConfig()
        self.experiment_configs = {}
    
    def load_system_config(self, filename: str = "system.yaml") -> SystemConfig:
        """加载系统配置"""
        config_path = self.config_dir / filename
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                if filename.endswith('.yaml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self.system_config = SystemConfig(**data)
        
        return self.system_config
    
    def save_system_config(self, filename: str = "system.yaml"):
        """保存系统配置"""
        config_path = self.config_dir / filename
        
        data = asdict(self.system_config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if filename.endswith('.yaml'):
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_experiment_config(
        self,
        experiment_id: str,
        **kwargs
    ) -> ExperimentConfig:
        """创建实验配置"""
        config = ExperimentConfig(experiment_id=experiment_id, **kwargs)
        self.experiment_configs[experiment_id] = config
        return config
    
    def save_experiment_config(
        self,
        experiment_id: str,
        filename: Optional[str] = None
    ):
        """保存实验配置"""
        if experiment_id not in self.experiment_configs:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if filename is None:
            filename = f"{experiment_id}.json"
        
        config_path = self.config_dir / filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(
                self.experiment_configs[experiment_id].to_dict(),
                f, indent=2, ensure_ascii=False
            )
    
    def load_experiment_config(self, filename: str) -> ExperimentConfig:
        """加载实验配置"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {filename} not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = ExperimentConfig(**data)
        self.experiment_configs[config.experiment_id] = config
        return config
    
    def get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """获取实验配置"""
        return self.experiment_configs.get(experiment_id)
    
    def list_experiments(self) -> list:
        """列出所有实验配置"""
        return list(self.experiment_configs.keys())