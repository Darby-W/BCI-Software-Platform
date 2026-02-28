import os
import importlib
from typing import Dict, Type
from .base import BaseAlgorithm

class AlgorithmRegistry:
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}

    @classmethod
    def register(cls, algo_class: Type[BaseAlgorithm]) -> None:
        """注册一个算法类"""
        algo_name = algo_class().name
        if algo_name in cls._algorithms:
            raise ValueError(f"Algorithm {algo_name} already registered.")
        cls._algorithms[algo_name] = algo_class

    @classmethod
    def get(cls, algo_name: str) -> Type[BaseAlgorithm]:
        """根据名称获取算法类"""
        if algo_name not in cls._algorithms:
            raise ValueError(f"Algorithm {algo_name} not found. Available: {list(cls._algorithms.keys())}")
        return cls._algorithms[algo_name]

    @classmethod
    def discover(cls, package_path: str = "算法插件注册表及参数调用机制") -> None:  # 这里的字符串是根目录文件名，可改
        """自动发现并加载指定包下的所有算法插件"""
        for root, _, files in os.walk(package_path):
            for file in files:
                if file.endswith(".py") and file not in {"__init__.py", "_init_.py", "base.py", "registry.py"}:
                    module_name = os.path.splitext(file)[0]
                    module_path = os.path.join(root, module_name).replace(os.sep, ".")
                    importlib.import_module(module_path)