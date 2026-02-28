from .registry import AlgorithmRegistry
from .base import BaseAlgorithm

# 当算法模块被导入时，自动注册
def register_algorithm(algo_class):
    AlgorithmRegistry.register(algo_class)
    return algo_class
