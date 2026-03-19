# src/pipeline/run_pipeline.py

import os
import numpy as np
from pathlib import Path

from src import BCIDataSystem
from src import AlgorithmRegistry

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import Preprocessing
from src.feature_extraction import FeatureExtractor

def run_pipeline(algo_name="svm", data_dir=None):
    """
    最小BCI实验pipeline
    """

    print("========== BCI Pipeline Start ==========")

    # 1 初始化数据系统（优先使用函数参数，其次环境变量，最后项目内默认目录）
    project_root = Path(__file__).resolve().parents[2]
    configured_data_dir = data_dir or os.getenv("BCI_DATA_DIR") or "src/data_mgmt/data_tools/third_party_device_data"
    resolved_data_dir = Path(configured_data_dir)
    if not resolved_data_dir.is_absolute():
        resolved_data_dir = project_root / resolved_data_dir

    bci = BCIDataSystem(data_dir=str(resolved_data_dir))

    # 2 查询数据
    data_ids = bci.query_data()

    if not data_ids:
        raise ValueError("没有找到任何数据，请先放入CSV或EDF数据")

    print("可用数据:", data_ids)

    # 3 读取第一份数据
    data_id = data_ids[0]

    print("加载数据:", data_id)

    X, y, meta = bci.load_feature(data_id)

    print("数据形状:", X.shape)
    print("标签形状:", y.shape)

    # 4 预处理
    try:
        fs = meta.get("sampling_rate", 250)
        preprocessing = Preprocessing(fs)

        X = preprocessing.apply(X)

        print(f"完成预处理（Notch + Bandpass） | 数据形状: {X.shape}")

    except Exception as e:
        raise RuntimeError(f"预处理失败: {str(e)}")

    # ========== ✅ 正确做法：不再切 trial ==========
    print(f"原始X shape: {X.shape}")

    if X.ndim == 2:
        raise ValueError("当前数据是二维，说明你CSV加载有问题")

    elif X.ndim == 3:
        print("数据已经是trial级 (N, channels, time)，跳过切分")

    else:
        raise ValueError(f"未知数据维度: {X.shape}")

    if algo_name != "eegnet":
        extractor = FeatureExtractor(fs)
        X = extractor.extract(X)
        print("特征提取完成:", X.shape)
    else:
        print("EEGNet跳过特征提取")

    # 5划分训练测试
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("训练集:", X_train.shape)
    print("测试集:", X_test.shape)

    # 获取算法
    AlgoClass = AlgorithmRegistry.get(algo_name)

    model = AlgoClass()

    print("使用算法:", algo_name)

    # 6 训练模型
    model.train(X_train, y_train)

    # 7 预测
    y_pred = model.predict(X_test)

    # 8 计算指标
    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": acc,
        "f1": f1
    }

    print("===== 实验结果 =====")
    print(metrics)

    print("========== Pipeline End ==========")

    return metrics

#主程序入口
if __name__ == "__main__":
    # 测试运行（可切换算法：svm/logistic_reg/eegnet）
    run_pipeline(algo_name="eegnet")