# src/pipeline/run_pipeline.py

import os
import numpy as np
from pathlib import Path
from datetime import datetime

from src import BCIDataSystem
from src import AlgorithmRegistry

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import Preprocessing
from src.feature_extraction import FeatureExtractor
from src.feature_extraction.csp_multiclass import MultiClassCSP

from src.data_mgmt.storage.data_hierarchical_directory_structure import log_experiment


def run_pipeline(algo_name="svm", data_dir=None):

    start_time = datetime.now().isoformat()

    print("========== BCI Pipeline Start ==========")

    # ===== 数据加载 =====
    project_root = Path(__file__).resolve().parents[2]
    configured_data_dir = data_dir or os.getenv("BCI_DATA_DIR") or "src/data_mgmt/data_tools/third_party_device_data"

    resolved_data_dir = Path(configured_data_dir)
    if not resolved_data_dir.is_absolute():
        resolved_data_dir = project_root / resolved_data_dir

    bci = BCIDataSystem(data_dir=str(resolved_data_dir))

    data_ids = bci.query_data()
    if not data_ids:
        raise ValueError("没有找到数据")

    data_id = data_ids[0]

    X, y, meta = bci.load_feature(data_id)

    print("数据形状:", X.shape)

    # ===== 预处理 =====
    fs = meta.get("sampling_rate", 250)

    preprocessing = Preprocessing(fs)
    X = preprocessing.apply(X)

    X = (X - np.mean(X, axis=-1, keepdims=True)) / (
        np.std(X, axis=-1, keepdims=True) + 1e-6
    )

    # ===== 划分数据（🔥关键：必须在CSP前）=====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===== 特征提取 =====
    if algo_name == "svm":

        print("使用 多分类 CSP")

        csp = MultiClassCSP(n_components=2, reg=0.01)

        X_train = csp.fit_transform(X_train, y_train)
        X_test = csp.transform(X_test)

    elif algo_name != "eegnet":

        print("使用 PSD+FFT")

        extractor = FeatureExtractor(fs)

        X_train = extractor.extract(X_train)
        X_test = extractor.extract(X_test)

    else:
        print("EEGNet使用原始数据")

    print("训练集:", X_train.shape)

    # ===== 模型 =====
    AlgoClass = AlgorithmRegistry.get(algo_name)
    model = AlgoClass()

    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    # ===== 指标 =====
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1)
    }

    # ===== 日志 =====
    end_time = datetime.now().isoformat()

    log_experiment(
        exp_name=f"BCI_{algo_name}",
        data_id=data_id,
        parameters={"algorithm": algo_name},
        results=metrics,
        status="success",
        start_time=start_time,
        end_time=end_time
    )

    print("结果:", metrics)

    return metrics

#主程序入口
if __name__ == "__main__":
    # 测试运行（可切换算法：svm/logistic_reg/eegnet）
    run_pipeline(algo_name="svm")