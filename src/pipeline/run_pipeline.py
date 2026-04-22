# src/pipeline/run_pipeline.py
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

from src import BCIDataSystem
from src import AlgorithmRegistry

from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import Preprocessing
from src.feature_extraction import FeatureExtractor
from src.feature_extraction.csp_multiclass import MultiClassCSP

from src.data_mgmt.storage.data_hierarchical_directory_structure import log_experiment


# ===== 算法类型注册（关键！）=====
ALGO_TYPE = {
    "svm": "ml",
    "logistic_regression": "ml",
    "eegnet": "dl",
    "eeg_conformer": "dl"
}


def run_pipeline(algo_name="svm", data_dir=None, low=8, high=30, csp_reg=0.01, csp_components=4):
    print("========== BCI Pipeline Start ==========")

    algo_type = ALGO_TYPE.get(algo_name)
    if algo_type is None:
        raise ValueError(f"未知算法类型: {algo_name}")

    print(f"当前算法: {algo_name} ({algo_type.upper()})")

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

    subjects = sorted(list(set([d[:3] for d in data_ids if d.startswith("A")])))

    all_metrics = []

    # ===== 遍历被试 =====
    for sub_id in subjects:
        start_time = datetime.now().isoformat()
        print(f"\n===== 被试 {sub_id} =====")

        train_id = f"{sub_id}T"
        test_id = f"{sub_id}E"

        X_train, y_train, meta = bci.load_feature(train_id)

        X_test, y_test = None, None
        if test_id in data_ids:
            try:
                X_test, y_test, _ = bci.load_feature(test_id)
                if y_test is None or len(y_test) == 0:
                    y_test = None
            except:
                X_test, y_test = None, None

        # ===== 预处理 =====
        fs = meta.get("sampling_rate", 250)
        preprocessing = Preprocessing(fs, low=low, high=high)

        X_train = preprocessing.apply(X_train)
        if X_test is not None:
            X_test = preprocessing.apply(X_test)

        # ===== 标准化 =====
        def normalize(x):
            return (x - np.mean(x, axis=-1, keepdims=True)) / (
                np.std(x, axis=-1, keepdims=True) + 1e-6
            )

        X_train = normalize(X_train)
        if X_test is not None:
            X_test = normalize(X_test)

        # =====================================================
        # =============== ML PIPELINE ==========================
        # =====================================================
        if algo_type == "ml":
            print(">>> 使用 ML Pipeline")

            # 截取时间窗
            start = int(0.5 * fs)
            end = int(2.5 * fs)
            X_train = X_train[:, :, start:end]
            if X_test is not None:
                X_test = X_test[:, :, start:end]

            # CSP 特征
            csp = MultiClassCSP(n_components=csp_components, reg=csp_reg)
            X_train = csp.fit_transform(X_train, y_train)
            if X_test is not None:
                X_test = csp.transform(X_test)

        # =====================================================
        # =============== DL PIPELINE ==========================
        # =====================================================
        elif algo_type == "dl":
            print(">>> 使用 DL Pipeline")
            # ===== 降采样 =====
            X_train = X_train[:, :, ::4]
            if X_test is not None:
                X_test = X_test[:, :, ::4]

            # DL一般不做CSP，不截取
            # 如果你以后想加 sliding window，可以在这里扩展
            pass

        # ===== 模型训练 =====
        AlgoClass = AlgorithmRegistry.get(algo_name)
        model = AlgoClass()

        # DL模型通常需要传更多参数（可扩展）
        model.train(X_train, y_train)

        # ===== 评估 =====
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="macro")

        print(f"训练 Acc={train_acc:.4f}, F1={train_f1:.4f}")

        y_pred, test_acc, test_f1 = None, None, None
        if X_test is not None:
            y_pred = model.predict(X_test)
            if y_test is not None:
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average="macro")
                print(f"测试 Acc={test_acc:.4f}, F1={test_f1:.4f}")
            else:
                pd.DataFrame(y_pred).to_csv(f"pred_{sub_id}.csv", index=False)

        # ===== 记录 =====
        metrics = {
            "subject": sub_id,
            "train_accuracy": float(train_acc),
            "train_f1": float(train_f1)
        }

        if test_acc is not None:
            metrics["test_accuracy"] = float(test_acc)
            metrics["test_f1"] = float(test_f1)

        all_metrics.append(metrics)

        log_experiment(
            exp_name=f"BCI_{algo_name}_{sub_id}",
            data_id=f"{train_id}_to_{test_id}",
            parameters={
                "algorithm": algo_name,
                "type": algo_type
            },
            results=metrics,
            status="success",
            start_time=start_time,
            end_time=datetime.now().isoformat()
        )

    print("\n========== FINAL ==========")
    print(pd.DataFrame(all_metrics).mean(numeric_only=True))

    return all_metrics


if __name__ == "__main__":
    run_pipeline(algo_name="eeg_conformer")