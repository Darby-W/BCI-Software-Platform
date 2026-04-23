# src/pipeline/run_pipeline.py
import os
import numpy as np
from datetime import datetime

from src import BCIDataSystem
from src import AlgorithmRegistry
from src.utils.paths import default_third_party_data_dir, resolve_project_path

from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import Preprocessing
from src.feature_extraction import FeatureExtractor
from src.feature_extraction.csp_multiclass import MultiClassCSP

from src.data_mgmt.storage.data_hierarchical_directory_structure import log_experiment
import pandas as pd


def run_pipeline(algo_name="svm", data_dir=None, low=8, high=30, csp_reg=0.01, csp_components=4):
    print("========== BCI Pipeline Start ==========")

    # ===== 数据加载 =====
    configured_data_dir = data_dir or os.getenv("BCI_DATA_DIR")
    resolved_data_dir = resolve_project_path(configured_data_dir) if configured_data_dir else default_third_party_data_dir()

    bci = BCIDataSystem(data_dir=str(resolved_data_dir))
    data_ids = bci.query_data()
    print("检测到数据:", data_ids)
    if not data_ids:
        raise ValueError("没有找到数据")

    # ===== 获取所有9个被试编号 A01-A09 =====
    subjects = []
    for d in data_ids:
        if d.startswith("A") and len(d) >= 3:
            sub = d[:3]
            if sub not in subjects:
                subjects.append(sub)
    subjects = sorted(subjects)
    print(f"自动识别到被试: {subjects}")

    # 存储所有被试的结果
    all_metrics = []

    # ===== 遍历每一个被试，独立训练+测试 =====
    for sub_id in subjects:
        start_time = datetime.now().isoformat()
        print(f"\n=====================================")
        print(f"正在处理 被试: {sub_id}")

        train_id = f"{sub_id}T"
        test_id = f"{sub_id}E"

        # 加载训练数据
        print(f"使用训练集: {train_id}")
        X_train, y_train, meta = bci.load_feature(train_id)

        # 加载测试数据
        X_test, y_test = None, None
        if test_id in data_ids:
            print(f"使用测试集: {test_id}")
            try:
                X_test, y_test, _ = bci.load_feature(test_id)
                if y_test is None or len(y_test) == 0:
                    print("⚠️ E数据无标签，只做预测")
                    y_test = None
            except Exception:
                print("⚠️ 读取E数据失败，只做预测")
                X_test, y_test = None, None

        # ===== 预处理 =====
        fs = meta.get("sampling_rate", 250)
        preprocessing = Preprocessing(fs, low=low, high=high)
        X_train = preprocessing.apply(X_train)
        if X_test is not None:
            X_test = preprocessing.apply(X_test)

        # ===== 标准化 =====
        X_train = (X_train - np.mean(X_train, axis=-1, keepdims=True)) / (
                    np.std(X_train, axis=-1, keepdims=True) + 1e-6)
        if X_test is not None:
            X_test = (X_test - np.mean(X_test, axis=-1, keepdims=True)) / (
                        np.std(X_test, axis=-1, keepdims=True) + 1e-6)

        # ===== 数据截取（SVM需要，EEGNet不截取） =====
        if algo_name == "svm":
            start = int(0.5 * fs)
            end = int(2.5 * fs)
            X_train = X_train[:, :, start:end]
            if X_test is not None:
                X_test = X_test[:, :, start:end]
        elif algo_name == "eegnet":
            pass

        # ===== 特征提取 =====
        if algo_name == "svm":
            print("使用 MultiClass CSP 特征")
            csp = MultiClassCSP(n_components=csp_components, reg=csp_reg)
            X_train = csp.fit_transform(X_train, y_train)
            if X_test is not None:
                X_test = csp.transform(X_test)
        elif algo_name == "eegnet":
            print("使用 EEGNet 特征")
        else:
            extractor = FeatureExtractor(fs)
            X_train = extractor.extract(X_train)
            if X_test is not None:
                X_test = extractor.extract(X_test)

        # ===== 模型训练 =====
        AlgoClass = AlgorithmRegistry.get(algo_name)
        model = AlgoClass()
        model.train(X_train, y_train)

        # ===== 训练指标 =====
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="macro")
        print(f"【{sub_id}】训练指标 -> Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

        # ===== 预测 =====
        y_pred = None
        test_acc, test_f1 = None, None
        if X_test is not None:
            y_pred = model.predict(X_test)
            if y_test is not None:
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average="macro")
                print(f"【{sub_id}】测试指标 -> Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            else:
                print(f"【{sub_id}】预测未知E数据完成")
                pd.DataFrame(y_pred, columns=[f"prediction_{sub_id}"]).to_csv(f"predictions_{sub_id}.csv", index=False)

        # ===== 日志记录 =====
        metrics = {
            "subject": sub_id,
            "train_accuracy": float(train_acc),
            "train_f1": float(train_f1)
        }
        if test_acc is not None:
            metrics["test_accuracy"] = float(test_acc)
            metrics["test_f1"] = float(test_f1)
        all_metrics.append(metrics)

        end_time = datetime.now().isoformat()
        log_experiment(
            exp_name=f"BCI_{algo_name}_{sub_id}",
            data_id=f"{train_id}_to_{test_id}" if test_id else f"{train_id}_split",
            parameters={"algorithm": algo_name, "low": low, "high": high, "csp_reg": csp_reg,
                        "csp_components": csp_components},
            results=metrics,
            status="success",
            start_time=start_time,
            end_time=end_time
        )

    # ===== 输出所有被试的平均结果 =====
    print("\n=====================================")
    print("========== 所有9个被试 最终结果 ==========")
    avg_train_acc = np.mean([m["train_accuracy"] for m in all_metrics])
    avg_train_f1 = np.mean([m["train_f1"] for m in all_metrics])
    print(f"平均训练准确率: {avg_train_acc:.4f}")
    print(f"平均训练F1: {avg_train_f1:.4f}")

    # 计算测试集平均值（如果有）
    test_acc_list = [m["test_accuracy"] for m in all_metrics if "test_accuracy" in m]
    if test_acc_list:
        avg_test_acc = np.mean(test_acc_list)
        avg_test_f1 = np.mean([m["test_f1"] for m in all_metrics if "test_f1" in m])
        print(f"平均测试准确率: {avg_test_acc:.4f}")
        print(f"平均测试F1: {avg_test_f1:.4f}")

    print("\nPipeline全部完成！")
    return all_metrics


if __name__ == "__main__":
    run_pipeline(algo_name="eegnet")