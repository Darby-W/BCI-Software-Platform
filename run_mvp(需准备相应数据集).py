import argparse
import numpy as np
import pandas as pd  # 新增：导入pandas解析CSV
from scipy.io import loadmat  # 可选：导入scipy解析MAT文件
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from algorithms.registry import AlgorithmRegistry

def main():
    parser = argparse.ArgumentParser(description="Run algorithm plugin.")
    parser.add_argument("--algo", required=True, help="Name of the algorithm to run (e.g., baseline, dummy)")
    args = parser.parse_args()

    # 1. 自动发现并加载所有算法插件
    AlgorithmRegistry.discover()

    # 2. 根据 --algo 参数获取算法实例
    try:
        algo_class = AlgorithmRegistry.get(args.algo)
        algo = algo_class()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # 3. 加载自定义数据集（替换原示例数据）
    try:
        if args.algo == "baseline":
            # 加载分类任务的自定义CSV数据集
            # 请替换为你的CSV文件路径
            data_path = "./data/my_bci_classify.csv"
            df = pd.read_csv(data_path)
            # 分离特征（X）和标签（y）：假设最后一列是标签
            X = df.iloc[:, :-1].values  # 所有行，除最后一列外的所有列
            y = df.iloc[:, -1].values   # 所有行，最后一列
            # 划分训练/测试集（保持和原代码一致的参数）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        elif args.algo == "dummy":
            # 加载回归任务的自定义CSV数据集
            data_path = "./data/my_bci_regress.csv"
            df = pd.read_csv(data_path)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            print(f"Error: No custom data configured for {args.algo}")
            return
    except FileNotFoundError:
        print(f"Error: Dataset file not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 4. 训练和预测（不变）
    print(f"Running algorithm: {algo.name}")
    algo.train(X_train, y_train)
    y_pred = algo.predict(X_test)

    # 5. 计算并输出指标（不变）
    print("\n--- Evaluation Metrics ---")
    if args.algo == "baseline":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
    elif args.algo == "dummy":
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()