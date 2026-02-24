import argparse
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from 算法插件注册表及参数调用机制.registry import AlgorithmRegistry

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

    # 3. 加载示例数据
    if args.algo == "baseline":
        # 分类任务数据
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    elif args.algo == "dummy":
        # 回归任务数据
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print(f"Error: No sample data configured for {args.algo}")
        return

    # 4. 训练和预测
    print(f"Running algorithm: {algo.name}")
    algo.train(X_train, y_train)
    y_pred = algo.predict(X_test)

    # 5. 计算并输出指标
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