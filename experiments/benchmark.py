import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.algorithms.registry import AlgorithmRegistry
from src.data_mgmt.query.data_query_reading_interface import BCIDataSystem
import matplotlib.pyplot as plt


def run_benchmark():

    print("\n==============================")
    print("      BCI Benchmark System")
    print("==============================\n")

    # 发现算法
    AlgorithmRegistry.discover()
    algorithms = AlgorithmRegistry._algorithms

    print("可用算法：", list(algorithms.keys()))

    # 加载数据系统
    bci = BCIDataSystem()

    dataset_ids = bci.query_data()
    print("可用数据：", dataset_ids)

    results = []

    for data_id in dataset_ids:

        print(f"\n加载数据集：{data_id}")

        X, y, meta = bci.load_feature(data_id)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        for algo_name, algo_class in algorithms.items():

            print(f"\n运行算法：{algo_name}")

            algo = algo_class()

            # 训练
            algo.train(X_train, y_train)

            # 预测
            y_pred = algo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            print(f"Accuracy: {acc:.4f}")

            results.append({
                "dataset": data_id,
                "algorithm": algo_name,
                "accuracy": acc,
                "f1_score": f1
            })

    df = pd.DataFrame(results)

    print("\n========== Benchmark Result ==========")
    print(df)

    df.to_csv("benchmark_results.csv", index=False)

    df.plot(
        x="algorithm",
        y="accuracy",
        kind="bar",
        title="Algorithm Comparison"
    )

    plt.show()
    print("\n结果已保存：benchmark_results.csv")



if __name__ == "__main__":
    run_benchmark()