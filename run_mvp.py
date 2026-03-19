import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.algorithms.registry import AlgorithmRegistry
from src import BCIDataSystem

def main():
    parser = argparse.ArgumentParser(description="算法插件化运行框架")
    parser.add_argument("--algo", required=True, help="要运行的算法名称")
    parser.add_argument("--data_id", required=True, help="数据ID(inquiry from BCIDataSystem")
    parser.add_argument("--data_dir", default=None, help="数据目录（优先于环境变量 BCI_DATA_DIR）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    configured_data_dir = args.data_dir or os.getenv("BCI_DATA_DIR") or "src/data_mgmt/data_tools/third_party_device_data"
    resolved_data_dir = Path(configured_data_dir)
    if not resolved_data_dir.is_absolute():
        resolved_data_dir = project_root / resolved_data_dir

    # ====================== 自动发现所有算法 ======================
    AlgorithmRegistry.discover()

    # ====================== 初始化数据系统 ======================
    bci = BCIDataSystem(data_dir=str(resolved_data_dir))

    print("\n可用数据ID：", bci.query_data())

    # ====================== 加载数据 ======================
    X, y, meta = bci.load_feature(args.data_id)

    print("\n加载数据：", args.data_id)
    print("数据形状：", X.shape)

    # ====================== 自动实例化算法 ======================
    algo_class = AlgorithmRegistry.get(args.algo)
    algo = algo_class()

    # ====================== 自动划分训练/测试集 ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ====================== 自动训练 ======================
    print(f"\n▶ 开始训练算法：{algo.name}")
    algo.train(X_train, y_train)

    # ====================== 自动评估 & 输出指标 ======================
    # 关键：所有输出都在算法内部，主程序 1 行调用即可
    algo.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()