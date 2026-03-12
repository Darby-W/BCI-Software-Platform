# src/pipeline/run_pipeline.py

from src import BCIDataSystem
from src import AlgorithmRegistry

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing.band_pass_filter import BandpassFilter


def run_pipeline(algo_name="svm"):
    """
    最小BCI实验pipeline
    """

    print("========== BCI Pipeline Start ==========")

    # 1 初始化数据系统
    bci = BCIDataSystem(data_dir="D:\\pycharm\\code\\BCI-Software-Platform\\third_party_device_data")

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

    #4带通滤波预处理
    try:
        #先从mete读取采样率，如果没有则默认使用250，这样可以适配不同bci设备
        fs = meta.get("sampling_rate",250)
        bandpass = BandpassFilter(
            lowcut=8,
            highcut=30,
            fs=fs
        )
        X = bandpass.apply(X)#执行滤波并替换原始X
        print(f"完成带通滤波（采样率{fs}Hz，8-30Hz）|滤波后数据形状为: {X.shape}")
    except Exception as e:
        raise RuntimeError(f"未能成功执行带通滤波：{str(e)}")

    # 5划分训练测试
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("训练集:", X_train.shape)
    print("测试集:", X_test.shape)

    # 获取算法
    AlgoClass = AlgorithmRegistry.get(algo_name)

    model = AlgoClass()

    print("使用算法:", algo_name)

    # 6 训练模型
    model.fit(X_train, y_train)

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
    # 测试运行（可切换算法：svm/logistic_reg）
    run_pipeline(algo_name="svm")