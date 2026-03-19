import numpy as np


def split_eeg_into_trials(X, y, fs=250, trial_duration=2):
    """
    把连续EEG数据切分成试次（核心函数，直接复制）
    参数：
        X: 2维数组 (n_samples, n_channels) → 原始连续数据（你的687000,6）
        y: 1维数组 (n_samples,) → 原始标签（你的687000,）
        fs: 采样率（默认250Hz，和你的数据匹配）
        trial_duration: 每个试次的时长（秒，默认2秒，运动想象实验常用）
    返回：
        X_trials: 3维数组 (n_trials, n_channels, trial_length) → 切分后的试次数据
        y_trials: 1维数组 (n_trials,) → 每个试次对应的标签
    """
    # 1. 计算每个试次的采样点数量（2秒×250Hz=500个点）
    trial_length = int(fs * trial_duration)
    n_samples, n_channels = X.shape

    # 2. 计算能完整切分的试次数（舍去最后不足1个试次的部分）
    n_trials = n_samples // trial_length  # 687000 // 500 = 1374个试次

    # 3. 切分X：把连续数据拆成多个试次
    # 先截断到能整除的长度 → 重塑为 (试次数, 通道数, 试次长度)
    X_trimmed = X[:n_trials * trial_length]  # 截断为1374×500=687000个点（刚好整除）
    X_trials = X_trimmed.reshape(n_trials, n_channels, trial_length)

    # 4. 切分y：每个试次取「出现次数最多的标签」（避免单时间点标签波动）
    y_trials = []
    for i in range(n_trials):
        # 取当前试次对应的标签段
        trial_y = y[i * trial_length: (i + 1) * trial_length]
        # 过滤掉0值（休息标签），如果全是0则保留0
        trial_y_valid = trial_y[trial_y != 0]
        if len(trial_y_valid) > 0:
            # 取出现次数最多的有效标签
            trial_label = np.argmax(np.bincount(trial_y_valid.astype(int)))
        else:
            trial_label = 0  # 全是休息则标签为0
        y_trials.append(trial_label)

    return X_trials, np.array(y_trials)