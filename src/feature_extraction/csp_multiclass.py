import numpy as np


class MultiClassCSP:
    def __init__(self, n_components=2, reg=0.01):
        """
        n_components: 每个类别取多少对特征
        最终特征维度 = n_classes * 2 * n_components
        """
        self.n_components = n_components
        self.reg = reg

        self.filters_ = {}   # 每个类别一个CSP
        self.classes_ = None

        self.mean_ = None
        self.std_ = None

    # =========================
    # 训练
    # =========================
    def fit(self, X, y):

        self.classes_ = np.unique(y)

        # 标准化（全局）
        self.mean_ = np.mean(X, axis=(0, 2), keepdims=True)
        self.std_ = np.std(X, axis=(0, 2), keepdims=True) + 1e-8
        X = (X - self.mean_) / self.std_

        for c in self.classes_:

            # one-vs-rest
            y_binary = (y == c).astype(int)

            X_pos = X[y_binary == 1]
            X_neg = X[y_binary == 0]

            cov_pos = self._cov(X_pos)
            cov_neg = self._cov(X_neg)

            # 加正则化（核心）
            n_channels = cov_pos.shape[0]

            cov_pos += self.reg * np.eye(n_channels) * np.trace(cov_pos) / n_channels
            cov_neg += self.reg * np.eye(n_channels) * np.trace(cov_neg) / n_channels

            eigvals, eigvecs = np.linalg.eig(
                np.linalg.pinv(cov_neg) @ cov_pos
            )

            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)

            idx = np.argsort(eigvals)[::-1]

            selected = np.concatenate([
                idx[:self.n_components],
                idx[-self.n_components:]
            ])

            self.filters_[c] = eigvecs[:, selected]

        return self

    # =========================
    # 特征提取
    # =========================
    def transform(self, X):

        X = (X - self.mean_) / self.std_

        features_all = []

        for trial in X:

            feat_trial = []

            for c in self.classes_:

                W = self.filters_[c]

                Z = W.T @ trial
                var = np.var(Z, axis=1)

                feat_trial.extend(np.log(var + 1e-8))

            features_all.append(feat_trial)

        return np.array(features_all)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    # =========================
    def _cov(self, X):

        cov = 0

        for trial in X:
            c = trial @ trial.T
            c = c / np.trace(c)
            cov += c

        return cov / len(X)