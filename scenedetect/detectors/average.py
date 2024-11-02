import numpy as np


# TODO 
# 1. 自适应阈值 可以根据历史的diff值来调整阈值
# 2. 类似kalman filter 的滤波器，可以根据历史的diff值来调整滤波器的参数
# 3. 不用last 的dim 而是更多的历史信息

# kalmal filter


class KalmanFilter:
    def __init__(self, dim, process_variance=1e-5, measurement_variance=1e-1):
        # 初始化卡尔曼滤波参数
        self.dim = dim  # 状态的维度（这里是直方图的bin数）
        self.A = np.eye(dim)  # 状态转移矩阵
        self.H = np.eye(dim)  # 观测矩阵
        self.Q = process_variance * np.eye(dim)  # 过程噪声协方差
        self.R = measurement_variance * np.eye(dim)  # 测量噪声协方差
        self.P = np.eye(dim)  # 估计误差协方差
        self.x = np.zeros(dim)  # 初始状态

    def predict(self):
        # 状态预测
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        # 更新估计
        y = z - self.H @ self.x  # 计算残差
        S = self.H @ self.P @ self.H.T + self.R  # 计算S矩阵
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P
        self.predict()
        return self.x  # 返回更新后的状态


class BaseThreshold:
    def __init__(self):
        pass

    def update(self, diff):
        pass






class BaseFilter:
    def __init__(self):
        pass

    def predict(self):
        pass

    def update(self, z):
        pass


class ExpFilter(BaseFilter):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.x = None

    def predict(self):
        pass

    def update(self, z):
        if self.x is None:
            self.x = z
        else:
            self.x = self.alpha * z + (1 - self.alpha) * self.x
        self.predict()
        return self.x


class AvgFilter(BaseFilter):
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = []

    def predict(self):
        pass

    def update(self, z):
        self.window.append(z)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self.predict()
        return sum(self.window) / len(self.window)



