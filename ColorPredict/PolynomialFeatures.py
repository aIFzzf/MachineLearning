import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl

mpl.use('TkAgg')
# 生成随机数据
np.random.seed(0)
x = np.linspace(-3, 3, 100)
y = np.sin(x) + np.random.randn(100) * 0.1

# 将x转换为二维矩阵
X = x[:, np.newaxis]
print(X)
# 创建多项式特征
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_poly, y)

# 生成预测数据
x_pred = np.linspace(-3, 3, 100)
X_pred = poly.fit_transform(x_pred[:, np.newaxis])
y_pred = model.predict(X_pred)

# 绘制数据和拟合曲线
plt.scatter(x, y, s=10)
plt.plot(x_pred, y_pred, color='r')
plt.show()