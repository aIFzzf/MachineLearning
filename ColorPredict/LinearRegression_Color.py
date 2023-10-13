
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import common


import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 读取数据
color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array = common.read_data()


X_train = np.concatenate((color1_array, color2_array), axis=1)

y_train = blendcolor_array


# 拟合模型
model = LinearRegression()
model.fit(X_train, y_train)


# 预测
A_new = np.array([1, 1, 0])
B_new = np.array([0, 0, 1])
X_new = np.concatenate((A_new, B_new)).reshape(1, -1)


C_new = model.predict(X_new)

print("Predicted intermediate color:", C_new)


# 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制原始数据点
# # ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Original Data')
#
# # 创建用于绘制拟合平面的网格
# xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
# zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#
# # 绘制拟合平面
# ax.plot_surface(xx, yy, zz, alpha=0.5, color='r', label='Fitted Plane')
#
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()

