
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
A_new = np.array([1, 0, 0])
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

# 输入参数为两个RGB颜色和一个验证RGB颜色
# def predict_color(color1, color2, val_color):
#     # 将RGB颜色转换为向量
#     vector1 = np.array(color1)
#     vector2 = np.array(color2)
#     val_vector = np.array(val_color)
#
#     # 将向量合并为矩阵
#     X = np.vstack((vector1, vector2))
#
#     # 创建线性回归模型
#     model = LinearRegression()
#
#     # 训练模型
#     model.fit(X, val_vector)
#
#     # 预测输出颜色
#     predicted_vector = model.predict(X)
#     predicted_color = tuple(predicted_vector.astype(int))
#
#     return predicted_color
#
#


# # 创建线性回归模型
# model = LinearRegression()
#
# # fit model
# model.fit(color1_array, color2_array)
#
# # make a prediction
# yhat = model.predict(blendcolor_array)
#
# # summarize prediction
# print(yhat)