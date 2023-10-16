import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
import common
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mpl.use('TkAgg')

# 读取数据
color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array = common.read_data()

X_train = np.concatenate((color1_array, color2_array), axis=1)

y_train = blendcolor_array

# 交叉验证
def get_cv_score(degree):
    # 创建多项式特征生成器
    poly = PolynomialFeatures(degree=degree)

    # 转换输入数据
    X_train_poly = poly.fit_transform(X_train)

    # 创建并训练线性回归模型
    model = LinearRegression()

    # 使用交叉验证计算分数
    # cv参数表示折叠次数，通常选择5或10
    scores = cross_val_score(model, X_train_poly, y_train, cv=5)

    # 返回平均分数
    return np.mean(scores)

max_degree = 10  # 设置要测试的最大多项式次数
best_degree = 3
# best_score = -np.inf
#
# for degree in range(1, max_degree + 1):
#     score = get_cv_score(degree)
#     if score > best_score:
#         best_score = score
#         best_degree = degree
#
# print("Best polynomial degree:", best_degree)


X = np.hstack((color1_array, color2_array))
y = blendcolor_array

# 创建一个使用二次多项式特征的 Lasso 回归模型
lasso_poly = make_pipeline(PolynomialFeatures(degree=best_degree), Lasso(alpha=0.5))

# 训练模型
lasso_poly.fit(X, y)

# 创建一个多项式回归模型
# poly = PolynomialFeatures(degree=best_degree)
# X_train_poly = poly.fit_transform(X_train)
# model = LinearRegression()
# model.fit(X_train_poly, y_train)


# 数据输入
A_new = np.array([1, 1, 0])
B_new = np.array([0, 0, 1])

# lasso 回归
input_data = np.hstack((A_new, B_new)).reshape(1, -1)
predicted_middle_color = lasso_poly.predict(input_data)
print("Predicted middle color: ", predicted_middle_color)


print("Coefficients: ", lasso_poly.named_steps['lasso'].coef_)
# X_new = np.concatenate((A_new, B_new)).reshape(1, -1)

# 使用多项式特征生成器转换新的输入数据
# X_new_poly = poly.transform(X_new)

# C_new = model.predict(X_new_poly)
# print("Predicted intermediate color:", C_new)

# 获取多项式系数
# coefs = model.coef_
# intercept = model.intercept_
#
# print("多项式系数：", coefs)
# print("截距：", intercept)

# 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制原始数据点
# # ax.scatter(color1_array[:, 0], color1_array[:, 1], color1_array[:, 2], color='b', label='Original Data')
# # ax.scatter(color2_array[:, 0], color2_array[:, 1], color2_array[:, 2], color='b', label='Original Data')
# # ax.scatter(blendcolor_array[:, 0], blendcolor_array[:, 1], blendcolor_array[:, 2], color='r', label='Original Data')
#
# x_values_r = np.linspace(0, 1, 100)
# x_values_g = np.linspace(0, 1, 100)
# x_values_b = np.linspace(0, 1, 100)
# y_values = np.zeros((100, 3))
# lower_bound = 0
# upper_bound = 1
# # input_data[0, 0] =x_values_r[50]
# # input_data[0, 1] =x_values_g[20]
# # input_data[0, 2] =x_values_b[60]
#
# for j in range(100):
#     input_data = np.zeros((1, 6))
#
#     input_data[0,0] = x_values_r
#     input_data[0,1] = x_values_g
#     input_data[0,2] = x_values_b
#     random_value_r = np.random.uniform(lower_bound, upper_bound)
#     random_value_g = np.random.uniform(lower_bound, upper_bound)
#     random_value_b = np.random.uniform(lower_bound, upper_bound)
#     input_data[0,3] = random_value_r
#     input_data[0,4] = random_value_g
#     input_data[0,5] = random_value_b
#
#     print(input_data)
#     # X_train = np.concatenate((color1_array, color2_array), axis=1)
#
#     y_values[j] = model.predict(input_data)
# ax.plot(x_values_r, y_values, 'r-', label='Fitted curve')
#
#
#
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()