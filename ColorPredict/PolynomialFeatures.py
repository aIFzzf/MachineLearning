import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
import common
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

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



# 创建一个多项式回归模型
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)



A_new = np.array([1, 1, 0])
B_new = np.array([0, 0, 1])
X_new = np.concatenate((A_new, B_new)).reshape(1, -1)

# 使用多项式特征生成器转换新的输入数据
X_new_poly = poly.transform(X_new)

C_new = model.predict(X_new_poly)
print("Predicted intermediate color:", C_new)


