import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
import common
from sklearn.pipeline import make_pipeline

mpl.use('TkAgg')

# 读取数据
color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array = common.read_data()

X_train = np.concatenate((color1_array, color2_array), axis=1)

y_train = blendcolor_array


# 创建一个多项式回归模型
poly = PolynomialFeatures(degree=2)

# 使用多项式特征生成器转换输入数据
X_train_poly = poly.fit_transform(X_train)

# 拟合模型
model = LinearRegression()
model.fit(X_train_poly, y_train)


A_new = np.array([1, 1, 0])
B_new = np.array([0, 0, 1])
X_new = np.concatenate((A_new, B_new)).reshape(1, -1)

# 使用多项式特征生成器转换新的输入数据
X_new_poly = poly.transform(X_new)

C_new = model.predict(X_new_poly)
print("Predicted intermediate color:", C_new)


