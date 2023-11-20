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
from sklearn.decomposition import PCA

mpl.use('TkAgg')

# 读取数据
color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array,lerp_list,lerp_array = common.read_data()

# X_train = np.concatenate((color1_array, color2_array), axis=1)
X_train = np.column_stack((color1_array,color2_array,lerp_array))

y_train = blendcolor_array

def create_poly_model(degree=2):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


def generate_data(input_colors1, input_colors2, mix_ratios, output_colors):
    # X = np.column_stack((input_colors1, input_colors2, mix_ratios))
    X = np.hstack((input_colors1, input_colors2, mix_ratios.reshape(-1, 1)))
    y = output_colors
    return X, y

def train_model(model, X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X, y)
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print("Mean squared error: ", mse)

    poly_features = model.named_steps['polynomialfeatures']
    linear_regression = model.named_steps['linearregression']
    try:
        print("Polynomial coefficients: \n", poly_features.get_feature_names_out())
    except AttributeError:
        print("Polynomial coefficients: \n", poly_features.get_feature_names())
    print("Regression coefficients: \n", linear_regression.coef_)
    print("Regression rank: \n", linear_regression.rank_)


    return model


def train_polynomial_regression_model(X1, X2, y, degree=2):
    input_data = np.hstack((X1, X2))
    poly = PolynomialFeatures(degree=degree)
    input_data_poly = poly.fit_transform(input_data)
    model = LinearRegression()
    model.fit(input_data_poly, y)
    return model, poly


def predict_color(model, color1, color2, mix_ratio):
    # input_data = np.array([color1 + color2 + [mix_ratio]])
    input_data = np.hstack((color1, color2, mix_ratio.reshape(-1, 1)))
    return model.predict(input_data)[0]



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

# PCA算法
def pca(data, n_components):
    # 1. 数据标准化
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_normalized = (data - mean) / std

    # 2. 计算协方差矩阵
    covariance_matrix = np.cov(data_normalized.T)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 4. 选择主成分
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, :n_components]

    # 5. 转换数据
    transformed_data = data_normalized.dot(eigenvectors)

    return transformed_data

def testdegreee(max_degree):
    best_degree = 1
    best_score = -np.inf

    for degree in range(1, max_degree + 1):
        score = get_cv_score(degree)
        if score > best_score:
            best_score = score
            best_degree = degree

    print("best degree : ", best_degree)
    return best_degree

# ----- PCA -----
# 降维到3维
n_components = 3

# pca = PCA(n_components = n_components)
# X_pca = pca.fit_transform(X_train)
# print(X_pca)

# ----- PCA -----

max_degree = 10  # 设置要测试的最大多项式次数
best_degree = 3


# X, y = generate_data(color1_list, color2_list, lerp_list, blendcolor_list)
X, y = generate_data(color1_array, color2_array, lerp_array, blendcolor_array)
model = create_poly_model()
trained_model = train_model(model, X, y)



color1 = [1, 1, 0]
color2 = [0, 0, 1]
mix_ratios = np.array([0.5])
predicted_color = predict_color(trained_model, color1, color2, mix_ratios)
print(predicted_color)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# #
s = np.arange(0, 1, 0.05)
for i in s:
    mix_ratio = i
    predicted_color = predict_color(trained_model, color1, color2, mix_ratio)
#     # print(predicted_color)
    ax.scatter(predicted_color[0], predicted_color[1], predicted_color[2], color='b', label='Original Data')
#
#

# X_new =  np.hstack((A_new, B_new)).reshape(1, -1)

# X_transformed_data = pca.transform(X_new)

# X_new_poly = poly.transform(X_new)

# C_new = reg.predict(X_new_poly)


# 获取多项式系数
# coefs = model.coef_
# intercept = model.intercept_
#
# print("多项式系数：", coefs)
# print("截距：", intercept)


# 绘制原始数据点

# ax.scatter(color2_array[:, 0], color2_array[:, 1], color2_array[:, 2], color='b', label='Original Data')
# ax.scatter(blendcolor_array[:, 0], blendcolor_array[:, 1], blendcolor_array[:, 2], color='r', label='Original Data')


ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.show()
# plt.show()