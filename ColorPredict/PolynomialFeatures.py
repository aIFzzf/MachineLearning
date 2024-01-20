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
# color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array,lerp_list,lerp_array = common.read_data()
color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array,lerp_list,lerp_array = common.read_json()


def rgb_to_cmyk(rgb):
    # rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = np.expand_dims(rgb, axis=0)

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    k = 1 - np.max(rgb, axis=1)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    # k = 1 - max(r, g, b)
    # if k == 1:
    #     c = 0
    #     m = 0
    #     y = 0
    # else:
    #     c = (1 - r - k) / (1 - k)
    #     m = (1 - g - k) / (1 - k)
    #     y = (1 - b - k) / (1 - k)
    # return c, m, y, k


    return np.stack([c, m, y, k], axis=1)

def cmyk_to_rgb(cmyk):
    r = 1.0 - min(1.0, cmyk[0] * (1.0 - cmyk[3]) + cmyk[3])
    g = 1.0 - min(1.0, cmyk[1] * (1.0 - cmyk[3]) + cmyk[3])
    b = 1.0 - min(1.0, cmyk[2] * (1.0 - cmyk[3]) + cmyk[3])

    return r,g,b

def create_poly_model(degree=2):
    poly_reg = PolynomialFeatures(degree=degree)

    lin_reg = LinearRegression()

    return poly_reg , lin_reg


def generate_data(input_colors1, input_colors2, mix_ratios, output_colors):
    X_train = np.concatenate((input_colors1, input_colors2), axis=1)
    X_train = np.column_stack((X_train, mix_ratios))
    y_train = output_colors
    return X_train, y_train

def train_model(poly_reg,lin_reg, X, y):
    X_poly = poly_reg.fit_transform(X)
    lin_reg.fit(X_poly,y)

    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print("Mean squared error: ", mse)
    feature_list = []


    try:
        print("Polynomial coefficients: \n", poly_reg.get_feature_names_out())
        feature_list = poly_reg.get_feature_names_out()
    except AttributeError:
        print("Polynomial coefficients: \n", poly_reg.get_feature_names())
        feature_list = poly_reg.get_feature_names()
    print("Regression coefficients: \n", lin_reg.coef_)
    reg_coef = lin_reg.coef_
    print("Regression rank: \n", lin_reg.rank_)


    return poly_reg, lin_reg,feature_list,reg_coef


def train_polynomial_regression_model(X1, X2, y, degree=2):
    input_data = np.hstack((X1, X2))
    poly = PolynomialFeatures(degree=degree)
    input_data_poly = poly.fit_transform(input_data)
    model = LinearRegression()
    model.fit(input_data_poly, y)
    return model, poly


def predict_color(color1, color2, mix_ratio, poly_reg, lin_reg,reg_coef):
    color1 = np.array(color1).reshape(1, -1)
    color2 = np.array(color2).reshape(1, -1)
    input_data = np.concatenate((color1, color2), axis=1)
    input_data = np.append(input_data, mix_ratio).reshape(1, -1)

    poly_input_data = poly_reg.transform(input_data)

    predicted_color = lin_reg.predict(poly_input_data)
    return  np.around(predicted_color[0],2)



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

X, y = generate_data((color1_array), (color2_array), lerp_array, (blendcolor_array))
poly_reg,lin_reg = create_poly_model()
poly_reg ,lin_reg,feature_list,reg_coef = train_model(poly_reg,lin_reg, X, y)

# feature_list =   ['1', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x0 x4', 'x0 x5', 'x0 x6', 'x0 x7', 'x0 x8', 'x1^2', 'x1 x2', 'x1 x3', 'x1 x4', 'x1 x5', 'x1 x6', 'x1 x7', 'x1 x8', 'x2^2', 'x2 x3', 'x2 x4', 'x2 x5', 'x2 x6', 'x2 x7', 'x2 x8', 'x3^2', 'x3 x4', 'x3 x5', 'x3 x6', 'x3 x7', 'x3 x8', 'x4^2', 'x4 x5', 'x4 x6', 'x4 x7', 'x4 x8', 'x5^2', 'x5 x6', 'x5 x7', 'x5 x8', 'x6^2', 'x6 x7', 'x6 x8', 'x7^2', 'x7 x8', 'x8^2']
# reg_coef = np.array( [[-9.25202956e-19,  9.54332972e-01, -7.19484810e-02, -3.41206602e-02,
#   -1.44078255e-03, -5.68245365e-02, -6.83661843e-02, -6.73989474e-02,
#    1.56969606e-02, -2.31623348e-01,  1.35355810e-01,  2.84990687e-02,
#   -5.67557095e-02, -1.02319891e-01,  1.27170725e-01, -2.24343048e-01,
#   -1.46301324e-01,  1.10009613e-01, -1.03565436e+00 , 8.77379465e-02,
#   -6.67988050e-02,  1.06043302e-01, -2.11822304e-01 , 1.65356731e-01,
#   -1.37661187e-01, -4.08730878e-02, -2.42135707e-02 , 9.08153509e-02,
#   -5.30691039e-02, -1.40816245e-01, -1.33641949e-01 , 1.45619518e-01,
#    7.05184933e-02, -3.97080234e-02, -5.71394217e-03 , 8.02565654e-02
#   -4.88542467e-02,  8.14615193e-02, -3.39719094e-02 , 2.29273009e-02,
#    1.24103923e-01,  1.89119057e-02, -5.70357756e-02 ,-1.22077314e-01,
#    1.03622912e+00,  7.67041122e-02, -6.71276296e-02 , 6.53091408e-02,
#    2.77712252e-02, 8.39727722e-02 ,-7.69670093e-02 , 4.35709594e-02,
#    1.99958990e-02, -3.27471942e-02,  2.33203386e-01],
#  [ 2.44079216e-19, -4.09571210e-02,  9.65503536e-01, -5.57983874e-03,
#    1.24027408e-02, -4.96886455e-02, -8.58122636e-02, -7.09476689e-02,
#    2.17593547e-02, -2.18028718e-01,  6.52733951e-02, -7.68830888e-02,
#   -1.13415484e-02, -6.50490552e-05,  1.25354266e-01, -9.93055506e-02,
#   -1.44974151e-01,  1.10036889e-02, -8.66350189e-03,  1.61087769e-01,
#    9.67226774e-02, -1.57674086e-01, -1.00056063e-01,  9.66800045e-02,
#   -2.62596602e-01,  1.25013367e-01, -1.04082076e+00,  2.77320543e-02,
#    1.06958255e-02, -1.46921014e-01, -2.62167788e-01,  1.97015328e-01,
#   -4.31155547e-02, -4.33463970e-02, -9.91145064e-03, -4.17914661e-03,
#    1.18866521e-01, -2.77268652e-02, -2.66788668e-02,  2.76435457e-02,
#    6.05858590e-02, -6.96592800e-02, -8.88537135e-04 , 1.25776046e-02,
#    1.41979897e-02,  1.70916511e-01,  1.07584103e-01, -1.42863674e-01,
#    1.04322143e+00,  2.45395653e-02, 4.06874166e-02,  4.73399716e-02,
#    6.18279001e-04, -2.93621335e-02,  2.15859422e-01],
#  [-2.05779737e-18, -3.46931367e-02, -4.08020745e-02,  9.94268505e-01,
#    8.03135436e-02, -3.17212078e-02, -5.38782099e-02, -4.21462763e-02,
#    1.37601454e-01, -1.24352808e-01,  4.56109736e-02, -4.12879831e-02,
#   -2.71677446e-03,  1.94538317e-02,  1.47793192e-01, -1.23249106e-01,
#   -1.49599575e-01, -2.04052985e-02, -3.90317807e-03,  6.50250671e-02,
#   -6.66170557e-02, -6.80007692e-02, -1.18719867e-01,  1.39106691e-01,
#   -1.14137482e-01,  6.81479975e-02, -1.55881917e-02,  2.22548374e-01,
#   -1.57280178e-01, -1.50511808e-01, -1.13896288e-01, -6.90825208e-02,
#    1.05725144e-01, -1.05845129e+00, -9.56432157e-02, -3.36815068e-02,
#    6.72775195e-02,  1.06993974e-01,  1.34345934e-03,  1.84416664e-02,
#    4.66677440e-02, -4.31158507e-02, -6.70040163e-03,  8.45431062e-03,
#    7.72807034e-03,  6.06068636e-02, -5.89809425e-02, -6.70895962e-02,
#    1.97957378e-02,  1.99380027e-01, -1.98369592e-01,  1.06040570e+00,
#   -1.33179874e-01, -3.21051218e-02,  1.26884377e-01],
#  [-1.08617337e-17,  1.83780085e-02,  3.53219106e-02, -1.14538527e-02,
#    1.03437255e+00,  4.68197892e-02,  6.40402054e-02,  4.84198380e-02,
#   -9.86776223e-03,  1.54060358e-01, -3.99180399e-02,  3.01019798e-02,
#   -5.53417218e-03, -7.90819563e-02, -9.56916310e-02,  1.24338944e-01,
#    1.50482877e-01,  4.31588099e-02,  1.40856006e-02, -5.83861598e-02,
#    1.84776367e-02, -3.26339360e-02,  1.18095059e-01, -1.02697307e-01,
#    1.27803769e-01, -1.51774320e-02,  1.81068338e-02, -6.61955712e-02,
#   -1.40828285e-02,  1.52828525e-01,  1.27195851e-01, -7.28569682e-02,
#    4.68495911e-02,  3.54972409e-02, -4.96613101e-02,  2.43708270e-02,
#   -1.49930836e-02,  2.84438570e-02, -1.65963229e-02, -9.83073936e-01,
#   -5.01571102e-02,  2.41092477e-02, -1.27581002e-02, -7.74123947e-02,
#   -1.40204800e-02, -6.69647963e-02,  1.07697594e-02, -4.00789989e-02,
#   -2.06393908e-02, -6.54628495e-02, -4.21542670e-02, -3.87689437e-02,
#    1.89775493e-02,  9.92747416e-01, -1.56284238e-01]])

common.toShader(feature_list,reg_coef)


color1 = np.array([1, 1, 0])
color2 = np.array([0, 0, 1])


mix_ratios = np.array([0.5])
predicted_color = predict_color((color1), (color2), mix_ratios,poly_reg,lin_reg,reg_coef)
print(predicted_color)
# print(predicted_color)


# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
s = np.arange(0, 1, 0.05)
# predicted_color_list = []
for i in s:
    mix_ratio = i
    predicted_color = predict_color(color1, color2, mix_ratio,poly_reg,lin_reg,reg_coef)
    predicted_color = pow(predicted_color,2.2)
    ax.scatter(predicted_color[0], predicted_color[1], predicted_color[2], color='b', label='Original Data')
    # predicted_color_list.append(predicted_color)
#
#
# predicted_color_array = np.array(predicted_color_list)
#
#
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.show()
# plt.show()