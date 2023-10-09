
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

csvfilepath = "E:/UnityProjects/BlendColor/Assets/StreamingAssets/data.csv"

wine_reviews = pd.read_csv(csvfilepath)

def num_sum(wine_reviews,name):
    new_num_list = []
    for i in wine_reviews[name]:
        new_i = i.replace("(", ";")
        new_ii = new_i.replace(")", ";").split(";")[1:-2]
        new_num = []

        for num in new_ii:
            new_num.append(float(num))
        new_num_list.append(new_num)

    return new_num_list


# read data from csv
color1_list = num_sum(wine_reviews,"Color1")
color1_array = np.array(color1_list)
color2_list = num_sum(wine_reviews,"Color2")
color2_array = np.array(color2_list)
blendcolor_list = num_sum(wine_reviews,"BlendColor")
blendcolor_array = np.array(blendcolor_list)

# 输入参数为两个RGB颜色和一个验证RGB颜色
def predict_color(color1, color2, val_color):
    # 将RGB颜色转换为向量
    vector1 = np.array(color1)
    vector2 = np.array(color2)
    val_vector = np.array(val_color)

    # 将向量合并为矩阵
    X = np.vstack((vector1, vector2))

    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(X, val_vector)

    # 预测输出颜色
    predicted_vector = model.predict(X)
    predicted_color = tuple(predicted_vector.astype(int))

    return predicted_color



# 创建线性回归模型
model = LinearRegression()

# fit model
model.fit(color1_array, color2_array)

# make a prediction
yhat = model.predict(blendcolor_array)

# summarize prediction
print(yhat)