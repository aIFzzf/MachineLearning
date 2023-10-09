from sklearn.svm import SVR
import numpy as np
import pandas as pd

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


def predict_color(color1, color2, ref_color):
    # 将输入颜色和参考颜色转换为numpy数组
    X = np.array([color1, color2])
    y = np.array([ref_color])

    # 训练SVR模型
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr.fit(X, y)

    # 预测输出颜色
    pred_color = svr.predict(X.mean(axis=0).reshape(1, -1))

    # 将预测颜色的值限制在0-255之间
    pred_color = np.clip(pred_color, 0, 255)

    # 将预测颜色转换为整数并返回
    return tuple(pred_color.astype(int)[0])


# read data from csv
color1_list = num_sum(wine_reviews,"Color1")
color1_array = np.array(color1_list)
color2_list = num_sum(wine_reviews,"Color2")
color2_array = np.array(color2_list)
blendcolor_list = num_sum(wine_reviews,"BlendColor")
blendcolor_array = np.array(blendcolor_list)


X = []
for i in range(len(color1_list)):
    for j in range(len(color2_list)):
        X.append(color1_list[i] + color2_list[j])

input_X = np.array([[255,0,0],[0,0,255]])
input_Y =  np.array([128,0,128])
print(input_Y)

svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(input_X, input_Y)





