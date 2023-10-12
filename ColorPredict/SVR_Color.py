from sklearn.svm import SVR
import numpy as np
import pandas as pd
import common
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 读取数据
color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array = common.read_data()

X_train = np.concatenate((color1_array, color2_array), axis=1)
y_train = blendcolor_array



model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))
model.fit(X_train, y_train)

# 预测
A_new = np.array([1, 1, 0])
B_new = np.array([0, 0, 1])
X_new = np.concatenate((A_new, B_new)).reshape(1, -1)

C_new = model.predict(X_new)
print("Predicted intermediate color:", C_new)