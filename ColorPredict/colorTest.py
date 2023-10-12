import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 准备数据集
# 例如，这里创建了一个包含几种颜色及其混合值的数据集
colors = [
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
    [255, 255, 0],  # 黄色
    [0, 255, 255],  # 青色
    [255, 0, 255],  # 品红色
]

mixtures = [
    [128, 128, 0],  # 红色和绿色的混合值
    [128, 0, 128],  # 红色和蓝色的混合值
    [0, 128, 128],  # 绿色和蓝色的混合值
]

# 创建输入数据，即两个颜色的组合
X = []
for i in range(len(colors)):
    for j in range(i+1, len(colors)):
        X.append(colors[i] + colors[j])

# 创建输出数据，即混合值
y = mixtures
print(X)
print("----------------------------------------------------------------------------------")
print(y)
# # 将数据集划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型实例
model = LinearRegression()

# 使用训练数据训练模型
model.fit(X, y)

# 使用模型进行预测
color1 = [255, 0, 0]  # 红色
color2 = [0, 255, 0]  # 绿色
predicted_mixture = model.predict([color1 + color2])

print("Predicted mixture for red and green: ", predicted_mixture)