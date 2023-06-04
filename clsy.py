# 导入所需库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加载手写数字数据集
digits = datasets.load_digits()

# 划分训练集和测试集，测试集占比为0.2
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 创建并训练朴素贝叶斯分类器
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 对测试集进行预测
y_pred = gnb.predict(X_test)

# 计算并输出精度
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# 输出混淆矩阵
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 输出分类器报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵热力图
plt.matshow(confusion_matrix(y_test, y_pred))
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
