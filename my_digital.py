import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from PIL import Image


#图像预处理
def Trans(img):
        # 转换为灰度格式
        img = img.convert('L')
        # 缩放图像
        img = img.resize((8, 8), Image.LANCZOS)
        # 调整灰度值
        img = np.array(img) / 16  # 原来的灰度值在0-255之间，我们通过除以16将它们转换为0-16之间
        # SK-Learn的图像是以16为白色，0为黑色的，我的图像是以0为白色，255为黑色，需要反转灰度值
        img = 16 - img
        # 将图像的形状展平，因为分类器需要的输入是一维的
        image = img.flatten()
        image_flattened = np.array([image])  # 将列表转换为NumPy数组
        return image_flattened


# 加载模型
import pickle
with open('perceptron_model.pkl', 'rb') as f:
    best_perceptron = pickle.load(f)
with open('GaussianNB_model.pkl', 'rb') as f:
    best_naive_bayes = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    best_knn = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    best_svm = pickle.load(f)

# 读取图像
import matplotlib.pyplot as plt
import matplotlib as mpl
#读取其中一张图片“8”
image = Image.open(r'D:\Allclass\digit_classification1\digit_classification1\1.jpg')
image_flattened = Trans(image)
#图像输出
plt.imshow(image_flattened.reshape(8,8), cmap='gray')
plt.show()


# 使用最佳模型进行预测
knn_pred = best_knn.predict(image_flattened)
svm_pred = best_svm.predict(image_flattened)
perceptron_pred = best_perceptron.predict(image_flattened)
GaussianNB_pred = best_naive_bayes.predict(image_flattened)

# 输出预测结果
print("KNN Prediction:", knn_pred)
print("SVM Prediction:", svm_pred)
print("perceptron Prediction:", perceptron_pred)
print("GaussianNB Prediction:", GaussianNB_pred)
