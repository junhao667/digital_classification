import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from PIL import Image
# In[]
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
# In[1] 加载手写数据集
# 加载数据
digits = load_digits()
data = digits.data

# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.20, random_state=33)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)


# In[10]
# 扩展2部分代码
from sklearn.decomposition import PCA
# 创建PCA对象
pca = PCA(50)  # 根据需要选择主成分数量
# 对特征进行降维
X = pca.fit_transform(data)

train_x, test_x, train_y, test_y = train_test_split(X, digits.target, test_size=0.20, random_state=33)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)


# In[2]KNN分类器
# 定义KNN分类器和参数范围
knn = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

# 使用K-fold交叉验证和GridSearchCV进行参数寻优
k_fold = KFold(n_splits=5, shuffle=True, random_state=33)
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=k_fold)
knn_grid_search.fit(train_ss_x, train_y)

# 获取最佳KNN分类器和参数
best_knn = knn_grid_search.best_estimator_
best_knn_params = knn_grid_search.best_params_

print("Best KNN Parameters:", best_knn_params)

# 执行K-fold交叉验证
knn_scores = cross_val_score(best_knn, train_ss_x, train_y, cv=k_fold)
knn_accuracy = np.mean(knn_scores)

print("KNN Accuracy (K-fold): %.4lf" % knn_accuracy)

X_test = test_ss_x
y_test = test_y
X_train = train_ss_x
y_train = train_y
# 混淆矩阵
from sklearn.metrics import confusion_matrix

y_pred = knn_grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 计算测试集和训练集下的分类器报告
from sklearn.metrics import classification_report
# 在训练集上进行预测
y_pred =  knn_grid_search.predict(X_train)
# 生成分类器报告
report = classification_report(y_train, y_pred)
print("Train Report: ", report)

# 在测试集上进行预测
y_pred = knn_grid_search.predict(X_test)
# 生成分类器报告
report = classification_report(y_test, y_pred)
print("Test Report: ", report)

# 采用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)
# In[3] SVM分类器
# 定义SVM分类器和参数范围
svm = SVC(probability=True)
svm_param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}

# 使用K-fold交叉验证和GridSearchCV进行参数寻优
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=k_fold)
svm_grid_search.fit(train_mm_x, train_y)

# 获取最佳SVM分类器和参数
best_svm = svm_grid_search.best_estimator_
best_svm_params = svm_grid_search.best_params_

print("Best SVM Parameters:", best_svm_params)

# 执行K-fold交叉验证
svm_scores = cross_val_score(best_svm, train_mm_x, train_y, cv=k_fold)
svm_accuracy = np.mean(svm_scores)

print("SVM Accuracy (K-fold): %.4lf" % svm_accuracy)

X_test = test_mm_x
y_test = test_y
X_train = train_mm_x
y_train = train_y

# 混淆矩阵
from sklearn.metrics import confusion_matrix

y_pred = svm_grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 计算测试集和训练集下的分类器报告
from sklearn.metrics import classification_report
# 在训练集上进行预测
y_pred = svm_grid_search.predict(X_train)
# 生成分类器报告
report = classification_report(y_train, y_pred)
print("Train Report: ", report)

# 在测试集上进行预测
y_pred = svm_grid_search.predict(X_test)
# 生成分类器报告
report = classification_report(y_test, y_pred)
print("Test Report: ", report)
# In[4] Naive Bayes分类器
# 创建Navie Bayes分类器
mnb = MultinomialNB()

# 定义Multinomial Naive Bayes参数范围
mnb_param_grid = {'alpha': [0.1, 1.0, 10.0]}

# 使用K-fold交叉验证和GridSearchCV进行参数寻优
mnb_grid_search = GridSearchCV(mnb, mnb_param_grid, cv=k_fold)
mnb_grid_search.fit(train_mm_x, train_y)

# 获取最佳Multinomial Naive Bayes分类器和参数
best_mnb = mnb_grid_search.best_estimator_
best_mnb_params = mnb_grid_search.best_params_

print("Best Multinomial Naive Bayes Parameters:", best_mnb_params)

# 执行K-fold交叉验证
mnb_scores = cross_val_score(best_mnb, train_mm_x, train_y, cv=k_fold)
mnb_accuracy = np.mean(mnb_scores)

print("Multinomial Naive Bayes Accuracy (K-fold): %.4lf" % mnb_accuracy)

# 混淆矩阵
from sklearn.metrics import confusion_matrix

y_pred = mnb_grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 计算测试集和训练集下的分类器报告
from sklearn.metrics import classification_report
# 在训练集上进行预测
y_pred = mnb_grid_search.predict(X_train)
# 生成分类器报告
report = classification_report(y_train, y_pred)
print("Train Report: ", report)

# 在测试集上进行预测
y_pred = mnb_grid_search.predict(X_test)
# 生成分类器报告
report = classification_report(y_test, y_pred)
print("Test Report: ", report)

# In[5]
# Perceptron分类器
# 创建感知器分类器
ppn = Perceptron()
# 定义Perceptron参数范围
ppn_param_grid = {'eta0': [0.1, 0.01, 0.001, 0.0001], 'max_iter': [500, 1000, 1500, 2000, 2500]}

# 使用K-fold交叉验证和GridSearchCV进行参数寻优
ppn_grid_search = GridSearchCV(ppn, ppn_param_grid, cv=k_fold)
ppn_grid_search.fit(train_ss_x, train_y)

# 获取最佳Perceptron分类器和参数
best_ppn = ppn_grid_search.best_estimator_
best_ppn_params = ppn_grid_search.best_params_

print("Best Perceptron Parameters:", best_ppn_params)

# 执行K-fold交叉验证
ppn_scores = cross_val_score(best_ppn, train_ss_x, train_y, cv=k_fold)
ppn_accuracy = np.mean(ppn_scores)

print("Perceptron Accuracy (K-fold): %.4lf" % ppn_accuracy)

# In[6]
# 计算测试集上的预测准确率
knn_pred = best_knn.predict(test_ss_x)
knn_test_accuracy = accuracy_score(test_y, knn_pred)
knn_pred_prob = best_knn.predict_proba(test_ss_x)

svm_pred = best_svm.predict(test_mm_x)
svm_test_accuracy = accuracy_score(test_y, svm_pred)
svm_pred_prob = best_svm.predict_proba(test_mm_x)

mnb_pred = best_mnb.predict(test_mm_x)
mnb_test_accuracy = accuracy_score(test_y, mnb_pred)
mnb_pred_prob = best_mnb.predict_proba(test_mm_x)

print("KNN Test Accuracy: %.4lf" % knn_test_accuracy)
print("KNN Predicted Probabilities:\n", knn_pred_prob)
print("SVM Test Accuracy: %.4lf" % svm_test_accuracy)
print("SVM Predicted Probabilities:\n", svm_pred_prob)
print("Multinomial Naive Bayes Test Accuracy: %.4lf" % mnb_test_accuracy)
print("Multinomial Naive Bayes Predicted Probabilities:\n", mnb_pred_prob)
print("Perceptron Test Accuracy: %.4lf" % ppn_accuracy)

# In[7]
# 读取图像
import matplotlib.pyplot as plt
import matplotlib as mpl
image = Image.open(r'D:\Allclass\digit_classification1\digit_classification1\0.jpg')# D:/C/Desktop/1
image_flattened = Trans(image)

plt.imshow(image_flattened.reshape(8,8), cmap='gray')
plt.show()


# In[8]
components = pca.components_
image_flattened = image_flattened.dot(components.T)
# 使用最佳模型进行预测
knn_pred = best_knn.predict(image_flattened)
svm_pred = best_svm.predict(image_flattened)
mnb_pred = best_mnb.predict(image_flattened)
perceptron_pred = best_ppn.predict(image_flattened)
# 输出预测结果
print("KNN Prediction:", knn_pred)
print("SVM Prediction:", svm_pred)
print("Multinomial Naive Bayes Prediction:", mnb_pred)
print("perceptron Prediction:", perceptron_pred)

print("KNN Test Accuracy: %.4lf" % knn_test_accuracy)
print("SVM Test Accuracy: %.4lf" % svm_test_accuracy)
print("Multinomial Naive Bayes Test Accuracy: %.4lf" % mnb_test_accuracy)
print("Perceptron Test Accuracy: %.4lf" % ppn_accuracy)
#可视化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 创建PCA对象
pca = PCA(n_components=2)  # 选择主成分数量为2
X = pca.fit_transform(digits.data)

# 将结果可视化
plt.figure(figsize=(10, 10))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# 遍历所有类别（手写数字0到9）
for color, i, target_name in zip(colors, range(10), digits.target_names):
    plt.scatter(X[digits.target == i, 0], X[digits.target == i, 1], color=color, alpha=.8,
                label=target_name)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title('PCA of Digits Dataset')
plt.show()
