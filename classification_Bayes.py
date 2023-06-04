from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载手写数字数据集
digits = load_digits()

# 将数据集拆分为训练集和测试集

# 使用train_test_split函数划分数据集
# 参数test_size表示测试集占总数据集的比例，这里设置为0.2，即80%的数据用于训练，20%的数据用于测试
# X_train表示训练集的特征数据，X_test表示测试集的特征数据，y_train表示训练集的标签，y_test表示测试集的标签
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 定义贝叶斯分类器模型和参数网格
naive_bayes = GaussianNB()
param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}

# 使用 GridSearchCV 进行参数寻优
grid_search = GridSearchCV(naive_bayes, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳准确率
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

# 使用最佳参数创建贝叶斯分类器模型
best_naive_bayes = GaussianNB(var_smoothing=grid_search.best_params_['var_smoothing'])
best_naive_bayes.fit(X_train, y_train)

# 使用 K-Fold 进行交叉验证并计算平均准确率
scores = cross_val_score(best_naive_bayes, digits.data, digits.target, cv=5)
print("Cross-validation accuracy:", scores.mean())

# 在测试集上计算准确率、混淆矩阵和分类器报告
test_pred = best_naive_bayes.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
test_confusion_matrix = confusion_matrix(y_test, test_pred)
test_classification_report = classification_report(y_test, test_pred)

print("Test accuracy:", test_accuracy)
print("Test confusion matrix:\n", test_confusion_matrix)
print("Test classification report:\n", test_classification_report)

# 可视化混淆矩阵
plt.figure(figsize=(8, 8))
sns.heatmap(test_confusion_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fig = plt.figure(figsize=(10,10),dpi=100)
ax= fig.add_subplot(111)
cm_display = ConfusionMatrixDisplay(test_confusion_matrix).plot(ax=ax)

#保存模型
import pickle
with open('GaussianNB_model.pkl', 'wb') as f:
    pickle.dump(best_naive_bayes, f)