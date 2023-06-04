from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# 加载手写数字数据集
digits = load_digits()

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 定义 SVM 模型和参数网格
svm = SVC()
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['linear', 'rbf']}

# 使用 GridSearchCV 进行参数寻优
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳准确率
print("Best SVM parameters:", svm_grid_search.best_params_)
print("Best accuracy:", svm_grid_search.best_score_)

# 使用最佳参数创建 SVM 模型
best_svm = SVC(C=svm_grid_search.best_params_['C'], gamma=svm_grid_search.best_params_['gamma'], kernel=svm_grid_search.best_params_['kernel'])
best_svm.fit(X_train, y_train)

# 使用 K-Fold 进行交叉验证并计算平均准确率
scores = cross_val_score(best_svm, digits.data, digits.target, cv=5)
print("Cross-validation accuracy:", scores.mean())

# 在测试集上计算准确率、混淆矩阵和分类器报告
test_pred = best_svm.predict(X_test)
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

# 使用 scikit-learn 提供的混淆矩阵可视化工具
fig = plt.figure(figsize=(10,10),dpi=100)
ax= fig.add_subplot(111)
cm_display = ConfusionMatrixDisplay(test_confusion_matrix).plot(ax=ax)

#保存模型
import pickle
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(best_svm, f)