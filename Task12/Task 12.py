import t11
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# 1. Split Data (80% Train, 20% Test)
X = t11.df[t11.num_cols]
y = t11.df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50, stratify=y)

# 2. Train Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=50)
model.fit(X_train, y_train)

# 3. Predict & Calculate Metrics
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 4. Plot Confusion Matrix & Tree
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].grid(False)

# Decision Tree Diagram
plot_tree(model, feature_names=t11.num_cols, class_names=model.classes_, filled=True, ax=axes[1])
axes[1].set_title('Decision Tree Structure')

plt.tight_layout()
plt.show()