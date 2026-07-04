import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset into Pandas
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Train/test split + scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# 3. Fit logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Predictions + metrics
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Precision:", round(precision_score(y_test, y_pred),3))
print("Recall:", round(recall_score(y_test, y_pred),3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob),3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test,y_prob):.3f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel