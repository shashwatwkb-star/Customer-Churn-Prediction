# ------------------ 1. IMPORT LIBRARIES ------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

# ------------------ 2. LOAD DATA ------------------

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ------------------ 3. DATA PREPROCESSING ------------------

# Drop unnecessary column
df = df.drop("customerID", axis=1)

# Fix TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Drop missing values
df = df.dropna()

# Strip spaces from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ------------------ 4. FEATURE & TARGET ------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ------------------ 5. TRAIN-TEST SPLIT ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ 6. MODEL TRAINING ------------------

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ------------------ 7. MODEL EVALUATION ------------------

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Probabilities
y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Metrics
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))

# ------------------ 8. GRAPH 1: CHURN DISTRIBUTION ------------------

plt.figure()
df['Churn'].value_counts().plot(kind='bar')
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# ------------------ 9. GRAPH 2: FEATURE IMPORTANCE ------------------

importance = rf.feature_importances_

feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure()
plt.barh(feat_imp['Feature'], feat_imp['Importance'])
plt.title("Top 10 Features Affecting Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------ 10. GRAPH 3: ROC CURVE ------------------

plt.figure()
RocCurveDisplay.from_estimator(lr, X_test, y_test)
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve Comparison")
plt.show()

# ------------------ 11. GRAPH 4: CONFUSION MATRIX ------------------

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Confusion Matrix - Random Forest")
plt.show()
