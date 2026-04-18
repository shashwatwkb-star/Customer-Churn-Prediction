import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ------------------ DATA CLEANING ------------------

# Drop unnecessary column
df = df.drop("customerID", axis=1)

# Fix TotalCharges (convert to numeric)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Drop missing values
df = df.dropna()

# Remove extra spaces from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert target variable (VERY IMPORTANT)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Encode remaining categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ------------------ MODEL TRAINING ------------------

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ------------------ FEATURE IMPORTANCE ------------------

# Get feature importance
importance = model.feature_importances_

# Create dataframe
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by="Importance", ascending=False).head(10)

# ------------------ PLOTTING ------------------

plt.figure()
plt.barh(feat_imp['Feature'], feat_imp['Importance'])

plt.title("Top 10 Features Affecting Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

