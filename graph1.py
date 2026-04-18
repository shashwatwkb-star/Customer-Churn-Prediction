import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Count churn values
churn_counts = df['Churn'].value_counts()

# Plot
plt.figure()
churn_counts.plot(kind='bar')

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (Yes/No)")
plt.ylabel("Number of Customers")

plt.show()
