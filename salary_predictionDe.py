import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

print("Salary Prediction using Linear Regression")

# load dataset
try:
    df = pd.read_csv('salary_dataset.csv')
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("File not found. Check if salary_dataset.csv exists.")
    print("Files present:", os.listdir('.'))
    exit()

# check dataset
print("\nShape of dataset:", df.shape)
print("\nFirst few rows:\n", df.head())

print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())

# preprocessing
print("\nPreprocessing data...")

req_cols = ['Job Title', 'Location', 'Education Level', 'YearsExperience', 'Salary']
missing = [c for c in req_cols if c not in df.columns]

if missing:
    print("Missing columns:", missing)
    exit()

# label encoding
cat_cols = ['Job Title', 'Location', 'Education Level']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"{col} encoded successfully")

# features and target
X = df[['Job Title', 'Location', 'Education Level', 'YearsExperience']]
y = df['Salary']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# model training
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Training complete.")

# model evaluation
print("\nEvaluating model...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# coefficients
print("\nFeature Coefficients:")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coef_df)

# plots
print("\nGenerating plots...")

plt.figure(figsize=(15, 10))

# Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')

# Residuals
plt.subplot(2, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='green', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Correlation
plt.subplot(2, 2, 3)
corr = df.corr()['Salary'].sort_values(ascending=False)
colors = ['green' if c > 0 else 'red' for c in corr.values]
plt.barh(range(len(corr)), corr.values, color=colors)
plt.yticks(range(len(corr)), corr.index)
plt.xlabel('Correlation')
plt.title('Feature Correlation with Salary')

# Experience vs Salary
plt.subplot(2, 2, 4)
plt.scatter(df['YearsExperience'], df['Salary'], color='purple', alpha=0.6)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')

plt.tight_layout()
plt.savefig('salary_analysis.png', dpi=300)
plt.show()

print("Plots saved as salary_analysis.png")

# sample predictions
print("\nSample predictions:")
for i in range(3):
    x_sample = X_test.iloc[i:i+1]
    actual = y_test.iloc[i]
    pred = model.predict(x_sample)[0]
    diff = abs(actual - pred)
    print(f"\nExample {i+1}")
    print(f"Actual Salary: {actual:.2f}")
    print(f"Predicted Salary: {pred:.2f}")
    print(f"Difference: {diff:.2f}")

# interpretation
print("\nModel interpretation:")
for feature, coef in zip(X.columns, model.coef_):
    if coef > 0:
        print(f"{feature} increases salary by around {coef:.2f}")
    else:
        print(f"{feature} decreases salary by around {abs(coef):.2f}")

print(f"\nBase intercept (approx starting salary): {model.intercept_:.2f}")

# summary
print("\nFINAL PROJECT SUMMARY\n")

print(f"Total Records: {df.shape[0]} | Features: {df.shape[1]}")
print("Model Used: Linear Regression")
print(f"R² Score: {r2:.4f}")
print(f"Average Error: ${mae:,.2f}")
print("Visualizations Generated: 4 Charts")
print("Output File: salary_analysis.png")