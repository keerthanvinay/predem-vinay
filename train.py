import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

# 1. Load the generated dataset
df = pd.read_csv("mess_attendance_pilot.csv")

# 2. Set features and target
feature_cols = ["baseline_4w_avg", "menu_rating", "exam_deadline", "weekend_proximity"]
target_col = "actual_attendance"

X = df[feature_cols]
y = df[target_col]

# 3. Train-test split (80% train, 20% test without shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate against test set
y_pred_test = model.predict(X_test)
baseline_test = X_test["baseline_4w_avg"]

mae_model = mean_absolute_error(y_test, y_pred_test)
rmse_model = root_mean_squared_error(y_test, y_pred_test)
mae_baseline = mean_absolute_error(y_test, baseline_test)
rmse_baseline = root_mean_squared_error(y_test, baseline_test)

print("=" * 45)
print("       PREDEM VALIDATION BENCHMARK")
print("=" * 45)
print(f"Rolling Baseline MAE : {mae_baseline:.2f} | RMSE: {rmse_baseline:.2f}")
print(f"PreDem Model MAE     : {mae_model:.2f} | RMSE: {rmse_model:.2f}")
print("-" * 45)

# 6. Save the trained model artifact
joblib.dump(model, "predem_model.joblib")
print("SUCCESS: Model saved as predem_model.joblib")