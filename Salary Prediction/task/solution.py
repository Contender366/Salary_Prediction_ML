import os
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


# ==============================
# 1. Ensure data availability
# ==============================

DATA_DIR = '../Data'
DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
DATA_URL = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"

# Create data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Download dataset if not already present
if not os.path.exists(DATA_PATH):
    response = requests.get(DATA_URL, allow_redirects=True)
    with open(DATA_PATH, 'wb') as file:
        file.write(response.content)


# ==============================
# 2. Load and preprocess data
# ==============================

data = pd.read_csv(DATA_PATH)

# Compute correlation matrix (excluding target variable)
correlation = data.drop(columns=['salary']).corr(numeric_only=True)

# Select features highly correlated with 'rating'
high_correlation = correlation["rating"].abs()
selected_vars = high_correlation[high_correlation > 0.2].index.tolist()

# Define predictors (X) and target (y)
X = data.drop(columns=['salary'])
y = data['salary']


# ==============================
# 3. Create feature subsets
# ==============================

# Different feature combinations by dropping selected variables
X_sets = {
    "X1": X.drop(columns=[selected_vars[0]], errors="ignore"),
    "X2": X.drop(columns=[selected_vars[1]], errors="ignore"),
    "X3": X.drop(columns=[selected_vars[2]], errors="ignore"),
    "X4": X.drop(columns=[selected_vars[0], selected_vars[1]], errors="ignore"),
    "X5": X.drop(columns=[selected_vars[0], selected_vars[2]], errors="ignore"),
    "X6": X.drop(columns=[selected_vars[1], selected_vars[2]], errors="ignore"),
}


# ==============================
# 4. Evaluate models (MAPE)
# ==============================

mape_scores = {}

for key, X_subset in X_sets.items():
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.3, random_state=100
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mape_scores[key] = mape(y_test, y_pred)


# ==============================
# 5. Select best feature set
# ==============================

best_model_key = min(mape_scores, key=mape_scores.get)

# Re-train model using the best feature set
X_train, X_test, y_train, y_test = train_test_split(
    X_sets[best_model_key], y, test_size=0.3, random_state=100
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ==============================
# 6. Post-process predictions
# ==============================

# Strategy 1: Replace negative predictions with 0
y_pred_zero = y_pred.copy()
for i in range(len(y_pred_zero)):
    if y_pred_zero[i] < 0:
        y_pred_zero[i] = 0

mape_zero = mape(y_test, y_pred_zero)


# Strategy 2: Replace negative predictions with median of training target
y_pred_median = y_pred.copy()
median_value = y_train.median()

for i in range(len(y_pred_median)):
    if y_pred_median[i] < 0:
        y_pred_median[i] = median_value

mape_median = mape(y_test, y_pred_median)


# ==============================
# 7. Final result
# ==============================

# Output the best (lowest) MAPE after post-processing
final_score = min(mape_zero, mape_median)
print(round(final_score, 5))
