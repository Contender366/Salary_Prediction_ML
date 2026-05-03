import os
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

correlation = data.drop(columns=['salary']).corr(numeric_only=True)
high_correlation = correlation["rating"].abs()
selected_vars = high_correlation[high_correlation > 0.2].index.tolist()

# Extract predictor variables
X = data.drop(columns=['salary'])
y = data["salary"]


X_sets = {
    "X1": X.drop(columns=[selected_vars[0]], errors="ignore"),
    "X2": X.drop(columns=[selected_vars[1]], errors="ignore"),
    "X3": X.drop(columns=[selected_vars[2]], errors="ignore"),
    "X4": X.drop(columns=[selected_vars[0], selected_vars[1]], errors="ignore"),
    "X5": X.drop(columns=[selected_vars[0], selected_vars[2]], errors="ignore"),
    "X6": X.drop(columns=[selected_vars[1], selected_vars[2]], errors="ignore"),
}
mape_list = {}
for key, X in X_sets.items():
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape_value = mape(y_test, y_pred)
    mape_list[key]= mape_value

best_model = min(mape_list, key=mape_list.get)
X_train, X_test, y_train, y_test = train_test_split(X_sets[best_model], y, test_size=0.3, random_state=100)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred1 = y_pred.copy()
for i in range(len(y_pred1)):
    if y_pred1[i] < 0:
        y_pred1[i] = 0
mape_value1 = mape(y_test, y_pred1)

y_pred2 = y_pred.copy()
for i in range(len(y_pred2)):
    if y_pred2[i] < 0:
        y_pred2[i] = y_train.median()
mape_value2 = mape(y_test, y_pred2)
print(round(min(mape_value1, mape_value2), 5))



