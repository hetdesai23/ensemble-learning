# Bagging Regression

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor

# Load Dataset

X, y = load_diabetes(return_X_y=True)

print("Dataset Information")
print("Dataset Features Size:", X.shape)
print("Dataset Target Size:", y.shape)

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
train_size=0.80,
test_size=0.20,
random_state=123
)

print("\nTrain/Test Sets Sizes")
print(
"X_train:",
X_train.shape,
"X_test:",
X_test.shape
)

print(
"y_train:",
y_train.shape,
"y_test:",
y_test.shape
)

# Create Individual Regression Models

lr = LinearRegression()

dt = DecisionTreeRegressor(
random_state=42
)

knn = KNeighborsRegressor()

# Train Individual Models

lr.fit(X_train, y_train)

dt.fit(X_train, y_train)

knn.fit(X_train, y_train)

# Make Predictions

lr_prediction = lr.predict(X_test)

dt_prediction = dt.predict(X_test)

knn_prediction = knn.predict(X_test)

# Evaluate Individual Models

lr_score = r2_score(
y_test,
lr_prediction
)

dt_score = r2_score(
y_test,
dt_prediction
)

knn_score = r2_score(
y_test,
knn_prediction
)

print("\nIndividual Regressor Performance")

print(
f"Linear Regression R2 Score: "
f"{lr_score:.4f}"
)

print(
f"Decision Tree R2 Score: "
f"{dt_score:.4f}"
)

print(
f"KNN Regressor R2 Score: "
f"{knn_score:.4f}"
)

# Basic Bagging Regressor

bagging_regressor = BaggingRegressor(
estimator=DecisionTreeRegressor(),
n_estimators=100,
random_state=42,
n_jobs=-1
)

bagging_regressor.fit(
X_train,
y_train
)

# Evaluate Bagging Regressor

bagging_train_score = bagging_regressor.score(
X_train,
y_train
)

bagging_test_score = bagging_regressor.score(
X_test,
y_test
)

print("\nBagging Regressor Performance")

print(
f"Training R2 Score: "
f"{bagging_train_score:.4f}"
)

print(
f"Test R2 Score: "
f"{bagging_test_score:.4f}"
)

# GridSearchCV Hyperparameter Tuning

parameters = {
"estimator": [
DecisionTreeRegressor(),
LinearRegression(),
KNeighborsRegressor()
],

"n_estimators": [
    20,
    50,
    100
],

"max_samples": [
    0.5,
    1.0
],

"max_features": [
    0.5,
    1.0
],

"bootstrap": [
    True,
    False
],

"bootstrap_features": [
    True,
    False
]
}

print("\nRunning GridSearchCV...")

grid_search = GridSearchCV(
estimator=BaggingRegressor(
random_state=42,
n_jobs=-1
),

param_grid=parameters,

cv=3,

scoring="r2",

n_jobs=-1,

verbose=1
)

grid_search.fit(
X_train,
y_train
)

# GridSearchCV Results

best_model = grid_search.best_estimator_

grid_train_score = best_model.score(
X_train,
y_train
)

grid_test_score = best_model.score(
X_test,
y_test
)

print("\nGridSearchCV Results")

print(
f"Training R2 Score: "
f"{grid_train_score:.4f}"
)

print(
f"Test R2 Score: "
f"{grid_test_score:.4f}"
)

print(
f"Best Cross Validation R2 Score: "
f"{grid_search.best_score_:.4f}"
)

print(
"\nBest Parameters:"
)

print(
grid_search.best_params_
)

# Final Performance Comparison

results = {
"Linear Regression": lr_score,
"Decision Tree": dt_score,
"KNN Regressor": knn_score,
"Basic Bagging Regressor": bagging_test_score,
"GridSearch Best Bagging Model": grid_test_score
}

print("\nFinal Performance Comparison")

for name, score in results.items():

 print(
    f"{name}: "
    f"{score:.4f}"
)

# Find Best Model

best_method = max(
results,
key=results.get
)

print("\nBest Performing Model")

print(
f"{best_method}: "
f"{results[best_method]:.4f}"
)
