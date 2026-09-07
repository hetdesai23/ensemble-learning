# Voting Regression

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# Load Dataset

X, y = load_diabetes(return_X_y=True)

print("Dataset Shape")
print("Features:", X.shape)
print("Target:", y.shape)


# Create Individual Regression Models

lr = LinearRegression()

dt = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

et = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf")
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=5)
)


# Store Models

estimators = [
    ("Linear Regression", lr),
    ("Decision Tree", dt),
    ("Random Forest", rf),
    ("Extra Trees", et),
    ("Gradient Boosting", gb),
    ("Support Vector Regressor", svr),
    ("KNN Regressor", knn)
]


# Evaluate Individual Models

print("\nIndividual Model Performance")
print("-" * 40)

for name, model in estimators:

    scores = cross_val_score(
        model,
        X,
        y,
        scoring="r2",
        cv=5
    )

    print(f"{name}: {np.mean(scores):.4f}")


# Voting Regressor

voting_regressor = VotingRegressor(
    estimators=estimators
)

scores = cross_val_score(
    voting_regressor,
    X,
    y,
    scoring="r2",
    cv=5
)

normal_voting_score = np.mean(scores)

print("\nVoting Regressor Performance")
print("-" * 40)
print("R2 Score:", round(normal_voting_score, 4))


# Weighted Voting Regressor

weighted_voting_regressor = VotingRegressor(

    estimators=estimators,

    weights=[
        2,  # Linear Regression
        1,  # Decision Tree
        3,  # Random Forest
        3,  # Extra Trees
        3,  # Gradient Boosting
        2,  # SVR
        1   # KNN
    ]
)

weighted_scores = cross_val_score(
    weighted_voting_regressor,
    X,
    y,
    scoring="r2",
    cv=5
)

weighted_voting_score = np.mean(weighted_scores)

print("\nWeighted Voting Regressor Performance")
print("-" * 40)
print("R2 Score:", round(weighted_voting_score, 4))


# Best Models Voting Regressor

print("\nBest Models Voting Regressor")
print("-" * 40)

best_estimators = [
    ("Linear Regression", lr),
    ("Random Forest", rf),
    ("Extra Trees", et),
    ("Gradient Boosting", gb)
]

best_voting_regressor = VotingRegressor(
    estimators=best_estimators
)

best_scores = cross_val_score(
    best_voting_regressor,
    X,
    y,
    scoring="r2",
    cv=5
)

best_voting_score = np.mean(best_scores)

print("R2 Score:", round(best_voting_score, 4))


# Final Performance Comparison

print("\nFinal Performance Comparison")
print("-" * 40)

print(f"Normal Voting Regressor:   {normal_voting_score:.4f}")
print(f"Weighted Voting Regressor: {weighted_voting_score:.4f}")
print(f"Best Models Voting:        {best_voting_score:.4f}")


# Find Best Overall Approach

results = {
    "Normal Voting": normal_voting_score,
    "Weighted Voting": weighted_voting_score,
    "Best Models Voting": best_voting_score
}

best_method = max(results, key=results.get)

print("\nBest Voting Approach")
print("-" * 40)

print(f"{best_method}: {results[best_method]:.4f}")