# Bagging Ensemble Learning Techniques

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

# Create Dataset

X, y = make_classification(
n_samples=10000,
n_features=10,
n_informative=3,
random_state=42
)

print("Dataset Information")
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# Decision Tree Baseline

decision_tree = DecisionTreeClassifier(
random_state=42
)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

decision_tree_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nDecision Tree Performance")
print(f"Accuracy: {decision_tree_accuracy:.4f}")

# Basic Bagging Classifier

bagging_classifier = BaggingClassifier(
estimator=DecisionTreeClassifier(),
n_estimators=500,
max_samples=0.5,
bootstrap=True,
random_state=42,
n_jobs=-1
)

bagging_classifier.fit(X_train, y_train)

y_pred = bagging_classifier.predict(X_test)

bagging_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nBagging Classifier Performance")
print(f"Accuracy: {bagging_accuracy:.4f}")

# Bagging with Support Vector Classifier

svc_bagging = BaggingClassifier(
estimator=SVC(),
n_estimators=100,
max_samples=0.5,
bootstrap=True,
random_state=42,
n_jobs=-1
)

svc_bagging.fit(X_train, y_train)

y_pred = svc_bagging.predict(X_test)

svc_bagging_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nBagging with SVC Performance")
print(f"Accuracy: {svc_bagging_accuracy:.4f}")

# Pasting

pasting_classifier = BaggingClassifier(
estimator=DecisionTreeClassifier(),
n_estimators=500,
max_samples=0.5,
bootstrap=False,
random_state=42,
n_jobs=-1
)

pasting_classifier.fit(X_train, y_train)

y_pred = pasting_classifier.predict(X_test)

pasting_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nPasting Classifier Performance")
print(f"Accuracy: {pasting_accuracy:.4f}")

# Random Subspaces

random_subspaces = BaggingClassifier(
estimator=DecisionTreeClassifier(),
n_estimators=500,
max_samples=1.0,
bootstrap=False,
max_features=0.5,
bootstrap_features=True,
random_state=42,
n_jobs=-1
)

random_subspaces.fit(X_train, y_train)

y_pred = random_subspaces.predict(X_test)

random_subspaces_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nRandom Subspaces Performance")
print(f"Accuracy: {random_subspaces_accuracy:.4f}")

# Random Patches

random_patches = BaggingClassifier(
estimator=DecisionTreeClassifier(),
n_estimators=500,
max_samples=0.25,
bootstrap=True,
max_features=0.5,
bootstrap_features=True,
random_state=42,
n_jobs=-1
)

random_patches.fit(X_train, y_train)

y_pred = random_patches.predict(X_test)

random_patches_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nRandom Patches Performance")
print(f"Accuracy: {random_patches_accuracy:.4f}")

# OOB Score

oob_bagging = BaggingClassifier(
estimator=DecisionTreeClassifier(),
n_estimators=500,
max_samples=0.25,
bootstrap=True,
oob_score=True,
random_state=42,
n_jobs=-1
)

oob_bagging.fit(X_train, y_train)

y_pred = oob_bagging.predict(X_test)

oob_test_accuracy = accuracy_score(
y_test,
y_pred
)

print("\nOut-of-Bag Performance")
print(f"OOB Score: {oob_bagging.oob_score_:.4f}")
print(f"Test Accuracy: {oob_test_accuracy:.4f}")

# GridSearchCV for Hyperparameter Tuning

parameters = {
"n_estimators": [50, 100, 500],
"max_samples": [0.1, 0.4, 0.7, 1.0],
"bootstrap": [True, False],
"max_features": [0.1, 0.4, 0.7, 1.0]
}

grid_search = GridSearchCV(
estimator=BaggingClassifier(
estimator=DecisionTreeClassifier(),
random_state=42,
n_jobs=-1
),
param_grid=parameters,
cv=5,
scoring="accuracy",
n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nGridSearchCV Results")
print("Best Parameters:")
print(grid_search.best_params_)

print(f"Best Cross Validation Score: {grid_search.best_score_:.4f}")

# Final Performance Comparison

results = {
"Decision Tree": decision_tree_accuracy,
"Basic Bagging": bagging_accuracy,
"Bagging with SVC": svc_bagging_accuracy,
"Pasting": pasting_accuracy,
"Random Subspaces": random_subspaces_accuracy,
"Random Patches": random_patches_accuracy,
"OOB Bagging Test Accuracy": oob_test_accuracy
}

print("\nFinal Performance Comparison")

for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Find Best Approach

best_method = max(
results,
key=results.get
)

print("\nBest Performing Approach")
print(f"{best_method}: {results[best_method]:.4f}")
