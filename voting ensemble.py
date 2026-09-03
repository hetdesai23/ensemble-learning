import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import make_classification

df = pd.read_csv("Iris.csv")

print(df.head())

df = df.iloc[:, 1:]

encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

print(df.head())

# sns.pairplot(df, hue="Species")
# plt.show()

new_df = df[df["Species"] != 0][
    ["SepalLengthCm", "SepalWidthCm", "Species"]
]

print(new_df.head())
print(new_df.shape)

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(random_state=42)
clf3 = KNeighborsClassifier()

estimators = [
    ("lr", clf1),
    ("rf", clf2),
    ("knn", clf3)
]

for name, estimator in estimators:
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=10,
        scoring="accuracy"
    )
    print(name, np.round(np.mean(scores), 2))

vc_hard = VotingClassifier(
    estimators=estimators,
    voting="hard"
)

scores = cross_val_score(
    vc_hard,
    X,
    y,
    cv=10,
    scoring="accuracy"
)

print(np.round(np.mean(scores), 2))

vc_soft = VotingClassifier(
    estimators=estimators,
    voting="soft"
)

scores = cross_val_score(
    vc_soft,
    X,
    y,
    cv=10,
    scoring="accuracy"
)

print(np.round(np.mean(scores), 2))

for i in range(1, 4):
    for j in range(1, 4):
        for k in range(1, 4):

            vc = VotingClassifier(
                estimators=estimators,
                voting="soft",
                weights=[i, j, k]
            )

            scores = cross_val_score(
                vc,
                X,
                y,
                cv=10,
                scoring="accuracy"
            )

            print(
                [i, j, k],
                np.round(np.mean(scores), 2)
            )

X_svm, y_svm = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=2
)

svm1 = SVC(probability=True, kernel="poly", degree=1)
svm2 = SVC(probability=True, kernel="poly", degree=2)
svm3 = SVC(probability=True, kernel="poly", degree=3)
svm4 = SVC(probability=True, kernel="poly", degree=4)
svm5 = SVC(probability=True, kernel="poly", degree=5)

svm_estimators = [
    ("svm1", svm1),
    ("svm2", svm2),
    ("svm3", svm3),
    ("svm4", svm4),
    ("svm5", svm5)
]

for name, estimator in svm_estimators:
    scores = cross_val_score(
        estimator,
        X_svm,
        y_svm,
        cv=10,
        scoring="accuracy"
    )
    print(name, np.round(np.mean(scores), 2))

svm_voting = VotingClassifier(
    estimators=svm_estimators,
    voting="soft"
)

scores = cross_val_score(
    svm_voting,
    X_svm,
    y_svm,
    cv=10,
    scoring="accuracy"
)

print(np.round(np.mean(scores), 2))