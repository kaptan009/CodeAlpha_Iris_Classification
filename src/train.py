import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocess import scale_data

data = pd.read_csv("data/Iris.csv")
data = data.drop("Id", axis=1)

X = data.drop("Species", axis=1)
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_test, scaler = scale_data(X_train, X_test)

models = {
    "lr": LogisticRegression(max_iter=200),
    "rf": RandomForestClassifier(),
    "svm": SVC()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(name, acc)
    if acc > best_score:
        best_score = acc
        best_model = model

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Done")
