
import argparse
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=6)
parser.add_argument("--test-size", type=float, default=0.2)

args = parser.parse_args()

with mlflow.start_run():

    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=args.test_size,
        random_state=42,
        stratify=data.target
    )

    mlflow.log_params(vars(args))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    })

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Training complete!")

