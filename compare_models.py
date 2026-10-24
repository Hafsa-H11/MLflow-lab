import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_experiment("final-model-comparison")

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 150, "max_depth": 8}
]

for config in experiments:
    with mlflow.start_run(
        run_name=f"RF-{config['n_estimators']}-depth-{config['max_depth']}"
    ):
        mlflow.log_params(config)

        model = RandomForestClassifier(
            **config,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions)
        }

        mlflow.log_metrics(metrics)
        
        # Added skops_trusted_types here
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(config, metrics)
