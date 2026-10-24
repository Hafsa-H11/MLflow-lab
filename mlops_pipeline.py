import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EXPERIMENT = "mlops-pipeline"
MODEL_NAME = "ProductionClassifier"
ACCURACY_GATE = 0.999

mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

candidates = [
    ("RandomForest", make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))),
    ("GradientBoosting", make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))),
    ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)))
]

best_auc = 0.0
best_model_uri = None
best_model_name = ""

for name, model in candidates:
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree", "sklearn.pipeline.Pipeline"]
        )
        
        print(f"Model: {name} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        
        if acc > best_auc:
            best_auc = acc
            best_model_uri = f"runs:/{run.info.run_id}/model"
            best_model_name = name

print(f"\nBest Model: {best_model_name} with Accuracy: {best_auc:.4f}")

if best_auc >= ACCURACY_GATE:
    print(f"Accuracy Gate ({ACCURACY_GATE}) PASSED! Registering model...")
    reg_result = mlflow.register_model(best_model_uri, MODEL_NAME)
    print(f"Model successfully registered as version {reg_result.version}!")
else:
    print(f"GATE FAILED! Best accuracy ({best_auc:.4f}) is below threshold ({ACCURACY_GATE}). Skipping registration.")
