import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlflow.tracking import MlflowClient

mlflow.set_experiment("sklearn-classifier")

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# Start a fresh successful run
with mlflow.start_run(run_name="successful-registry-run") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions)
    })
    mlflow.log_params({"n_estimators": 100, "max_depth": 6})
    
    # Log model with trusted types so artifact is successfully saved
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    
    run_id = run.info.run_id
    print(f"Created successful run ID: {run_id}")

# Now register this model using the new run ID
client = MlflowClient()
MODEL_NAME = "BreastCancerClassifier"
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Registered model:", MODEL_NAME)
print("Version:", result.version)

# Add description and tags
client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description="Random Forest classifier trained on the Scikit-learn Breast Cancer dataset."
)
client.set_registered_model_tag(MODEL_NAME, "framework", "sklearn")
client.set_registered_model_tag(MODEL_NAME, "owner", "mlops-lab-student")

print("Registration complete successfully!")
