import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

mlflow.set_experiment("sklearn-classifier")

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="cv-bonus-run"):
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Five-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    
    # Log parameters
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 3)
    
    # Log cross-validation metrics
    mlflow.log_metrics({
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std()
    })
    
    # Log model with trusted types
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print("Model with 5-fold cross-validation logged successfully!")
