import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Enable XGBoost autologging
mlflow.xgboost.autolog()

mlflow.set_experiment("xgboost-autolog-demo")

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="xgboost-autolog-run"):
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("XGBoost model trained and autologged successfully!")
