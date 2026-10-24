
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_breast_cancer

MODEL_NAME = "BreastCancerClassifier"

# Use the actual registered version number
MODEL_URI = f"models:/{MODEL_NAME}/1"

# Load through MLflow's generic Python interface
pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)

# Prepare 5 sample rows
data = load_breast_cancer()

sample = pd.DataFrame(
    data.data[:5],
    columns=data.feature_names
)

predictions = pyfunc_model.predict(sample)

print("Predictions:", predictions)

# Load using the sklearn-specific interface
sk_model = mlflow.sklearn.load_model(MODEL_URI)

probabilities = sk_model.predict_proba(sample)

# In this dataset: class 0 = malignant, class 1 = benign
malignant_probabilities = probabilities[:, 0]

print(
    "Malignant class probabilities:",
    malignant_probabilities.round(3)
)

