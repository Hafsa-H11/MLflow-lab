import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_breast_cancer

MODEL_URI = "models:/BreastCancerClassifier/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

data = load_breast_cancer()
sample = pd.DataFrame(data.data[:2], columns=data.feature_names)

predictions = model.predict(sample)
print("Predictions successfully generated via MLflow PyFunc:", predictions)
