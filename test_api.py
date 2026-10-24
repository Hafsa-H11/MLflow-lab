import requests
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
sample = pd.DataFrame(data.data[:2], columns=data.feature_names)

payload = {
    "dataframe_split": {
        "columns": list(sample.columns),
        "data": sample.values.tolist()
    }
}

response = requests.post("http://127.0.0.1:5002/invocations", json=payload)
print("Status:", response.status_code)
print("Response:", response.text)
