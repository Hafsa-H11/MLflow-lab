
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mlflow

mlflow.set_experiment("artifact-logging")

with mlflow.start_run(run_name="artifact-demo"):

    # 1. Create an evaluation report
    report = {
        "accuracy": 0.92,
        "f1": 0.89,
        "roc_auc": 0.95
    }

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    mlflow.log_artifact(
        "eval_report.json",
        artifact_path="reports"
    )

    # 2. Create a training curve
    epochs = np.arange(1, 21)

    train_loss = 1 - 0.05 * epochs
    val_loss = 1 - 0.04 * epochs

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()

    fig.savefig("training_curve.png", dpi=100)
    mlflow.log_artifact(
        "training_curve.png",
        artifact_path="plots"
    )
    plt.close(fig)

    # 3. Create a sample CSV
    df = pd.DataFrame({
        "actual": [1, 0, 1, 1, 0],
        "predicted": [1, 0, 0, 1, 0]
    })

    df.to_csv("data_sample.csv", index=False)
    mlflow.log_artifact("data_sample.csv")

    # Log evaluation metrics
    mlflow.log_metrics(report)

    print("Artifacts logged successfully!")

