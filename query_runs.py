import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Find the experiment
exp = client.get_experiment_by_name("sklearn-classifier")

if exp is None:
    raise ValueError(
        "Experiment not found. Run train_model.py first."
    )

print("Experiment ID:", exp.experiment_id)

# Search and rank runs
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.accuracy > 0.92",
    order_by=["metrics.roc_auc DESC"],
    max_results=10
)

print(f"\nFound {len(runs)} qualifying runs")

rows = []

for run in runs:
    rows.append({
        "run_id": run.info.run_id[:8],
        "status": run.info.status,
        "n_estimators": run.data.params.get("n_estimators"),
        "max_depth": run.data.params.get("max_depth"),
        "accuracy": round(
            run.data.metrics.get("accuracy", 0), 4
        ),
        "roc_auc": round(
            run.data.metrics.get("roc_auc", 0), 4
        )
    })

if runs:
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    best = runs[0]
    print("\nBest run ID:", best.info.run_id)
    print("Best params:", best.data.params)

    # --- Bonus: Download Artifacts ---
    try:
        artifact_path = client.download_artifacts(
            best.info.run_id,
            "plots/feature_importance.png",
            "downloaded_artifacts"
        )
        print("Downloaded artifact:", artifact_path)
    except Exception as e:
        print("Note: 'plots/feature_importance.png' artifact not found in this specific run (which is normal if it wasn't logged during training).")
else:
    print("No runs matched. Check your metrics and filters.")
