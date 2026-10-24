import mlflow

# Create or select an experiment
mlflow.set_experiment("my-first-experiment")

with mlflow.start_run(run_name="baseline"):

    # Parameters: training inputs
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Metrics: measured outputs
    for epoch in range(10):
        val_loss = 1.0 - epoch * 0.06
        train_loss = 1.0 - epoch * 0.08  # Defined train_loss

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # Tags: additional information
    mlflow.set_tag("team", "mlops-lab")
    mlflow.set_tag("data_version", "v1.0")

    print("Run ID:", mlflow.active_run().info.run_id)
