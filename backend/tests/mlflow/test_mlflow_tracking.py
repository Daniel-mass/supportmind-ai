"""
Basic MLflow tracking integration test for SupportMind AI.

Purpose:
    Verify that SupportMind can connect to the local MLflow Tracking Server,
    create/use a dedicated experiment, record a run, and log basic metadata.

This test does not integrate with the Agentic Layer yet.
"""

import mlflow


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "SupportMind-Agent-Evaluation"


def test_mlflow_tracking_connection() -> None:
    """Verify basic MLflow experiment and run tracking."""

    # Connect to the local MLflow Tracking Server.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create the SupportMind experiment if it does not already exist.
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start one test run.
    with mlflow.start_run(run_name="initial_tracking_test"):

        # Record the system configuration.
        mlflow.log_params(
            {
                "project": "supportmind-ai",
                "layer": "agentic",
                "llm_model": "qwen3:8b",
                "framework": "autogen",
            }
        )

        # Record a simple workflow health metric.
        mlflow.log_metric("workflow_success", 1.0)

        # Add searchable metadata.
        mlflow.set_tags(
            {
                "project": "supportmind-ai",
                "layer": "agentic",
                "test_type": "mlflow_connectivity",
                "environment": "local",
            }
        )

        run_id = mlflow.active_run().info.run_id

    print(f"\nMLflow tracking test completed.")
    print(f"Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment   : {EXPERIMENT_NAME}")
    print(f"Run ID       : {run_id}")