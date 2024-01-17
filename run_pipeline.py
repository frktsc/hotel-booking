from pipelines.training_pipeline import Train_Pipeline
from steps.cleaning_Data import clean_Data
from steps.evaluation_Data import Evaluation_Data
from steps.Ingest_Data import ingest_df
from steps.train_model import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    training = Train_Pipeline(
        ingest_df(),
        clean_Data(),
        train_model(),
        Evaluation_Data(),
    )

    training.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

