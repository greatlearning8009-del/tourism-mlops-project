
import pandas as pd
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-model-training")

# --------------------------------------------------
# Hugging Face
# --------------------------------------------------

api = HfApi()

# --------------------------------------------------
# Load Train/Test Data
# --------------------------------------------------

Xtrain = pd.read_csv(
    "hf://datasets/SRGL/tourism/Xtrain.csv"
)

Xtest = pd.read_csv(
    "hf://datasets/SRGL/tourism/Xtest.csv"
)

ytrain = pd.read_csv(
    "hf://datasets/SRGL/tourism/ytrain.csv"
).squeeze()

ytest = pd.read_csv(
    "hf://datasets/SRGL/tourism/ytest.csv"
).squeeze()

print("Training data loaded successfully.")

# --------------------------------------------------
# Handle Class Imbalance
# --------------------------------------------------

class_weight = (
    ytrain.value_counts()[0]
    / ytrain.value_counts()[1]
)

# --------------------------------------------------
# XGBoost Model
# --------------------------------------------------

xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=class_weight,
    eval_metric="logloss"
)

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# --------------------------------------------------
# MLflow Tracking
# --------------------------------------------------

with mlflow.start_run():

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(results["params"][i])

            mlflow.log_metric(
                "mean_test_score",
                results["mean_test_score"][i]
            )

            mlflow.log_metric(
                "std_test_score",
                results["std_test_score"][i]
            )

    mlflow.log_params(
        grid_search.best_params_
    )

    best_model = grid_search.best_estimator_

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"]
    })

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    model_path = "tourism_model_v1.joblib"

    joblib.dump(
        best_model,
        model_path
    )

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(
        f"Model saved locally: {model_path}"
    )

    # --------------------------------------------------
    # Upload to Hugging Face Model Hub
    # --------------------------------------------------

    repo_id = "SRGL/tourism-model"

    try:
        api.repo_info(
            repo_id=repo_id,
            repo_type="model"
        )

        print(
            f"Model repo '{repo_id}' already exists."
        )

    except RepositoryNotFoundError:

        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False
        )

        print(
            f"Created model repo '{repo_id}'."
        )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model"
    )

    print(
        "Model uploaded to Hugging Face."
    )

print("Training completed.")
print("Best Parameters:", grid_search.best_params_)
