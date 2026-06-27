import argparse, os
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = "telco_churn_preprocessing"
TARGET = "Churn"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--min_samples_split", type=int, default=5)
    a = p.parse_args()

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=a.n_estimators, max_depth=a.max_depth,
            min_samples_split=a.min_samples_split, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        mlflow.log_params(vars(a))
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)})

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, artifact_path="model",
            signature=signature, input_example=X_train.iloc[:5])
        print("[MLProject] Training selesai, model tercatat.")

if __name__ == "__main__":
    main()
