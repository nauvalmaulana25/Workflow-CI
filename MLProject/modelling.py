import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os


DATA_PATH = 'preprocessing_Train.csv'
MODEL_PATH = 'modeling_model.pkl'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Reached.on.Time_Y_N'])
    y = df['Reached.on.Time_Y_N']
    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_uri', type=str, default='http://127.0.0.1:5001')
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.sklearn.autolog()

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    with mlflow.start_run(run_name='RandomForest_Baseline') as run:
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        print('Test Metrics:')
        for name, value in metrics.items():
            print(f'  {name}: {value:.4f}')

        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path='models')
        print(f'Model saved to {MODEL_PATH}')
        print(f'MLflow Run ID: {run.info.run_id}')


if __name__ == '__main__':
    main()
