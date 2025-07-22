import pandas as pd
import duckdb
import joblib
import subprocess
import os
import warnings
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
from feast import FeatureStore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score, accuracy_score, fbeta_score, precision_recall_curve, precision_score
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np

warnings.filterwarnings("ignore")


def kill_duckdb_processes():
    try:
        result = subprocess.run(["ps", "aux"], stdout=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if "duckdb" in line.lower() and "python" in line.lower():
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    pid = int(parts[1])
                    if pid != os.getpid():
                        os.kill(pid, 9)
                        print(f"🔪 Killed DuckDB-related Python process: {pid}")
    except Exception as e:
        print(f"⚠️ Failed to kill DuckDB processes: {e}")


kill_duckdb_processes()


def split_entity_df_by_date(db_path: str, months_back: int = 2):
    con = duckdb.connect(db_path)
    query = "SELECT Codigo_Cita, Fecha_Cita FROM cita_previa"
    df = con.execute(query).fetchdf()
    con.close()

    df["Fecha_Cita"] = pd.to_datetime(df["Fecha_Cita"]).dt.tz_localize("UTC")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=months_back)

    df_train = df[df["Fecha_Cita"] < cutoff].copy()
    df_valid = df[df["Fecha_Cita"] >= cutoff].copy()

    now_utc = pd.Timestamp.now(tz="UTC")
    df_train["event_timestamp"] = now_utc
    df_valid["event_timestamp"] = now_utc

    return df_train[["Codigo_Cita", "event_timestamp"]], df_valid[["Codigo_Cita", "event_timestamp"]]


def get_feature_data(entity_df: pd.DataFrame, store_path: str = "citas/features/feast_repo") -> pd.DataFrame:
    store = FeatureStore(repo_path=store_path)
    feature_names = [
        "Estado_Cita", "Genero", "Agrupacion_Oficina_Atencion", "Oficina_Atencion",
        "Categoria", "Recordatorio_Correo", "Recordatorio_SMS", "Canal_Asignacion",
        "Entidad_Atencion", "Tramite", "Tipo_Documento_Identidad",
        "dia_semana_asignacion", "dia_semana_cita", "Demora",
        "rango_hora_asignacion", "rango_hora_cita", "Tramite_Oficina"
    ]
    features = [f"citas_features:{f}" for f in feature_names]
    
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
    if  entity_df["event_timestamp"].dt.tz is None:
            entity_df["event_timestamp"] = entity_df["event_timestamp"].dt.tz_localize("UTC")
    else:
        entity_df["event_timestamp"] = entity_df["event_timestamp"].dt.tz_convert("UTC")

    return store.get_historical_features(entity_df=entity_df, features=features).to_df()


def create_target_from_db(codigo_citas: pd.Series, db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path)
    citas = ",".join(f"'{c}'" for c in codigo_citas.tolist())
    query = f"""
    SELECT Codigo_Cita, Estado_Cita
    FROM cita_previa
    WHERE Codigo_Cita IN ({citas})
    """
    df = con.execute(query).fetchdf()
    con.close()
    df["target"] = (df["Estado_Cita"] != "Atendida").astype(int)
    return df[["Codigo_Cita", "target"]]


def optimize_model(X, y):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    def objective(space):
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=int(space['n_estimators']),
                max_depth=int(space['max_depth']),
                learning_rate=space['learning_rate'],
                subsample=space['subsample'],
                colsample_bytree=space['colsample_bytree'],
                use_label_encoder=False,
                eval_metric="logloss"
            ))
        ])

        model.fit(X, y)
        preds = model.predict(X)
        f1 = f1_score(y, preds)
        return {'loss': -f1, 'status': STATUS_OK, 'model': model}

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'subsample': hp.uniform('subsample', 0.7, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
    best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
    return best_trial['model']


def find_best_threshold(model, X_val, y_val, beta=2):
    probas = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probas)
    scores = [fbeta_score(y_val, probas >= t, beta=beta) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]


def train_model(X: pd.DataFrame, y: pd.Series):
    model = optimize_model(X, y)
    return model

def evaluate(model, df_valid: pd.DataFrame, db_path: str, min_recall: float = 0.8):
    features = get_feature_data(df_valid)
    y_true = create_target_from_db(features["Codigo_Cita"], db_path)

    data = features.merge(y_true, on="Codigo_Cita")
    X_val = data.drop(columns=["target", "event_timestamp", "Codigo_Cita"])
    y_val = data["target"]

    y_prob = model.predict_proba(X_val)[:, 1]

    # 🔹 Find best threshold using F2 score
    threshold, _ = find_best_threshold(model, X_val, y_val, beta=2)

    # 🔹 Predict using threshold
    y_pred = (y_prob >= threshold).astype(int)

    # 🔹 Compute and return metrics
    metrics = {
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "f2_score": fbeta_score(y_val, y_pred, beta=2),
        "roc_auc": roc_auc_score(y_val, y_prob)
    }

    print(f"✅ Evaluation at best F2 threshold = {threshold:.4f}")
    for k, v in metrics.items():
        print(f"  • {k:<10}: {v:.4f}")
    print("  • Confusion matrix:\n", confusion_matrix(y_val, y_pred))

    return threshold, metrics


def main():
    db_path = "citas/db/citas.db"
    model_path = "citas/models/trained_model.pkl"

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("no_show_prediction")

    with mlflow.start_run() as run:
        df_train, df_valid = split_entity_df_by_date(db_path)

        feat_train = get_feature_data(df_train)
        target_train = create_target_from_db(feat_train["Codigo_Cita"], db_path)

        data_train = feat_train.merge(target_train, on="Codigo_Cita")
        X_train = data_train.drop(columns=["target", "event_timestamp", "Codigo_Cita"])
        y_train = data_train["target"]

        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("num_train_samples", len(X_train))

        model = train_model(X_train, y_train)

        # 🔹 Get threshold and evaluation metrics
        threshold, eval_metrics = evaluate(model, df_valid, db_path)

        # 🔹 Log threshold and metrics
        mlflow.log_param("threshold", threshold)
        for metric_name, value in eval_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        # 🔹 Save and log the full bundle
        joblib.dump({"model": model, "threshold": threshold}, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_bundle")

        # 🔹 Register model
        mlflow.sklearn.log_model(model, artifact_path="sklearn_model", registered_model_name="NoShowPredictor")

        # 🔹 Log validation IDs
        df_valid.to_parquet("citas/db/validation_ids.parquet", index=False)
        mlflow.log_artifact("citas/db/validation_ids.parquet", artifact_path="data")

        print("✅ Training run logged in MLflow")



if __name__ == "__main__":
    main()
