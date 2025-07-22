import pandas as pd
import joblib
from pathlib import Path
from feast import FeatureStore
from citas.pipelines.train_pipeline import get_feature_data

MODEL_PATH = Path("citas/models/trained_model.pkl")


class NoShowPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]

    def generate_entity_df(self, feature_values: dict, n: int = 3) -> pd.DataFrame:
        """
        Creates a synthetic entity_df using a dictionary of feature values.
        Each feature should be a scalar (it will be repeated `n` times).
        """
        entity_dict = {k: [v] * n for k, v in feature_values.items()}
        entity_dict["Codigo_Cita"] = [f"dummy_id_{i}" for i in range(n)]

        # ❗ Make event_timestamp naive to match Feast expectations
        now_utc = pd.Timestamp.now(tz="UTC")
        entity_dict["event_timestamp"] = [now_utc] * n

        return pd.DataFrame(entity_dict)

    def predict_from_features(self, feature_values: dict) -> pd.DataFrame:
        """
        Predicts no-show probabilities based on selected input feature values.
        """
        entity_df = self.generate_entity_df(feature_values)
        features_df = get_feature_data(entity_df)
        X = features_df.drop(columns=["Codigo_Cita", "event_timestamp"])
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)

        features_df["predicted_label"] = y_pred
        features_df["predicted_proba"] = y_prob

        return features_df[["Oficina_Atencion", "predicted_label", "predicted_proba"]]

    def get_top_offices_by_risk(self, feature_values: dict, top_n: int = 5) -> pd.DataFrame:
        """
        Returns the top offices with highest predicted no-show risk
        for the given input feature values.
        """
        predictions = self.predict_from_features(feature_values)
        office_risks = predictions.groupby("Oficina_Atencion")["predicted_proba"].mean()
        return office_risks.sort_values(ascending=False).head(top_n).reset_index()
    
    def recommend_offices(
        self,
        tramite,
        genero,
        tipo_doc,
        categoria,
        canal,
        entidad,
        recordatorio_correo,
        recordatorio_sms,
        dia_semana_asignacion,
        dia_semana_cita,
        X_reference,
        agrupacion='LINEA MADRID',
        demora=3,
        hora_asignacion=10,
        hora_cita=11,
        office_list=None,
        top_k=3
    ):
        """
        Recommend Top-K offices based on predicted success probability.
        If office_list is None, use all known offices from X_reference.
        """
        if office_list is None:
            office_list = X_reference['Oficina_Atencion'].unique().tolist()

        df_candidates = pd.DataFrame({
            'Genero': [genero] * len(office_list),
            'Agrupacion_Oficina_Atencion': [agrupacion] * len(office_list),
            'Oficina_Atencion': office_list,
            'Categoria': [categoria] * len(office_list),
            'Recordatorio_Correo': [recordatorio_correo] * len(office_list),
            'Recordatorio_SMS': [recordatorio_sms] * len(office_list),
            'Canal_Asignacion': [canal] * len(office_list),
            'Entidad_Atencion': [entidad] * len(office_list),
            'Tramite': [tramite] * len(office_list),
            'Tipo_Documento_Identidad': [tipo_doc] * len(office_list),
            'dia_semana_asignacion': [dia_semana_asignacion] * len(office_list),
            'dia_semana_cita': [dia_semana_cita] * len(office_list),
            'Demora': [demora] * len(office_list),
            'rango_hora_asignacion': [hora_asignacion] * len(office_list),
            'rango_hora_cita': [hora_cita] * len(office_list),
            'dia_asignacion': [None] * len(office_list),
            'dia_cita': [None] * len(office_list)
        })

        df_candidates["Tramite_Oficina"] = (
            df_candidates["Tramite"].astype(str) + "_" + df_candidates["Oficina_Atencion"].astype(str)
        )

        # Run through the trained pipeline
        probs = self.model.predict_proba(df_candidates)[:, 1]  # Prob. of no-show

        df_candidates['prob_no_atendida'] = probs

        return df_candidates[['Oficina_Atencion', 'prob_no_atendida']].sort_values(by='prob_no_atendida', ascending=False).head(top_k)
