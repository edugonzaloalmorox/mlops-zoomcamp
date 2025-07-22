from fastapi import FastAPI
from pydantic import BaseModel
from citas.inference.predictor import NoShowPredictor
import pandas as pd

app = FastAPI()
predictor = NoShowPredictor()
X_reference = pd.read_parquet("citas/db/features.parquet")  # this should be loaded once

class RecommendationRequest(BaseModel):
    tramite: str
    genero: int
    tipo_doc: str
    categoria: str
    canal: str
    entidad: str
    recordatorio_correo: int
    recordatorio_sms: int
    dia_semana_asignacion: str
    dia_semana_cita: str
    top_k: int = 5

@app.post("/recommend_offices")
def recommend(req: RecommendationRequest):
    top_offices = predictor.recommend_offices(
        tramite=req.tramite,
        genero=req.genero,
        tipo_doc=req.tipo_doc,
        categoria=req.categoria,
        canal=req.canal,
        entidad=req.entidad,
        recordatorio_correo=req.recordatorio_correo,
        recordatorio_sms=req.recordatorio_sms,
        dia_semana_asignacion=req.dia_semana_asignacion,
        dia_semana_cita=req.dia_semana_cita,
        X_reference=X_reference,
        top_k=req.top_k
    )
    return top_offices.to_dict(orient="records")
