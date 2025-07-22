from feast import FeatureView, Field
from feast.types import String, Int32
from datetime import timedelta
from citas.features.feast_repo.data_sources import feature_source
from citas.features.feast_repo.entities import Codigo_Cita


citas_features = FeatureView(
    name="citas_features",
    entities=[Codigo_Cita],  
    ttl=timedelta(days=30),
    schema=[
        Field(name="Estado_Cita", dtype=String),
        Field(name="Genero", dtype=String),
        Field(name="Agrupacion_Oficina_Atencion", dtype=String),
        Field(name="Oficina_Atencion", dtype=String),
        Field(name="Categoria", dtype=String),
        Field(name="Recordatorio_Correo", dtype=String),
        Field(name="Recordatorio_SMS", dtype=String),
        Field(name="Canal_Asignacion", dtype=String),
        Field(name="Entidad_Atencion", dtype=String),
        Field(name="Tramite", dtype=String),
        Field(name="Tipo_Documento_Identidad", dtype=String),
        Field(name="dia_semana_asignacion", dtype=String),
        Field(name="dia_semana_cita", dtype=String),
        Field(name="Demora", dtype=Int32),
        Field(name="rango_hora_asignacion", dtype=Int32),
        Field(name="rango_hora_cita", dtype=Int32),
        Field(name="Tramite_Oficina", dtype=String),
    ],
    online=True,
    source=feature_source,
)
