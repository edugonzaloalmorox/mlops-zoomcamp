
import pandas as pd
import duckdb
import warnings
warnings.filterwarnings("ignore")

DB_PATH = 'citas/db/citas.db'
TABLE_NAME = 'cita_previa'

# Ordered weekday list
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def generate_features_from_duckdb() -> pd.DataFrame:
    # Connect to DB
    conn = duckdb.connect(DB_PATH)
    df = conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
    conn.close()

    # Strip and engineer date/time features
    df['dia_asignacion'] = pd.to_datetime(df['Fecha_Asignacion']).dt.date
    df['dia_cita'] = pd.to_datetime(df['Fecha_Cita']).dt.date
    df['hora_asignacion'] = pd.to_datetime(df['Fecha_Asignacion']).dt.time
    df['rango_hora_asignacion'] = pd.to_datetime(df['Fecha_Asignacion']).dt.hour
    df['hora_cita'] = pd.to_datetime(df['Fecha_Cita']).dt.time
    df['rango_hora_cita'] = pd.to_datetime(df['Fecha_Cita']).dt.hour

    df['dia_semana_asignacion'] = pd.to_datetime(df['Fecha_Asignacion']).dt.day_name()
    df['dia_semana_cita'] = pd.to_datetime(df['Fecha_Cita']).dt.day_name()

    # Categorical ordering
    df['dia_semana_asignacion'] = pd.Categorical(df['dia_semana_asignacion'], categories=DAY_ORDER, ordered=True)
    df['dia_semana_cita'] = pd.Categorical(df['dia_semana_cita'], categories=DAY_ORDER, ordered=True)

    # Keep only selected features
    df_mod = df[[ "Codigo_Cita", 
        'Estado_Cita', 'Genero', 'Agrupacion_Oficina_Atencion', 
        'Oficina_Atencion', 'Categoria', 'Recordatorio_Correo', 
        'Recordatorio_SMS', 'Canal_Asignacion', 'Entidad_Atencion', 
        'Tramite', 'Tipo_Documento_Identidad', 'dia_semana_asignacion',
        'dia_semana_cita', 'Demora', 'rango_hora_asignacion', 'rango_hora_cita' 
    ]].copy()

    # Type conversions
    categorical_cols = [
        "Codigo_Cita", 'Estado_Cita', 'Genero', 'Agrupacion_Oficina_Atencion', 
        'Oficina_Atencion', 'Categoria', 'Recordatorio_Correo', 
        'Recordatorio_SMS', 'Canal_Asignacion', 'Entidad_Atencion', 
        'Tramite', 'Tipo_Documento_Identidad', 'dia_semana_asignacion',
        'dia_semana_cita'
    ]

    numerical_cols = ['Demora', 'rango_hora_asignacion', 'rango_hora_cita']

    for col in categorical_cols:
        df_mod[col] = df_mod[col].astype("str")

    for col in numerical_cols:
        df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce')

    # Add granularity variable
    df_mod["Tramite_Oficina"] = df_mod["Tramite"].astype(str) + "_" + df_mod["Oficina_Atencion"].astype(str)
    df_mod["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    
    df_mod[categorical_cols + ["Tramite_Oficina"]] = df_mod[categorical_cols + ["Tramite_Oficina"]].astype(str)

    return df_mod
