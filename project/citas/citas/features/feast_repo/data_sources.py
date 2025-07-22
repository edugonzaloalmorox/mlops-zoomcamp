from feast import FileSource
from pathlib import Path


FEATURES_PATH = Path(__file__).resolve().parents[2] / "db" / "features.parquet"

feature_source = FileSource(
    name="citas_features_source", 
    path=str(FEATURES_PATH),      
    event_timestamp_column="event_timestamp",
)