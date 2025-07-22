from citas.features.feature_engineering import generate_features_from_duckdb
from citas.features.feast_repo.entities import Codigo_Cita
from citas.features.feast_repo.feature_views import citas_features
from feast import FeatureStore
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def run_feature_pipeline():
    df = generate_features_from_duckdb()
    df.to_parquet("citas/db/features.parquet", index=False)

    store = FeatureStore(repo_path="citas/features/feast_repo")

    # Print for sanity check
    print(f"Entity type: {type(Codigo_Cita)}")
    print(f"FeatureView type: {type(citas_features)}")

    # Apply objects
    store.apply([Codigo_Cita, citas_features])

    # Materialize
    store.materialize_incremental(
      
        end_date=pd.Timestamp.now()
    )

if __name__ == "__main__":
    run_feature_pipeline()
