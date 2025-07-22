from citas.inference.predictor import NoShowPredictor
import pandas as pd

if __name__ == "__main__":
    predictor = NoShowPredictor()
    input_df = pd.read_parquet("citas/db/future_predictions_input.parquet")
    
    results = predictor.predict_from_features(input_df.to_dict(orient="records")[0])
    print(results)