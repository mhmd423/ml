from pathlib import Path
import pandas as pd


def load_data(problem_number=1, subset_number=1, dataset_type="train", number_of_features=2):
    #parent of all the data 
    DATA_DIR = Path("data") / "cs229-2018-autumn" / "problem-sets" / f"PS{problem_number}" / "data"
    #file path of the dataset 
    file_path = DATA_DIR / f"ds{subset_number}_{dataset_type}.csv"

    df = pd.read_csv(file_path)

    feature_cols = [f"x_{i}" for i in range(1, number_of_features + 1)]

    X = df[feature_cols].to_numpy()
    y = df["y"].to_numpy()
    return X, y
