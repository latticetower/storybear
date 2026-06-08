import pandas as pd


def generate_data(csv):
    df = pd.read_csv("hf://datasets/latticetower/nrps_modules_asdb4.0/nrps_modules_info_cleaned.csv")
    df.to_csv(csv, index=None)