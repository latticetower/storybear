import pandas as pd


def generate_data(csv):
    raise NotImplementedError("generation method is not ready")


def download_example_data(csv, i=0):
    EXAMPLES = [
        "hf://datasets/phihung/titanic/train.csv",
        "hf://datasets/latticetower/nrps_modules_asdb4.0/nrps_modules_info_cleaned.csv"
    ]
    i = i % len(EXAMPLES)
    # df = pd.read_csv("")
    df = pd.read_csv(EXAMPLES[i])
    df.to_csv(csv, index=None)