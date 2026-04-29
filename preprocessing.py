import pandas as pd

def load_data():
    df = pd.read_csv("dataset.csv", header=None)
    return df

def create_transactions(df):
    transactions = df.apply(
        lambda row: [str(item) for item in row if str(item) != 'nan'],
        axis=1
    ).tolist()
    return transactions