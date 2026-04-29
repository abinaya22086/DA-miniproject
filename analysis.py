import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_data, columns=te.columns_)
    return df

def get_frequent_items(df):
    frequent_items = apriori(df, min_support=0.02, use_colnames=True)
    return frequent_items

def get_rules(frequent_items):
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    return rules