import pandas as pd
import os

def load_spam_ham():
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "spam_ham.csv")
    return pd.read_csv(filename, sep=",", header=0, quotechar="'", escapechar="\\")
