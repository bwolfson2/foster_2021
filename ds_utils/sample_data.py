from .features_pipeline import pipeline_from_config
import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np



def get_car_data(classification=True):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"

    predictor_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin']

    mpg_df = pd.read_csv(url,
                         delim_whitespace=True,
                         header=None,
                         names=['mpg'] + predictor_columns + ['car_name']).dropna()

    if classification:
        median = mpg_df["mpg"].median()

        Y = mpg_df["mpg"].apply(lambda m: 1 if m > median else 0)
        X = mpg_df[predictor_columns]

        return X, Y
    else:

        Y = mpg_df["mpg"]
        X = mpg_df[predictor_columns]

        return X, Y

def get_dummy_data():
    np.random.seed(56)
    X, Y = make_blobs(n_samples=4000, n_features=3, cluster_std=4, centers=3, shuffle=False, random_state=42)
    colors = ["red"] * 3800 + ["blue"] * 200
    Y = np.array([0] * 3800 + [1] * 200)

    order = np.random.choice(range(4000), 4000, False)

    X = X[order]
    Y = Y[order]

    X = pd.DataFrame(X, columns=['earning', 'geographic', 'experience'])

    return X, Y

def get_mailing_data():
    mailing_url = "https://gist.githubusercontent.com/anonymous/5275f1f59be561ec9734c90d80d176b9/raw/f92227f9b8cdca188c1e89094804b8e46f14f30b/-"
    mailing_df = pd.read_csv(mailing_url)

    config = [
        {
            "field": "Income",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "Firstdate",
            "transformers": [
                {"name": "standard_numeric"}
                ]
            },
        {
            "field": "Lastdate",
            "transformers": [
                {"name": "standard_numeric"}
                ]
            },
        {
            "field": "Amount",
            "transformers": [
                {"name": "standard_numeric"},
                {
                    "name": "quantile_numeric",
                    "config": {"n_quantiles": 10}
                    }
                ]
            },
        {
            "field": "rfaa2",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "rfaf2",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "pepstrfl",
            "transformers": [
                {"name": "dummyizer"}
                ]
            },
        {
            "field": "glast",
            "transformers": [
                {"name": "standard_numeric"},
                {
                    "name": "quantile_numeric",
                    "config": {"n_quantiles": 10}
                    }
                ]
            },
        {
            "field": "gavr",
            "transformers": [
                {"name": "standard_numeric"},
                {
                    "name": "quantile_numeric",
                    "config": {"n_quantiles": 10}
                    }
                ]
            }
        ]
    pipeline = pipeline_from_config(config)
    X = pipeline.fit_transform(mailing_df)

    return X, mailing_df["class"]


def get_project_data():
    n_projects = 1000000
    np.random.seed(0)
    # Generate project attributes
    new_c  = np.array(['Old', 'New'])[np.random.binomial(1, 0.5, n_projects)]
    new_s = np.array(['Old', 'New'])[np.random.binomial(1, 0.5, n_projects)]
    large_p = np.array(['Small', 'Large'])[np.random.binomial(1, 0.5, n_projects)]
    # Assign consultant
    prob_table1 = {'Old': {'Old': {'Small': 0.1, 'Large': 0.1},
                           'New': {'Small': 0.9, 'Large': 0.1}},
                   'New': {'Old': {'Small': 0.9, 'Large': 0.1},
                           'New': {'Small': 0.9, 'Large': 0.9}}}
    probs_b = [prob_table1[new_c[i]][new_s[i]][large_p[i]] for i in range(n_projects)]
    consultant_b = np.array(['Aaron', 'Ben'])[np.random.binomial(1, probs_b)]
    # Build data frame
    data = {"Consultant": consultant_b, "Customer": new_c, "Project": large_p, "Service": new_s}
    data = pd.DataFrame(data)
    data = data.groupby(['Consultant', 'Customer', 'Service', 'Project']).apply(lambda x: x.sample(frac=.0001))
    data = data.reset_index(drop=True).sample(frac=1)
    # Obtain performance
    performance = (data.Customer == 'New')*-4 + (data.Project == 'Large')*-3 + (data.Service == 'New')*-1
    performance += -performance.min() + 1 
    data['Performance'] = performance / performance.max()   
    return data