if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

import numpy as np
import pandas as pd
import requests
from io import BytesIO
from mage_ai.data_preparation.decorators import data_loader
import joblib
import os
import sys


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

@data_loader
def load_data(*args, **kwargs):
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(url)
    return df



@transformer
def read_dataframe(df, *args, **kwargs):
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

df = load_data()
df = read_dataframe(df)
print(df.info())


dv = DictVectorizer()

train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

y = df['duration']

model = LinearRegression()
model.fit(X_train, y)
print(model.intercept_)
print(model)

# Log the model
model_filename = 'linear_regression_model.joblib'
joblib.dump(model, model_filename)
print(f"Linear regression model saved as {model_filename}")

# Get the size of the saved model
model_size_bytes = os.path.getsize(model_filename)
print(f"Model size on disk: {model_size_bytes} bytes")

# Calculate the size of the model in memory
model_size_memory = sys.getsizeof(model)
for attr in dir(model):
    if not attr.startswith("__") and not callable(getattr(model, attr)):
        model_size_memory += sys.getsizeof(getattr(model, attr))
print(f"Estimated model size in memory: {model_size_memory} bytes")