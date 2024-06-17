import pickle
import pandas as pd
import numpy as np
import pyarrow
import argparse
import os

# Function to read data
def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# Function to process data and predict
def process_data(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(filename)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    std_value = np.std(y_pred)
    print(f'Standard deviation of predicted duration: {std_value:.2f} minutes')

    # Create ride_id
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    output_file = f'predicted_durations_{year:04d}_{month:02d}.parquet'

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Calculate mean predicted duration
    mean_predicted_duration = np.mean(y_pred)
    print(f'Mean predicted duration for {year}-{month}: {mean_predicted_duration:.2f} minutes')

    # Output file size
    file_size = os.path.getsize(output_file)
    print(f'The size of the file {output_file} is: {file_size / (1024 * 1024):.2f} megabytes')

# Parse command-line arguments
def main():
    parser = argparse.ArgumentParser(description='Predict taxi ride durations.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data.')
    parser.add_argument('--month', type=int, required=True, help='Month of the data.')
    args = parser.parse_args()

    with open('model.bin', 'rb') as f_in:
        global dv, model
        dv, model = pickle.load(f_in)

    global categorical
    categorical = ['PULocationID', 'DOLocationID']

    process_data(args.year, args.month)

if __name__ == "__main__":
    main()
