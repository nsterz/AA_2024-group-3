import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(data, start_date, end_date):
    # Ensure the index is a DatetimeIndex
    data = data.set_index('ds') if 'ds' in data.columns else data
    data.index = pd.to_datetime(data.index)  # Convert index to DatetimeIndex if not already
    
    # Slice the subset
    subset = data.loc[start_date:end_date]
    
    
    plt.figure(figsize=(12, 9))
    
    plt.subplot(411)
    plt.plot(subset.yhat)
    
    plt.tight_layout()
    plt.show()



def plot_subseries(data, start_date, end_date):
    # Ensure the index is a DatetimeIndex
    data = data.set_index('timestamp') if 'timestamp' in data.columns else data
    data.index = pd.to_datetime(data.index)  # Convert index to DatetimeIndex if not already
    
    # Slice the subset
    subset = data.loc[start_date:end_date]
    
    
    plt.figure(figsize=(12, 9))
    
    plt.subplot(411)
    plt.plot(subset.utilization_perc)
    
    plt.tight_layout()
    plt.show()


def is_in_gap(timestamp, gaps):
    timestamp = pd.Timestamp(timestamp)
    return int(any(pd.Timestamp(gap['start']) <= timestamp <= pd.Timestamp(gap['end']) for gap in gaps))





# Function to find gaps of at least 49 consecutive timestamps with 0 utilization
# more than 2 entire days (more than a weekend)
def find_gaps(df):
    # Create a mask for zero utilization
    zero_mask = df['utilization_absolute'] == 0
    
    # Initialize variables
    gaps = []
    start_idx = None
    
    # Iterate through the data
    for i in range(len(df)):
        if zero_mask.iloc[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                # Check if gap is at least 49 timestamps
                if i - start_idx >= 49:
                    gaps.append({
                        'start': df['timestamp'].iloc[start_idx],
                        'end': df['timestamp'].iloc[i-1],
                        'duration_hours': i - start_idx
                    })
                start_idx = None
    
    # Check for gap at the end of the dataset
    if start_idx is not None and len(df) - start_idx >= 49:
        gaps.append({
            'start': df['timestamp'].iloc[start_idx],
            'end': df['timestamp'].iloc[-1],
            'duration_hours': len(df) - start_idx
        })
    
    return gaps

def create_sequences_lstm(data, sequence_length, target_column):
    X, y = [], []
    target_index = data.columns.get_loc(target_column)  # Get the index of the target column
    for i in range(len(data) - sequence_length):
        # Use all columns except the target column for X
        X.append(data.iloc[i:i+sequence_length].drop(columns=[target_column]).values)
        # Use only the target column for y
        y.append(data.iloc[i+sequence_length][target_column])
    return np.array(X), np.array(y)