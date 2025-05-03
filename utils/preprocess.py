import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df, time_steps=3):
    df = df[['Tahun', 'Jumlah Penduduk', 'Laju Pertumbuhan Penduduk (%)']].dropna()
    df = df.sort_values(by='Tahun')

    tahun = df['Tahun'].values
    fitur = df[['Jumlah Penduduk', 'Laju Pertumbuhan Penduduk (%)']].values

    scaler = MinMaxScaler()
    fitur_scaled = scaler.fit_transform(fitur)

    def create_dataset(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps, 0])  # target: jumlah penduduk
        return np.array(X), np.array(y)

    X, y = create_dataset(fitur_scaled, time_steps)
    current_seq = fitur_scaled[-time_steps:].copy()

    return X, y, current_seq, tahun, scaler
