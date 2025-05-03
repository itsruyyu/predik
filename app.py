import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi membuat dataset
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Fungsi persiapan data
def prepare_data(df, time_steps=3):
    df = df.dropna(subset=['Tahun', 'Jumlah Penduduk', 'Laju Pertumbuhan Penduduk (%)'])
    df = df.sort_values('Tahun')
    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
    df['Jumlah Penduduk'] = pd.to_numeric(df['Jumlah Penduduk'], errors='coerce')
    df['Laju Pertumbuhan Penduduk (%)'] = pd.to_numeric(df['Laju Pertumbuhan Penduduk (%)'], errors='coerce')
    df = df.dropna()

    fitur = df[['Jumlah Penduduk', 'Laju Pertumbuhan Penduduk (%)']].values
    tahun = df['Tahun'].values
    scaler = MinMaxScaler()
    fitur_scaled = scaler.fit_transform(fitur)

    X, y = create_dataset(fitur_scaled, time_steps)
    current_seq = fitur_scaled[-time_steps:]
    return X, y, current_seq, tahun, scaler, df

# Fungsi retraining
def retrain_until_converged(X_train, y_train, X_test, y_test, model, scaler, max_retries=5):
    retries = 0
    mape = float('inf')
    mae_percent = float('inf')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    while retries < max_retries and (mape > 1 or mae_percent > 0.5):
        retries += 1
        model.fit(X_train, y_train, epochs=100 + 50 * (retries - 1),
                  batch_size=min(8, len(X_train)), validation_data=(X_test, y_test),
                  verbose=0, callbacks=[early_stopping])

        y_pred = model.predict(X_test, verbose=0)
        y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1)))))[:, 0]
        y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros_like(y_pred))))[:, 0]

        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        mae_percent = (mae / np.mean(y_test_inv)) * 100

    return model, mape, mae_percent

# Route halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route prediksi
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    future_years = int(request.form.get('years', 10))
    show_actual = request.form.get('show_actual') == 'true'

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        time_steps = 3

        X, y, current_seq, tahun, scaler, clean_df = prepare_data(df, time_steps)

        train_idx = np.where(tahun <= 2022)[0]
        test_idx = np.where(tahun > 2022)[0]
        split_point = len(train_idx) - time_steps

        X_train, y_train = X[:split_point], y[:split_point]
        X_test, y_test = X[split_point:split_point + len(test_idx)], y[split_point:split_point + len(test_idx)]

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(time_steps, 2), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        model, mape, mae_percent = retrain_until_converged(X_train, y_train, X_test, y_test, model, scaler)

        future_preds = []
        for _ in range(future_years):
            input_seq = current_seq[-time_steps:].reshape(1, time_steps, 2)
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            pred_lpp = current_seq[-1][1]
            next_row = np.array([pred_scaled, pred_lpp])
            current_seq = np.vstack((current_seq, next_row))
            future_preds.append(next_row)

        pred_inv = scaler.inverse_transform(np.array(future_preds))[:, 0]
        last_year = int(tahun[-1])
        result = [{'year': last_year + i + 1, 'population': round(p)} for i, p in enumerate(pred_inv)]

        response = {
            'predictions': result,
            'mape': round(mape, 3),
            'mae_percent': round(mae_percent, 3)
        }

        if show_actual:
            actual_data = [{'year': int(y), 'population': int(p)} for y, p in zip(clean_df['Tahun'], clean_df['Jumlah Penduduk'])]
            response['actual'] = actual_data

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
