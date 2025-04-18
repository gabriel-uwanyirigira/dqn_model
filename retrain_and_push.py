import requests, os, json, numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# === Config ===
CHANNEL_ID = '2560122'
READ_API_KEY = 'T90CELFXL2HRBLJA'
NUM_RESULTS = 500
FORECAST_PATH = 'forecast.json'

# === Fetch sensor data ===
def fetch_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={NUM_RESULTS}"
    r = requests.get(url)
    feeds = r.json()['feeds']
    return np.array([
        [float(f['field1']), float(f['field2']), float(f['field3'])]
        for f in feeds if all(f[f'field{i}'] for i in range(1, 4))
    ])

# === Prepare input/output for training ===
def prepare_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled) - 6):
        X.append(scaled[i])
        y.append(scaled[i+1:i+7].mean(axis=0))  # mean of next 6
    return np.array(X), np.array(y), scaler

# === Train model ===
def train_model(X, y):
    model = Sequential([
        InputLayer(shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=8, verbose=0)
    return model

# === Predict next 2 hours ===
def forecast(model, scaler, last_input):
    scaled_input = scaler.transform([last_input])
    prediction_scaled = model.predict(scaled_input)[0]
    return scaler.inverse_transform([prediction_scaled])[0]

# === Save forecast.json ===
def save_forecast(prediction):
    with open(FORECAST_PATH, 'w') as f:
        json.dump({
            "temperature": round(prediction[0], 2),
            "humidity": round(prediction[1], 2),
            "air_quality": round(prediction[2], 2)
        }, f, indent=2)

# === Main ===
def main():
    print("📡 Fetching data...")
    data = fetch_data()
    print("🧠 Preparing training data...")
    X, y, scaler = prepare_data(data)
    print("📈 Training model...")
    model = train_model(X, y)
    print("🔮 Forecasting...")
    prediction = forecast(model, scaler, data[-1])
    print("💾 Saving forecast.json...")
    save_forecast(prediction)
    print("✅ Done!")

if __name__ == "__main__":
    main()