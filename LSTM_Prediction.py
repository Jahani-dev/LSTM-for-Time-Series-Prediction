"""
Created on Thu Aug 21 22:30:27 2025

@author: Sahar Jahani
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# generate signal
n = 1000
t = np.linspace(0, 14*np.pi , n)
true_signal = 2 * np.sin(t)
noise = np.random.normal(0, 0.5, n)
noisy_signal = true_signal + noise


scaler = MinMaxScaler()
noisy_scaled = scaler.fit_transform(noisy_signal.reshape(-1, 1))
true_scaled = scaler.transform(true_signal.reshape(-1, 1))


# preparing data for LSTM
window_size  = 10


def dataprep(train, target, window_size):
    X, y = [], []
    for i in range(window_size, len(true_scaled)):
        X.append(train[i - window_size: i ])
        y.append((target[i]))
    X = np.array(X)
    y = np.array(y)
    return X, y
        

X, y = dataprep(noisy_scaled, true_scaled, window_size)

# split the data into train/test
split = int(0.7*len(noisy_signal))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# LSTM Model
model = Sequential()
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predict
y_pred = model.predict(X_test)

y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

plt.figure(figsize=(12,5))
plt.plot(range(len(y_test)), y_test, label='True Signal', linewidth=4)
plt.plot(range(len(X_test)), noisy_signal[window_size+split::], label='Noisy Signal', color='gray', linewidth=2)
plt.plot(range(len(y_pred)), y_pred, label='LSTM Prediction', linewidth = 2 , linestyle='--')
plt.title("LSTM Prediction", fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=14, fontweight='bold')
plt.ylabel("Amplitude", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")






