import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# Load the dataset
df = pd.read_csv("DowJones.csv")

# Select the 'Value' column and convert it to a numpy array
values = df["Value"].values.reshape(-1, 1)

# Normalize the values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)


# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : (i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)


# Creating sequences
sequence_length = 4  # For example, use the past 4 weeks to predict the next week
X, y = create_sequences(scaled_values, sequence_length)

# Splitting the data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshaping for the LSTM layer
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train)
print(y_train)
# exit()

# Building the RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Making predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(
    predicted_stock_price
)  # Invert scaling

y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Invert scaling


y_sub = y_test.reshape(-1, 1)
print(predicted_stock_price - y_sub)
print(predicted_stock_price.shape)
print(y_sub.shape)
print(predicted_stock_price - y_test)
print(predicted_stock_price)
print(y_test)
