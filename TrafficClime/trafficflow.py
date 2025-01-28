import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed data
train = pd.read_csv('train_preprocessed.csv')
test = pd.read_csv('test_preprocessed.csv')

# Prepare data for LSTM
def create_sequences(data, target, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(target[i + look_back])
    return np.array(X), np.array(y)

look_back = 3  # Number of previous time steps to use as input variables
features = ['Traffic Volume', 'Temperature']  # Adjust this to include any additional features

# Create sequences for training and testing
X_train, y_train = create_sequences(train[features].values, train['Traffic Volume'].values, look_back)
X_test, y_test = create_sequences(test[features].values, test['Traffic Volume'].values, look_back)

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')



# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Predict traffic volume using the test set
predicted_traffic = model.predict(X_test)

# Save the model
model.save('traffic_flow_lstm_model.h5')

# Load the model (for future use)
from tensorflow.keras.models import load_model #type:ignore
model = load_model('traffic_flow_lstm_model.h5')


# Assuming you used a MinMaxScaler for 'Traffic Volume'
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train[['Traffic Volume']])  # Fit only on the training data for inverse transformation

# Inverse transform predictions and actual values
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
predicted_traffic_inverse = scaler.inverse_transform(predicted_traffic)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_inverse, predicted_traffic_inverse)
rmse = np.sqrt(mse)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

plt.figure(figsize=(14, 5))
plt.plot(y_test_inverse, color='blue', label='Actual Traffic Volume')
plt.plot(predicted_traffic_inverse, color='red', label='Predicted Traffic Volume')
plt.title('Traffic Volume Prediction')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()