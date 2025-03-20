import tensorflow as tf
import pandas as pd
import numpy as np

# Load Training Data
df = pd.read_csv("data/train.csv")

# Prepare Training Data
def generate_train_data(data, window=38):
    train, train_y = [], []
    for cfips in data["cfips"].unique():
        data_x = data[data["cfips"] == cfips].set_index("first_day_of_month")
        train.append(data_x["microbusiness_density"].values[:38])
        train_y.append(data_x["microbusiness_density"].values[38])
    
    return np.array(train), np.array(train_y)

train, y_train = generate_train_data(df)

# Define LSTM Model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(38,1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

# Train Model
model = create_model()
history = model.fit(train, y_train, epochs=100)

# Save Model
model.save("saved_model/lstm_model.h5")
print("Model saved successfully!")
