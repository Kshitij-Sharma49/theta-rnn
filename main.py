import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def predict_value_after_months(n_months_ahead):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('7d99.csv')

    # Create a new DataFrame considering only 'timestamp' and 'valueTfuel'
    considered_df = df[['timestamp', 'valueTfuel']].copy()

    # Remove rows with null values in the considered DataFrame
    considered_df.dropna(inplace=True)

    # Perform log transformation on 'valueTfuel'
    considered_df['log_valueTfuel'] = np.log1p(considered_df['valueTfuel'])

    # Remove rows with log_Tfuel = -inf
    considered_df = considered_df[considered_df['log_valueTfuel'] != -np.inf]

    # Create a new DataFrame with 'timestamp' and 'log_valueTfuel'
    final_df = considered_df[['timestamp', 'log_valueTfuel']].copy()

    # Convert the 'timestamp' column to datetime type
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])

    # Sort the DataFrame by 'timestamp'
    final_df.sort_values('timestamp', inplace=True)

    # Set 'timestamp' as the index of the DataFrame
    final_df.set_index('timestamp', inplace=True)

    # Define a function to create a supervised learning dataset
    def create_dataset(data, n_features):
        X, y = [], []
        for i in range(len(data)-n_features-1):
            X.append(data[i:(i+n_features), 0])
            y.append(data[i + n_features, 0])
        return np.array(X), np.array(y)

    # Define the number of previous time steps to consider
    n_steps = 30

    # Create the supervised learning dataset
    X, y = create_dataset(final_df.values, n_steps)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(type(final_df))

    # # Reshape the input data for LSTM
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # # Define the RNN model
    # model = Sequential()
    # model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
    # # model.add(LSTM(50))
    # model.add(LSTM(50))
    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mse')

    # # Train the model
    # model.fit(X_train, y_train, epochs=57, batch_size=32)

    # # Predict the log_valueTfuel after n_months_ahead
    # last_month_data = final_df.tail(n_steps).values
    # last_month_data = np.reshape(last_month_data, (1, n_steps, 1))

    # for _ in range(n_months_ahead):
    #     predicted_value = model.predict(last_month_data)
    #     last_month_data = np.append(last_month_data, np.expand_dims(predicted_value, axis=1), axis=1)
    #     last_month_data = last_month_data[:, -n_steps:, :]

    # # Take anti-log to get the predicted value
    # predicted_value = np.expm1(predicted_value)

    return 0#predicted_value[0][0]

# Example hai lomdi 
# data_csv = 'data.csv'
n_months_ahead = 3
predicted_value = predict_value_after_months(n_months_ahead)
print("Predicted Value after", n_months_ahead, "month(s):", predicted_value)
