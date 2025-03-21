import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from tensorflow.keras.losses import MeanAbsoluteError  # type: ignore

# Get file path from command-line argument or use default
file_path = sys.argv[1] if len(sys.argv) > 1 else "data/dummy_test_reduced.csv"

# Load Test Data
df_test = pd.read_csv(file_path)

# Load Trained Model
model = tf.keras.models.load_model("saved_model/lstm_model.h5", custom_objects={"mae": MeanAbsoluteError()})

# Prepare Training Data
df_train = pd.read_csv("data/train.csv")  # Required for extracting cfips history

# Create mapping of cfips to historical data
cfips_map = {
    cfips: df_train[df_train["cfips"] == cfips]["microbusiness_density"].values for cfips in df_train["cfips"].unique()
}

# Convert mapping to list format required for prediction
x = [[cfips, list(values)] for cfips, values in cfips_map.items()]

# Generate Predictions
predicted_result = []
for index, row in df_test.iterrows():
    cfips_idx = 0
    for i in range(len(x)):
        if x[i][0] == row["cfips"]:
            cfips_idx = i
            break  # Stop searching once found

    predict = model.predict(np.array([x[cfips_idx][1][-38:]]))[0][0]
    x[cfips_idx][1].append(predict)  # Append predicted value to history
    predicted_result.append(predict)

# Store predictions in dataframe
df_test["microbusiness_density"] = predicted_result

# Save Predictions
df_test[["row_id", "microbusiness_density", "cfips", "first_day_of_month"]].to_csv("data/submission.csv", index=False)
print("Predictions saved successfully!")
