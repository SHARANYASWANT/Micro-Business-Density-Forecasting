import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from tensorflow.keras.losses import MeanAbsoluteError

# Get file path from command-line argument or use default
file_path = sys.argv[1] if len(sys.argv) > 1 else "data/dummy_test_2026.csv"

# Load Test Data
df_test = pd.read_csv(file_path)

# Load Trained Model
model = tf.keras.models.load_model("saved_model/lstm_model.h5", custom_objects={"mae": MeanAbsoluteError()})

# Prepare Test Data
df_train = pd.read_csv("data/train.csv")  # Required for extracting cfips history
cfips_map = {
    cfips: df_train[df_train["cfips"] == cfips]["microbusiness_density"].values for cfips in df_train["cfips"].unique()
}

# Generate Predictions
test_data = np.array([cfips_map.get(row["cfips"], [0] * 38)[-38:] for _, row in df_test.iterrows()])
predicted_result = model.predict(test_data)

# Save Predictions
df_test["microbusiness_density"] = predicted_result.flatten()
df_test[["row_id", "microbusiness_density", "cfips"]].to_csv("data/submission.csv", index=False)
print("Predictions saved successfully!")
