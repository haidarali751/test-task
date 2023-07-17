import pandas as pd
from prediction_canvas import PredictionCanvas

df = pd.read_csv("closed_dataset.csv", index_col=[0])

# Create an instance of the PredictionCanvas class
prediction_canvas = PredictionCanvas()


# Preprocess the data
preprocessed_data = prediction_canvas.preprocess_dataframe(df)

# Build the predictive model using the closed data
model = prediction_canvas.build_model(df)

# Generate business rules based on the closed data and the trained model
rules = prediction_canvas.generate_business_rules(
    df.drop("status", axis=1), df["status"], "status", model
)
print(rules)

# Create segments in the open data based on the generated rules
segmented_data = prediction_canvas.create_segments(df, rules)
print(segmented_data)
