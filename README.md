# Prediction Canvas
The Prediction Canvas is a Python class that provides methods for preprocessing data, building and training a CatBoost model for classification, generating business rules based on feature importances, and creating segments based on those rules.

## Methods
**preprocess_dataframe(df):** Preprocesses the input dataframe by dropping null values, renaming columns, and extracting date components from specific columns.

**build_model(data):** Builds and trains a CatBoost model for classification using the input data. It preprocesses the data, splits it into features and target variables, creates a CatBoost Pool for training, and trains the model with specified parameters.

**generate_business_rules(X, y, target_variable, model):** Generates business rules based on feature importances. It calculates the feature importances using the trained model, identifies important categorical features, and creates rules for those features and their categories based on label distribution.

**create_segments(df, business_rules):** Creates segments based on the generated business rules. It takes the preprocessed data and the generated rules as inputs and returns a dictionary of segments, where each segment is defined by a feature-category pair and contains the corresponding indices from the input data.
