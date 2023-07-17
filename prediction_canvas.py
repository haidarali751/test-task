import pandas as pd
import catboost as cb
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class PredictionCanvas:
    def preprocess_dataframe(self, df):
        """
        Preprocesses the input dataframe.

        Args:
        df (pd.DataFrame): The input dataframe.

        Returns:
        pd.DataFrame: The preprocessed dataframe."""

        column_name = df.columns
        # drop null values
        if df.isnull().sum().sum():
            df.dropna(inplace=True)

        new_col = {col: col.replace(" ", "_").lower() for col in df.columns}
        df.rename(columns=new_col, inplace=True)

        date_columns = ["last_modified_date", "close_date", "created_date"]

        for column in date_columns:
            df[column] = pd.to_datetime(df[column])
            df[column + "_year"] = df[column].dt.year
            df[column + "_month"] = df[column].dt.month
            df[column + "_day"] = df[column].dt.day
            df[column + "_dayofweek"] = df[column].dt.dayofweek

        unnecessary_col = [
            "close_date",
            "created_date",
            "last_modified_date",
            "last_modified_by",
            "created_by",
            "phone",
            "email",
            "account_name",
            "contact_name",
        ]
        df.drop(unnecessary_col, axis=1, inplace=True)

        return df

    def build_model(self, data):
        """
        Builds and trains a CatBoost model for classification.

        Args:
        data (pd.DataFrame): The input data containing features and target variable.

        Returns:
        cb.CatBoostClassifier: The trained CatBoost model."""

        # Preprocess your data and split it into features and target variables
        features = data.drop("status", axis=1)
        targets = data["status"]

        # Split the data into training and testing sets
        train_size = int(0.8 * len(data))  # Adjust the split ratio as needed
        features_train, features_test = features[:train_size], features[train_size:]
        targets_train, targets_test = targets[:train_size], targets[train_size:]

        # Get the column names of the categorical features
        categorical_features = features.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # Create a CatBoost Pool for training
        train_pool = cb.Pool(
            data=features_train, label=targets_train, cat_features=categorical_features
        )

        # Train the CatBoost model
        model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1)
        model.fit(train_pool)
        return model

    def generate_business_rules(self, X, y, target_variable, model):
        """
        Generates business rules based on feature importances.

        Args:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        target_variable (str): The name of the target variable.
        model: The trained CatBoost model.

        Returns:
        dict: A dictionary of business rules in the format {(feature, category): rule}.
        """
        # Generate business rules based on feature importances

        business_rules = {}
        feature_importances = model.get_feature_importance()

        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        for feature, importance in zip(X.columns, feature_importances):
            if importance > 0.2 and feature in categorical_features:
                unique_categories = X[feature].unique()

                for category in unique_categories:
                    category_indices = np.where(X[feature] == category)[0]
                    category_labels = y.iloc[category_indices]

                    # Calculate the label distribution for the category
                    label_distribution = category_labels.value_counts(normalize=True)

                    # Generate the rule using the category and label distribution
                    rule = f"If {feature} is '{category}', then target variable distribution: {label_distribution.to_dict()}"
                    business_rules[(feature, category)] = rule

        return business_rules

    def create_segments(self, df, business_rules):
        """
        Creates segments based on business rules.

        Args:
        df (pd.DataFrame): The input data.
        business_rules (dict): A dictionary of business rules in the format {(feature, category): rule}.

        Returns:
        dict: A dictionary of segments in the format {(feature, category): segment_indices}.
        """
        # Create segments based on business rules
        features = df.drop("status", axis=1)
        segments = {}

        for (feature, category), rule in business_rules.items():
            category_indices = np.where(features[feature] == category)[0]
            segments[(feature, category)] = category_indices.tolist()

        return segments
