from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_dataframes(dataframes_train_dict, dataframes_test_dict):
    """
    Scales all dataframes in the provided training dictionary using StandardScaler and
    applies the transformation learned from each training dataframe
    to the corresponding dataframe in the test dictionary.

    Parameters:
    - dataframes_train_dict (dict): Dictionary containing training dataframes to be scaled.
    - dataframes_test_dict (dict): Dictionary containing test dataframes where the scaling transformation learned from
      the training data is applied.

    Returns:
    - tuple: (Dictionary containing scaled training dataframes, Dictionary containing scaled test dataframes)
    """
    scaled_train_dict = {}
    scaled_test_dict = {}

    for key, train_df in dataframes_train_dict.items():
        if key != "CLI":
            scaler = StandardScaler()

            # Fit and transform the training data
            scaled_train_data = scaler.fit_transform(train_df)
            scaled_train_dict[key] = pd.DataFrame(scaled_train_data, columns=train_df.columns,
                                                  index=train_df.index)

            if dataframes_test_dict:
                # Transform the test data based on the fitted scaler
                test_df = dataframes_test_dict.get(key,
                                                   pd.DataFrame())  # get the test dataframe for the current
                # key, if it doesn't exist, use an empty dataframe
                scaled_test_data = scaler.transform(test_df)
                scaled_test_dict[key] = pd.DataFrame(scaled_test_data, index=test_df.index,
                                                     columns=test_df.columns)
        else:
            scaled_train_dict[key] = train_df
            if dataframes_test_dict:
                scaled_test_dict[key] = dataframes_test_dict[key]

    return scaled_train_dict, scaled_test_dict


def scale_dataframes_zscore(dataframes_train_dict, dataframes_test_dict=None):
    """
    Applies Z-score normalization to all dataframes in the provided training dictionary and
    uses the scaler trained on each dataframe in the dictionary to scale
    the corresponding dataframe in the test dictionary.

    Parameters:
    - dataframes_train_dict (dict): Dictionary containing training dataframes to be normalized.
    - dataframes_test_dict (dict): Dictionary containing test dataframes where the scaling transformation learned from
      the training data is applied.

    Returns:
    - tuple: (Dictionary containing scaled training dataframes, Dictionary containing scaled test dataframes)
    """
    scaler = StandardScaler()
    scaled_train_dict = {}
    scaled_test_dict = {}

    for key, train_df in dataframes_train_dict.items():
        if key != "CLI":
            # Fit on the training data
            scaler.fit(train_df)

            # Transform training data
            scaled_train_data = scaler.transform(train_df)
            scaled_train_dict[key] = pd.DataFrame(scaled_train_data, columns=train_df.columns, index=train_df.index)

            # Check if the key exists in dataframes_test_dict
            if dataframes_test_dict:
                if key in dataframes_test_dict:
                    # Transform the test data using the fitted scaler
                    test_df = dataframes_test_dict[key]
                    scaled_test_data = scaler.transform(test_df)
                    scaled_test_dict[key] = pd.DataFrame(scaled_test_data, columns=test_df.columns, index=test_df.index)
                else:
                    print(f"Warning: Key '{key}' not found in test data dictionary.")
        else:
            scaled_train_dict[key] = train_df
            if dataframes_test_dict:
                if key in dataframes_test_dict:
                    scaled_test_dict[key] = dataframes_test_dict[key]
                else:
                    print(f"Warning: Key '{key}' not found in test data dictionary.")

    return scaled_train_dict, scaled_test_dict