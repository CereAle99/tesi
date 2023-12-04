from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def filter_low_variance_features(dataframes_dict, thresholds=0.001, default_threshold=0.01, print_output=True):
    """
    Processes a dictionary of DataFrames to remove features (columns) that have a variance below a specified threshold.
    The function is useful for feature selection, especially in contexts where features with low variance might not
    contribute significantly to a model.

    Parameters:
      - dataframes_dict (dict): A dictionary where keys are strings representing the names
      (or types) of DataFrames, and values are the corresponding pandas DataFrames.
      - thresholds (float, int, or dict, optional): The variance threshold below which features will be removed.
        It can be:
                    A single float or int value applied to all DataFrames.
                    A dictionary with keys corresponding to the keys in dataframes_dict and values being the thresholds
                    for each DataFrame. Default is 0.001.
      - default_threshold (float, optional): The default variance threshold to use for any DataFrame not specified in
        the threshold's dictionary. Default is 0.01.
      - print_output (bool, optional): If set to True (default), the function will print the shape of each DataFrame
      after filtering and the names of the dropped features. If set to False, no output will be printed.

    Returns:
      - tuple: A tuple containing two items:
            filtered_dataframes_dict (dict): The input dictionary with low variance features removed from
            each DataFrame.
            dropped_features (dict): A dictionary where keys are the same as the input dictionary,
            and values are lists of feature names that were dropped due to low variance.
    """
    dropped_features = {}

    # Make a copy of the dataframes_dict to avoid modifying the original
    filtered_dataframes_dict = dataframes_dict.copy()

    if isinstance(thresholds, (float, int)):
        thresholds = {omic: thresholds for omic in dataframes_dict.keys()}

    for omic, df4 in filtered_dataframes_dict.items():
        threshold1 = thresholds.get(omic, default_threshold)

        selector = VarianceThreshold(threshold=threshold1)
        selected_data = selector.fit_transform(df4)
        dropped_indices = ~selector.get_support()
        dropped_feature_names = df4.columns[dropped_indices]
        dropped_features[omic] = dropped_feature_names

        df_selected = pd.DataFrame(selected_data, index=df4.index, columns=df4.columns[selector.get_support()])
        filtered_dataframes_dict[omic] = df_selected

        num_dropped = len(dropped_feature_names)

        if print_output:
            print(f"{omic} after removing low variance features: {df_selected.shape}")
            print(f"Number of features dropped in {omic}: {num_dropped}")
            if num_dropped > 0:
                print(f"Features dropped in {omic}: {', '.join(dropped_feature_names)}\n")

    return filtered_dataframes_dict, dropped_features