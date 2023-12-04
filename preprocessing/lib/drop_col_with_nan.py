def drop_columns_with_nan(dataframes_dict, print_output=True):
    """
    Processes a dictionary of DataFrames to remove columns that contain any NaN values.
    This can be particularly useful in data preprocessing where features with missing values
    need to be excluded from analyses or models.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are strings representing the names (or types) of DataFrames,
                              and values are the corresponding pandas DataFrames.

    - print_output (bool, optional): If set to True (default), the function will print the shape of each DataFrame
                                    after columns with NaN values have been dropped, as well as the names of the removed
                                     columns. If set to False, no output will be printed.

    Returns:
    - new_dataframes_dict (dict): The input dictionary with columns containing NaN values removed from each DataFrame.
    """
    new_dataframes_dict = {}
    removed_features = {}
    for omic, df5 in dataframes_dict.items():
        # Identify columns with NaN values
        nan_columns = df5.columns[df5.isna().any()].tolist()

        # Drop columns with NaN values
        new_df = df5.dropna(axis=1, how='any')

        # Store removed features
        removed_features[omic] = nan_columns

        new_dataframes_dict[omic] = new_df
        if print_output:
            print(f"{omic.upper()} shape after dropping columns with NaN values: {new_df.shape}")
            if nan_columns:
                print(f"Features removed in {omic}: {', '.join(nan_columns)}")
    return new_dataframes_dict