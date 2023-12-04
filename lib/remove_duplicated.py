def remove_duplicated_data(dataframes_dict, print_output=True):
    """
    Processes a dictionary of DataFrames to remove both duplicated rows and columns from each DataFrame.
    It also provides an option to print the original and deduplicated shapes of the DataFrames.

    Parameters:
    dataframes_dict (dict): A dictionary where keys are strings representing the names (or types)
                            of DataFrames, and values are the corresponding pandas DataFrames.

    print_output (bool, optional): If set to True (default), the function will print the original
                                   and deduplicated shapes of each DataFrame. If set False, no output will be printed.

    Returns:
    tuple: A tuple containing two dictionaries:
    dataframes_dict (dict): The input dictionary with each DataFrame having duplicated rows and columns removed.
    dropped_data (dict): A dictionary where keys are the same as the input dictionary, and values are another
                        dictionary containing:
                        "dropped_rows": A DataFrame of rows that were detected as duplicates and removed.
                        "dropped_columns": A list of column names that were detected as duplicates and removed.
    """
    dropped_data = {}  # Dictionary to store the dropped rows and columns for each dataframe

    for omic, df1 in dataframes_dict.items():
        original_shape = df1.shape

        # Reset index temporarily to check for duplicates
        temp_df = df1.reset_index()

        # Track and remove duplicated rows
        duplicated_rows = temp_df[temp_df.duplicated()]
        deduplicated_df = temp_df.drop_duplicates().set_index(df1.index.name or 'index')

        # Track and remove duplicated columns
        duplicated_columns = df1.columns[df1.T.duplicated()].tolist()
        deduplicated_df = deduplicated_df.T.drop_duplicates().T

        # Store dropped rows and columns in the dropped_data dictionary
        dropped_data[omic] = {
            "dropped_rows": duplicated_rows.set_index(df1.index.name or 'index'),
            "dropped_columns": duplicated_columns
        }

        dataframes_dict[omic] = deduplicated_df

        # Print shapes before and after removing duplicates on the same line
        if print_output:
            print(f"{omic.upper()}: Original shape {original_shape}, after removing duplicates {deduplicated_df.shape}")

    return dataframes_dict, dropped_data