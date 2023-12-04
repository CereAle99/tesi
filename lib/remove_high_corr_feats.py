import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def remove_highly_correlated_features(dataframes_dict, threshold1=0.90, print_output=True, plot_heatmap=False):
    """
    Processes a dictionary of DataFrames to remove features that are highly correlated with other features.
    High correlations among features can cause multicollinearity issues in certain models, so it can be beneficial to
    remove such features during data preprocessing.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are strings representing the names (or types) of DataFrames, and
      values are the corresponding pandas DataFrames.

    - threshold (float, optional): The correlation coefficient threshold above which features will be considered highly
      correlated and thus will be removed. Default is 0.90.

    - print_output (bool, optional): If set to True (default), the function will print the shape of each DataFrame after
      removal of highly correlated features, as well as the names of the removed columns. If set to False, no output
      will be printed.

    - plot_heatmap (bool, optional): If set to True, the function will plot a heatmap of the correlation matrix for each
      DataFrame. This can be useful for visual inspection of correlations among features. Default is False.

    Returns:
    - tuple: A tuple containing two items:
    - dataframes_dict (dict): The input dictionary with highly correlated features removed from each DataFrame.
    - dropped_features (dict): A dictionary where keys are the same as the input dictionary, and values are lists of
      feature names that were dropped due to high correlation.
    """
    dropped_features = {}

    for omic, df6 in dataframes_dict.items():

        # Calculate the correlation matrix
        corr_matrix = df6.corr().abs()

        # Create a mask to identify highly correlated features
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold1)]

        # Plot the heatmap if required
        if plot_heatmap:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f"Heatmap of Correlation for {omic}")
            plt.show()

        # Drop the highly correlated features
        df_filtered = df6.drop(columns=to_drop)

        # Store the list of removed features
        dropped_features[omic] = to_drop

        # Update the DataFrame in dataframes_dict
        dataframes_dict[omic] = df_filtered

        num_dropped = len(to_drop)

        if print_output:
            print(f"{omic} after removing highly correlated features: {df_filtered.shape}")
            print(f"Number of features dropped in {omic}: {num_dropped}")
            if num_dropped > 0:
                print(f"Features dropped in {omic}: {', '.join(to_drop)}\n")

    return dataframes_dict, dropped_features