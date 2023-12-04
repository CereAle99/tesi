import numpy as np
import pandas as pd

def bin_feature(dataframe,
                feature,
                na_placeholder=np.NaN,
                bins=None,
                labels=None,
                na_value='NA'
                ):
    """
    Bins a continuous feature into specified intervals. If the interval is not specified bins data into quartiles.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing the feature to be binned.
    - feature (str): Name of the feature to bin.
    - na_placeholder (int, float or None, optional): Placeholder value for NA values. Defaults to np.nan.
    - bins (list): list of the bins limits. Default to []
    - labels (list): labels of the bins. Default to []
    - na_value (str or int, optional): The representation of NA in the data. Defaults to 'NA'.

    Returns:
    - pd.DataFrame: DataFrame with the binned feature.
    """

    # Handle NA values
    dataframe[feature] = dataframe[feature].replace(na_value, na_placeholder)
    dataframe[feature] = dataframe[feature].astype(float)

    # Intervals for binning, with the quartiles intervals or customized ones
    if bins is None:
        quartiles = np.percentile(dataframe[feature].dropna(), [0, 25, 50, 75, 100])
    else:
        quartiles = bins

    # Create a new binned column
    dataframe[f'{feature}'] = pd.cut(dataframe[feature],
                                     bins=quartiles,
                                     labels=labels,
                                     right=True,
                                     include_lowest=True,
                                     duplicates='drop'
                                     )

    return dataframe
