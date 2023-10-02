import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 2D plotting library
import seaborn as sns  # Python data visualization library based on matplotlib
import warnings  # Typically issued in situations where it is useful to alert the user of some condition in a program
from collections import Counter  # Supports iterations
from os import path
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler  # Feature scaling

from lifelines import KaplanMeierFitter

warnings.filterwarnings('ignore')  # Ignores all warnings

current_directory = os.getcwd()
data_path = current_directory + '/data/'

print("RAW MULTIOMICS DATASET")

early_relapse = pd.read_csv(data_path + "early_relapse.csv", sep=',')  # Reading early relapse data
early_relapse = early_relapse.set_index('MPC')
early_relapse["PFS_I_MONTHS"] = early_relapse["PFS_I_MONTHS"] * 30

dataframe0 = pd.read_csv(data_path + "clinical_data_300523.csv",
                         sep=',',
                         dtype={'inst': object})  # Reading clinical dataset
dataframe0 = dataframe0.set_index('MPC')
dataframe0.index.name = None
print("CLINICAL DATA :", dataframe0.shape)

dataframe1 = pd.read_csv(data_path + "tot_rad_feats_CT.csv", sep=',')  # Reading CT dataset
dataframe1 = dataframe1.set_index('MPC')
dataframe1.index.name = None
print("CT :", dataframe1.shape)

dataframe2 = pd.read_csv(data_path + "tot_rad_feats_PET.csv", sep=',')  # Reading DD dataset
dataframe2 = dataframe2.set_index('MPC')
dataframe2.index.name = None
print("PET :", dataframe2.shape)

# Missing data analysis

# Replacing the string "nv" with NaN and single and double with 1 and 2 in pandas dataframe0
dataframe0 = dataframe0.replace("nv", np.nan)
dataframe0 = dataframe0.replace("single", 1)
dataframe0 = dataframe0.replace("double", 2)

threshold = 20  # Set missing data threshold in %

# Calculate the percentage of missing data for each column
missing_percentage = dataframe0.isna().mean().round(4) * 100

# Columns to keep based on the threshold
cols_to_keep = missing_percentage[missing_percentage <= threshold].index.tolist()

# Filter dataframe based on these columns
dataframe0 = dataframe0[cols_to_keep]


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


dataframe0 = bin_feature(dataframe0, "CREATININE")
dataframe0 = bin_feature(dataframe0, "CALCIUM")
dataframe0 = bin_feature(dataframe0, "PC_TOT")
dataframe0 = bin_feature(dataframe0, "PLT")
dataframe0 = bin_feature(dataframe0, "HB")
dataframe0 = bin_feature(dataframe0, "EM_SUV_max")
dataframe0 = bin_feature(dataframe0, "PM_SUVmax")
dataframe0 = bin_feature(dataframe0, "FL_SUV_max")
dataframe0 = bin_feature(dataframe0, "cfDNA_tumor_fraction")
dataframe0 = bin_feature(dataframe0, "gDNA_tumor_fraction")

# Defining the bins for age categorization and binning the age labels
age_bins = [0, 39, 65, float('inf')]
age_labels = ['<40y', '40-65y', '>65y']
dataframe0 = bin_feature(dataframe0, 'AGE', bins=age_bins, labels=age_labels)

# one hot encoding
dataframe0 = pd.get_dummies(dataframe0, drop_first=True)

# Adjusting dataframes

dataframes = {
    "CLI": dataframe0,
    "CT": dataframe1,
    "PET": dataframe2
}

# Iterate through dataframes and prepend omics name to column names for easier visualization
for omics_name, df in dataframes.items():
    # Check if the first column already starts with the omics_name
    if not df.columns[0].startswith(f"{omics_name}_"):
        new_columns = [f"{omics_name}_{col}" for col in df.columns]
        df.columns = new_columns
    print(f"{omics_name}: {df.shape}")

kmf1 = KaplanMeierFitter()
kmf1.fit(early_relapse["PFS_I_MONTHS"], early_relapse["early_relapse"])

kmf1.plot_survival_function()
plt.title("Kaplan-Meier Early Relapse Curve")
plt.xlabel("Time")
plt.ylabel("Probability of relapse in 12 months")
plt.grid(True)
plt.show()

column_dict = {col: i for i, col in enumerate(dataframes["CLI"].columns)}
print(column_dict)


def plot_km_curve_for_feature(early_relapse_df, dataframe, feature_identifier):
    """
    Plot the Kaplan-Meier early_relapse curves for the specified one-hot encoded feature.

    Parameters:
    - early_relapse_df: The early_relapse data DataFrame.
    - dataframe: The DataFrame with one-hot encoded features.
    - feature_identifier: Either the name or the index of the one-hot encoded feature to plot.

    Returns: None
    """

    kmf = KaplanMeierFitter()

    # Check if the identifier is an integer (indicating an index). If so, get the feature name by index.
    if isinstance(feature_identifier, int):
        feature_name = dataframe.columns[feature_identifier]
    else:
        feature_name = feature_identifier

    # Find common patients between the two dataframes
    common_patients = early_relapse_df.index.intersection(dataframe.index)

    # Group 0
    idx_0 = dataframe.loc[common_patients, feature_name] == 0
    patient_ids_0 = common_patients[idx_0]
    kmf.fit(early_relapse_df.loc[patient_ids_0, "PFS_I_MONTHS"], early_relapse_df.loc[patient_ids_0, "PFS_I_MONTHS"],
            label=f'{feature_name}_0')
    kmf.plot_survival_function()

    # Group 1
    idx_1 = dataframe.loc[common_patients, feature_name] == 1
    patient_ids_1 = common_patients[idx_1]
    kmf.fit(early_relapse_df.loc[patient_ids_1, "PFS_I_MONTHS"], early_relapse_df.loc[patient_ids_1, "PFS_I_MONTHS"],
            label=f'{feature_name}_1')
    kmf.plot_survival_function()

    plt.title(f"Kaplan-Meier early_relapse Curve for {feature_name}")
    plt.xlabel("Time")
    plt.ylabel("Probability of early_relapse")
    plt.grid(True)
    plt.show()


# Usage example:
plot_km_curve_for_feature(early_relapse, dataframes["CLI"], 113)


def plot_km_curve_for_features(survival_df, dataframe, features, value=1):
    """
    Plot the Kaplan-Meier survival curves for a list of specified features based on a given value (0 or 1).

    Parameters:
    - survival_df: The survival data DataFrame.
    - dataframe: The DataFrame with one-hot encoded features.
    - features: List of features to plot.
    - value: The value (0 or 1) to filter the data on for all features. Default is 1.

    Returns: None
    """
    kmf = KaplanMeierFitter()

    # Find common patients between the two dataframes
    common_patients = survival_df.index.intersection(dataframe.index)

    for feature in features:
        # Check if the feature is given as an integer (indicating index). If so, get the feature name by index.
        if isinstance(feature, int):
            feature = dataframe.columns[feature]

        idx = dataframe.loc[common_patients, feature] == value
        patient_ids = common_patients[idx]
        kmf.fit(survival_df.loc[patient_ids, "PFS_I_MONTHS"],
                survival_df.loc[patient_ids, "early_relapse"],
                label=feature)
        kmf.plot_survival_function()

    plt.title(f"Kaplan-Meier early_relapse Curves")
    plt.xlabel("Time")
    plt.ylabel("Probability of early_relapse")
    plt.grid(True)
    plt.legend()
    plt.show()


# Usage example:
features_to_plot = [111, 112, 113]
plot_km_curve_for_features(early_relapse, dataframes["CLI"], features_to_plot)


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
        duplicated_columns = df1.columns[df.T.duplicated()].tolist()
        deduplicated_df = deduplicated_df.T.drop_duplicates().T

        # Store dropped rows and columns in the dropped_data dictionary
        dropped_data[omic] = {
            "dropped_rows": duplicated_rows.set_index(df.index.name or 'index'),
            "dropped_columns": duplicated_columns
        }

        dataframes_dict[omic] = deduplicated_df

        # Print shapes before and after removing duplicates on the same line
        if print_output:
            print(f"{omic.upper()}: Original shape {original_shape}, after removing duplicates {deduplicated_df.shape}")

    return dataframes_dict, dropped_data


# Call the function with your dataframes_transposed
dataframes, dropped = remove_duplicated_data(dataframes, print_output=True)
