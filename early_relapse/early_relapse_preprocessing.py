import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 2D plotting library
import seaborn as sns  # Python data visualization library based on matplotlib
import warnings  # Typically issued in situations where it is useful to alert the user of some condition in a program
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines import KaplanMeierFitter

warnings.filterwarnings('ignore')  # Ignores all warnings


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




if __name__ == "__main__":

    current_directory = os.getcwd()
    data_path = os.path.join(current_directory , 'early_relapse', 'data')

    print("RAW MULTIOMICS DATASET")

    early_relapse = pd.read_csv(os.path.join(data_path, "early_relapse_complete.csv"), sep=',')  # Reading early relapse data
    early_relapse = early_relapse.set_index('MPC')
    early_relapse["PFS_I_MONTHS"] = early_relapse["PFS_I_MONTHS"] * 30

    dataframe0 = pd.read_csv(os.path.join(data_path, "clinical_data_complete.csv"),
                            sep=',',
                            dtype={'inst': object})  # Reading clinical dataset
    dataframe0 = dataframe0.set_index('MPC')
    dataframe0.index.name = None
    dataframe0 = dataframe0.drop(columns='LAST_FULLOW_UP', axis=1)
    print("CLINICAL DATA :", dataframe0.shape)

    dataframe1 = pd.read_csv(os.path.join(data_path, "tot_rad_feats_CT.csv"), sep=',')  # Reading CT dataset
    dataframe1 = dataframe1.set_index('MPC')
    dataframe1.index.name = None
    dataframe1 = dataframe1.drop(columns='MPC_EXAM_ID', axis=1)
    print("CT :", dataframe1.shape)

    dataframe2 = pd.read_csv(os.path.join(data_path, "tot_rad_feats_PET.csv"), sep=',')  # Reading DD dataset
    dataframe2 = dataframe2.set_index('MPC')
    dataframe2.index.name = None
    dataframe2 = dataframe2.drop(columns='MPC_EXAM_ID', axis=1)
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

    # BIN FEATURES


    dataframe0 = bin_feature(dataframe0, "CREATININE")
    dataframe0 = bin_feature(dataframe0, "CALCIUM")
    dataframe0 = bin_feature(dataframe0, "PC_TOT")
    dataframe0 = bin_feature(dataframe0, "PLT")
    dataframe0 = bin_feature(dataframe0, "HB")
    # dataframe0 = bin_feature(dataframe0, "EM_SUV_max")
    # dataframe0 = bin_feature(dataframe0, "PM_SUVmax")
    # dataframe0 = bin_feature(dataframe0, "FL_SUV_max")
    # dataframe0 = bin_feature(dataframe0, "cfDNA_tumor_fraction")
    # dataframe0 = bin_feature(dataframe0, "gDNA_tumor_fraction")
    # dataframe0 = bin_feature(dataframe0, "PCR")

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


    # PLOT KM CURVE WITH ONE FEATURE

    # Usage example:
    plot_km_curve_for_feature(early_relapse, dataframes["CLI"], 30)


    # PLOT KM CURVE WITH MORE FEATURES

    # Usage example:
    features_to_plot = [45, 46, 47]
    plot_km_curve_for_features(early_relapse, dataframes["CLI"], features_to_plot)


    # REMOVE DUPLICATED ROWS AND COLUMN

    # Call the function with your dataframes_transposed
    dataframes, dropped = remove_duplicated_data(dataframes, print_output=True)


    # Determine the number of omics DataFrames
    num_omics = len(dataframes)

    # Create subplots for all omics DataFrames
    fig, axes = plt.subplots(num_omics, 2, figsize=(12, 5 * num_omics))

    # Iterate through dataframes and create plots for each omics
    for i, (omics_name, df3) in enumerate(dataframes.items()):
        # Compute feature variability (variance) for columns
        feature_variability = df3.var(axis=0, skipna=True)

        # Identify columns with all zeros
        null_feature_count = (df3.sum(axis=0) == 0).sum()

        # Create KDE plots for feature variability in the i-th subplot
        sns.kdeplot(feature_variability, ax=axes[i, 0])
        axes[i, 0].set_title(f'{omics_name} - Feature Variability')
        axes[i, 0].set_xlabel('Variance')

        # Plot a bar for the count of null features
        axes[i, 1].bar(1, null_feature_count)  # Only one bar showing count of all-zero columns
        axes[i, 1].set_title(f'{omics_name} - Null Features')
        axes[i, 1].set_ylabel('Count of All-Zero Columns')
        axes[i, 1].set_xticks([])  # Remove x-ticks as they are not relevant here

    # Adjust layout and show all plots
    plt.tight_layout()
    plt.show()


    # FILTER LOW VARIANCE FEATURES


    # Example usage with different thresholds for each omic type and a default threshold of 0.03
    # thresholds = {'CLI': 0.01, 'T2': 0.0001,'T1': 0.0001,'T1Gd': 0.0001,'FLAIR': 0.0001}
    # default_threshold = 0.01
    dataframes, uninformative_features = filter_low_variance_features(dataframes, default_threshold=0.01, print_output=True)

    # Dealing with missing values and highly correlated features

    # Calculate NA ratios per sample for each omics
    na_ratios_per_sample = {omic: df.isna().sum() / df.shape[0] for omic, df in dataframes.items()}

    # Create a plot for each omics
    fig, axes = plt.subplots(len(dataframes), figsize=(12, 3 * len(dataframes)))

    for i, (omic, na_ratios) in enumerate(na_ratios_per_sample.items()):
        ax = axes[i]
        sns.barplot(
            x=np.arange(0, na_ratios.shape[0]),
            y=na_ratios.values,
            ax=ax
        )
        ax.set_xlabel('Sample')
        ax.set_ylabel('Ratio of NAs')
        ax.set_title(f'NA Ratios per Sample - {omic}')

    plt.tight_layout()
    plt.show()


    # DROP COLUMNS WITH NAN


    # Call the function with your dataframes_transposed
    dataframes = drop_columns_with_nan(dataframes, print_output=True)


    # REMOVE HIGHLY CORRELATED FEATURES


    dataframes, dropped_corr_features = remove_highly_correlated_features(dataframes,
                                                                        print_output=True,
                                                                        plot_heatmap=True,
                                                                        threshold1=0.90)

    # 1. Find Common Patients
    # Start with the patients in the survival dataframe
    common_patients = set(early_relapse.index)

    # Find intersection with patients in each omics dataframe
    for omic_name, df in dataframes.items():
        common_patients = common_patients.intersection(df.index)

    print("Number of patients present in all dataframes and the survival dataframe:", len(common_patients))

    common_patients = list(common_patients)

    # 2. Perform Stratified Sampling
    # Filter the survival dataframe to only the common patients
    filtered_survival_df = early_relapse.loc[common_patients]

    # Set the number of patients for the test set
    n_test_patients = 20

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_test_patients, random_state=42)
    train_index, test_index = next(splitter.split(filtered_survival_df, filtered_survival_df['early_relapse']))

    train_patients = filtered_survival_df.iloc[train_index].index
    test_patients = filtered_survival_df.iloc[test_index].index

    # 3. Split the Dataframes
    dataframes_train = {}
    dataframes_test = {}

    for omic_name, df in dataframes.items():
        dataframes_train[omic_name] = df.loc[df.index.difference(test_patients)]  # All rows except the test patients
        dataframes_test[omic_name] = df.loc[test_patients]

    # Displaying sizes
    for omic_name, df_train in dataframes_train.items():
        print(f"{omic_name} Train: {df_train.shape}")
        print(f"{omic_name} Test: {dataframes_test[omic_name].shape}")

    # Data standardization

    for df, item in dataframes.items():
        print(item.shape)

    # SCALE DATAFRAME

    # SCALE DATAFRAME ZSCORE


    scaled_train_dataframes, scaled_test_dataframes = scale_dataframes_zscore(dataframes_train, dataframes_test)
    dataframes, _ = scale_dataframes_zscore(dataframes)

    for df, item in scaled_train_dataframes.items():
        print(item.shape)
        print(df)
        file_path = data_path + f"/{df}_preprocessed_train.csv"
        item.to_csv(file_path, index=True, index_label="MPC")

    for df, item in scaled_test_dataframes.items():
        print(item.shape)
        print(df)
        file_path = data_path + f"/{df}_preprocessed_test.csv"
        item.to_csv(file_path, index=True, index_label="MPC")

    for df, item in dataframes.items():
        print(item.shape)
        print(df)
        file_path = data_path + f"/{df}_preprocessed.csv"
        item.to_csv(file_path, index=True, index_label="MPC")