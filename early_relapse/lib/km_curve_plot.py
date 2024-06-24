import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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

    plt.title(f"Kaplan-Meier survival curve for {feature_name}")
    plt.xlabel("Time (days)")
    plt.ylabel("Probability of survival")
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

    plt.title(f"Kaplan-Meier survival curves")
    plt.xlabel("Time (days)")
    plt.ylabel("Probability of survival")
    plt.grid(True)
    plt.legend()
    plt.show()

