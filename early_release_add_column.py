import pandas as pd
import os
current_directory = os.getcwd()

# path of clinical_data_300523.xlsx
relative_path = 'data/clinical_data_300523.xlsx'
file_path = os.path.join(current_directory, relative_path)

# load the file in a pandas dataframe
clinical_data = pd.read_excel(file_path, engine='openpyxl')

# add a column with the early relapse index
clinical_data['early_relapse'] = ((clinical_data['PFS_I_EVENT'] == 1) &
                                  (clinical_data['PFS_I_MONTHS'] <= 12)).astype(int)

# save it to the file clinical_data_early_relapse.csv
relative_path = 'data/clinical_data_early_relapse.csv'
file_path = os.path.join(current_directory, relative_path)
clinical_data.to_csv(file_path, index=False)


# paths of tot_rad_feats_CT.csv and tot_rad_feats_PET.csv
features_path1 = './data/tot_rad_feats_CT.csv'
features_path2 = './data/tot_rad_feats_PET.csv'

# load the file in a pandas dataframe
CT_features = pd.read_csv(features_path1)
PET_features = pd.read_csv(features_path2)

# print the info about the dataframes
print(CT_features.head())
print(CT_features.info())
print('\n ############### \n')
print(PET_features.head())
print(PET_features.info())


