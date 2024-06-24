import pandas as pd
import os
current_directory = os.getcwd()

# path of clinical_data_300523.xlsx
data_path = os.path.join(current_directory, 'early_relapse', 'data')
# file_path = os.path.join(data_path, 'MMCLINICGenoMed4All25102023_2139785039.xlsx')
# file_path = os.path.join(data_path, 'clinical_data_300523.csv')
file_path = os.path.join(data_path, '1aCasistica_ULPWGS&PET_010322_per GenoMed.xlsx')

# load the file in a pandas dataframe and save it as csv
clinical_data = pd.read_excel(file_path, engine='openpyxl', index_col='MPC', sheet_name='clinical_data_ulp_pet')
file_path = os.path.join(data_path, 'surv_corrected_112.csv')
# clinical_data = pd.read_csv(file_path, index_col='MPC')

# add a column with the early relapse index
# clinical_data = clinical_data.dropna(subset=["PFS_I_EVENT", "PFS_I_MONTHS"])
# clinical_data['early_relapse'] = ((clinical_data['PFS_I_EVENT'] == 1) &
#                                   (clinical_data['PFS_I_MONTHS'] <= 12)).astype(int)

# save a csv file with just the early_relapse data
# early_relapse = clinical_data.loc[:, ["MPC", "PFS_I_MONTHS", "early_relapse"]]
# file_path = os.path.join(data_path, "early_relapse_complete.csv")
pfs_surv = clinical_data.loc[:, ["PFS_I_MONTHS", "PFS_I_EVENT"]]
print(pfs_surv.info())
pfs_surv = pfs_surv.dropna(subset=["PFS_I_EVENT", "PFS_I_MONTHS"])
print(pfs_surv.info())
pfs_surv.to_csv(file_path, index=True)


# save it to the file clinical_data_early_relapse.csv
# file_path = os.path.join(data_path, 'clinical_data_early_relapse_complete.csv')
# clinical_data.to_csv(file_path, index=False)




