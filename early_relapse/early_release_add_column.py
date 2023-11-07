import pandas as pd
import os
current_directory = os.getcwd()

# path of clinical_data_300523.xlsx
data_path = os.path.join(current_directory, 'early_relapse', 'data')
file_path = os.path.join(data_path, 'MMCLINICGenoMed4All25102023_2139785039.xlsx')

# load the file in a pandas dataframe and save it as csv
clinical_data = pd.read_excel(file_path, engine='openpyxl')
file_path = os.path.join(data_path, 'clinical_data_complete.csv')
clinical_data.to_csv(file_path, index=False)

# add a column with the early relapse index
clinical_data['early_relapse'] = ((clinical_data['PFS_I_EVENT'] == 1) &
                                  (clinical_data['PFS_I_MONTHS'] <= 12)).astype(int)

# save a csv file with just the early_relapse data
early_relapse = clinical_data.loc[:, ["MPC", "PFS_I_MONTHS", "early_relapse"]]
file_path = os.path.join(data_path, "early_relapse_complete.csv")
early_relapse.to_csv(file_path, index=False)


# save it to the file clinical_data_early_relapse.csv
file_path = os.path.join(data_path, 'clinical_data_early_relapse_complete.csv')
clinical_data.to_csv(file_path, index=False)




