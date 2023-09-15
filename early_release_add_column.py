import pandas as pd



# path of clinical_data_300523.xlsx
file_path = './data/clincal_data_300523.xlsx'

# load the file in a pandas dataframe
clinical_data = pd.read_excel(file_path, engine='openpyxl')

# add a column with the early relapse index
clinical_data['early_relapse'] = ((clinical_data['PFS_I_EVENT'] == 1) & (clinical_data['PFS_I_MONTHS'] <= 12)).astype(int)

# save it back to the file clinical_data_early_relapse.csv
file_path = './data/clinical_data_early_relapse.csv'
clinical_data.to_csv(file_path, index=False)

