import pandas as pd

# Specifica il percorso del tuo file CSV
file_path = '.\data\clincal_data_300523.xlsx'

# Carica il file CSV in un DataFrame
df = pd.read_csv(file_path)

# Ora puoi lavorare con i dati nel DataFrame
# Ad esempio, per stampare le prime 5 righe:
print(df.head())
