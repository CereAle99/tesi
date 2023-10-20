import pandas as pd
import os

current_directory = os.getcwd()
data_path = os.path.join(current_directory, "data/")
filename = "CT_preprocessed.csv"

# Read data
data = pd.read_csv(data_path + filename, sep=",")
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1:]
