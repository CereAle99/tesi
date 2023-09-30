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


