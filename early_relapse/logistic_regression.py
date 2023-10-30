import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

current_directory = os.getcwd()
data_path = os.path.join(current_directory, "data/")

# Read data
early_relapse = pd.read_csv(data_path + "early_relapse.csv", sep=',')  # Reading early relapse data
early_relapse = early_relapse.set_index('MPC')
early_relapse = early_relapse["early_relapse"]

filename = "CT_preprocessed.csv"
CT_preprocessed = pd.read_csv(data_path + filename, sep=",")
CT_preprocessed = CT_preprocessed.set_index('MPC')

# Intersect the datasets
dataset = CT_preprocessed.merge(early_relapse, on='MPC', how='inner')

X = dataset.iloc[:, 0:-1]
y = dataset.iloc[:, -1:]
print(X.shape)
print(y.shape)

# Import as PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Crea un modello di regressione logistica con regolarizzazione L1
model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
model.fit(X_train, y_train)

# Calcola le previsioni sul set di test
y_pred = model.predict(X_test)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
