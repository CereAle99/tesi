import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

current_directory = os.getcwd()
data_path = os.path.join(current_directory, "data/")

# Read data
early_relapse = pd.read_csv(data_path + "early_relapse.csv", sep=',')  # Reading early relapse data
early_relapse = early_relapse.set_index('MPC')
early_relapse = early_relapse["early_relapse"]

filename = "PET_preprocessed.csv"
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, shuffle=True)

# Create a model and train it
lr_model = LogisticRegression(penalty='l2', C=1.0, solver='saga')
lr_model.fit(X_train, y_train.reshape(-1))

# Define number of bootstrasp
n_bootstrap_iterations = 1000

bootstrap_accuracies = []
for _ in range(n_bootstrap_iterations):
    # Randomly sample the test dataset
    indices = np.random.choice(len(X_test), len(X_test), replace=True)
    X_bootstrap = X_test[indices]
    y_bootstrap = y_test[indices]

    # Evaluate the predicted values
    y_pred = lr_model.predict(X_bootstrap)

    # Accuracy evaluation
    accuracy = accuracy_score(y_bootstrap, y_pred)
    bootstrap_accuracies.append(accuracy)

# Mean accuracy evaluation
mean_accuracy = np.mean(bootstrap_accuracies)

# Confidence interval at 95%
confidence_interval = np.percentile(bootstrap_accuracies, [2.5, 97.5])

print("Mean accuracy:", mean_accuracy)
print("Confidence interval at 95%:", confidence_interval)

with torch.no_grad():
    # Plot the ROC curve
    y_pred = lr_model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)  # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
