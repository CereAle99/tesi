import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

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
# print(X.shape)
# print(y.shape)


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(148, 450)
        self.relu = nn.ReLU()
        self.output = nn.Linear(450, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(148, 150)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(150, 150)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(150, 150)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


# Compare model sizes
model1 = Wide()
model2 = Deep()
# print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))
# print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))


def model_train(model, x_train, y_train, x_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss() # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_epochs = 250
    # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(x_train), batch_size)
    # Hold the best model
    best_acc = - np.inf
    # init to negative infinity
    best_weights = None
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                x_batch = x_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(loss), acc=float(acc))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(x_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


# define 5-fold cross-validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model_wide = Wide()
    accuracy = model_train(model_wide, X[train], y[train], X[test], y[test])
    print("accuracy (wide): %.2f" % accuracy)
    cv_scores.append(accuracy)
# evaluate the model
accuracy = np.mean(cv_scores)
std = np.std(cv_scores)
print("Model accuracy: %.2f%% (+/- %.2f%%)" % (accuracy*100, std*100))


