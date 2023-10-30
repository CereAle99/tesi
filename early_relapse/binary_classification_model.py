import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


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
        self.layer1 = nn.Linear(148, 300)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(300, 300)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(300, 300)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(300 , 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


def model_train(model, x_train, y_train, x_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_epochs = 250
    # number of epochs to run
    batch_size = 20  # size of each batch
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
                x_batch = x_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
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


if __name__ == "__main__":

    # Compare model sizes
    model1 = Wide()
    model2 = Deep()

    # print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))
    # print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))

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

    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # define 5-fold cross-validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores_wide = []
    for train, test in kfold.split(X_train, y_train):
        # create model, train, and get accuracy
        model = Wide()
        acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
        print("Accuracy (wide): %.2f" % acc)
        cv_scores_wide.append(acc)

    cv_scores_deep = []
    for train, test in kfold.split(X_train, y_train):
        # create model, train, and get accuracy
        model = Deep()
        acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
        print("Accuracy (deep): %.2f" % acc)
        cv_scores_deep.append(acc)

    # evaluate the model
    wide_acc = np.mean(cv_scores_wide)
    wide_std = np.std(cv_scores_wide)
    deep_acc = np.mean(cv_scores_deep)
    deep_std = np.std(cv_scores_deep)
    print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc * 100, wide_std * 100))
    print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc * 100, deep_std * 100))

    # rebuild model with full set of training data
    if wide_acc > deep_acc:
        print("Retrain a wide model")
        model = Wide()
    else:
        print("Retrain a deep model")
        model = Deep()
    acc = model_train(model, X_train, y_train, X_test, y_test)
    print(f"Final model accuracy: {acc*100:.2f}%")

    model.eval()

    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(5):
            y_pred = model(X_test[i:i+1])
            print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} " + f"(expected {y_test[i].numpy()})")

    with torch.no_grad():
        # Plot the ROC curve
        y_pred = model(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)  # ROC curve = TPR vs FPR
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
