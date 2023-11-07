import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(148, 3000)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(3000, 3000)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(3000, 3000)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(3000, 1)
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 30
    # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(x_train), batch_size)
    # Hold the best model
    best_acc = - np.inf
    # init to negative infinity
    best_weights = None
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
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

    current_directory = os.getcwd()
    data_path = os.path.join(current_directory,"early_relapse" , "data")

    # Read data
    early_relapse = pd.read_csv(os.path.join(data_path, "early_relapse_complete.csv"), sep=',')  # Reading early relapse data
    early_relapse = early_relapse.set_index('MPC')
    early_relapse = early_relapse["early_relapse"]

    filename = "CT_preprocessed.csv"
    CT_preprocessed = pd.read_csv(os.path.join(data_path, filename), sep=",")
    CT_preprocessed = CT_preprocessed.set_index('MPC')

    # Intersect the datasets
    dataset = CT_preprocessed.merge(early_relapse, on='MPC', how='inner')

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1:]

    # Import as PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

    # print(X.shape)
    # print(y.shape)

    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)

    # rebuild model with full set of training data
    model = Deep()
    acc = model_train(model, X_train, y_train, X_test, y_test)
    print(f"Final model accuracy: {acc*100:.2f}%")

    model.eval()

    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(10):
            y_pred = model(X_test[i:i+1])
            print(f" -> {y_pred[0].numpy()} " + f"(expected {y_test[i].numpy()})")

        # Define number of bootstrasp
        n_bootstrap_iterations = 100

        bootstrap_accuracies = []
        for _ in range(n_bootstrap_iterations):
            # Randomly sample the test dataset
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_bootstrap = X_test[indices]
            y_bootstrap = y_test[indices]

            # Evaluate the predicted values
            y_pred = model(X_bootstrap)

            # Accuracy evaluation
            accuracy = accuracy_score(y_bootstrap, y_pred.round())
            bootstrap_accuracies.append(accuracy)

        # Mean accuracy evaluation
        mean_accuracy = np.mean(bootstrap_accuracies)

        # confidence interval at 95%
        confidence_interval = np.percentile(bootstrap_accuracies, [2.5, 97.5])

        print("Mean accuracy:", mean_accuracy)
        print("Confidence interval at 95%:", confidence_interval)

        # Plot the ROC curve
        y_pred = model(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)  # ROC curve = TPR vs FPR
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
