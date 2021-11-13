import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_


def normalize(dataset):
    dataNorm = ((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataNorm["CreditLevel"] = dataset["CreditLevel"]-1
    return dataNorm


def prepare_data(train_dataset, train_dl_batch_size, test_dl_batch_size):
    x_train = torch.tensor(train_dataset.drop(
        'CreditLevel', axis=1).values.astype('float32'))
    y_values = train_dataset['CreditLevel'].values
    # y_copy = y_values.copy()
    # for i, s in enumerate(y_copy):
    #     if s < 4:
    #         y_copy[i] = 0
    #     elif s > 6:
    #         y_copy[i] = 2
    #     else:
    #         y_copy[i] = 1

    y_train = torch.tensor(y_values)

    train_tensor = TensorDataset(x_train, y_train)
    n_train = int(len(train_tensor)*0.8)
    n_test = len(train_tensor) - n_train
    train_ds, test_ds = random_split(train_tensor, [n_train, n_test])
    train_dl = DataLoader(
        train_ds, batch_size=train_dl_batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=test_dl_batch_size, shuffle=False)
    return train_dl, test_dl


class Net(nn.Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 14)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.dropout1 = nn.Dropout(0.3)
        self.act1 = nn.ReLU()

        # second hidden layer
        # self.hidden2 = nn.Linear(14, 10)
        # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # self.dropout2 = nn.Dropout(0.3)
        # self.act2 = nn.ReLU()

        # third hidden layer and output
        self.hidden3 = nn.Linear(14, 10)
        xavier_uniform_(self.hidden3.weight)
        #self.dropout3 = nn.Dropout(0.3)
        self.act3 = nn.Softmax(dim=1)

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.dropout1(X)
        X = self.act1(X)

        # second hidden layer
        #X = self.hidden2(X)
        #X = self.dropout2(X)
        #X = self.act2(X)

        # output layer
        X = self.hidden3(X)
        #X = self.dropout3(X)
        X = self.act3(X)
        return X


def train_model(train_dl, model):
    # define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


def evaluate_model(model, test_dl, test_dl_batch_size):
    model.eval()
    accuracy = 0.0
    total = 0.0
    number_of_labels = 10
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in test_dl:
            inputs, targets = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            accuracy += (predicted == targets).sum().item()
            c = (predicted == targets).squeeze()
            for i in range(test_dl_batch_size):
                target = targets[i]
                class_correct[target] += c[i].item()
                class_total[target] += 1

    accuracy = (100*accuracy/total)
    print(f"Accuracy for all data: {accuracy}")
    for i in range(number_of_labels):
        print('Accuracy of %5d : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    # data preprocessing
    train_dataset = pd.read_csv("BankChurners.csv")
    print(
        f"Original data frame has {train_dataset.shape[0]} rows and {train_dataset.shape[1]} columns.")
    train_dataset = train_dataset.drop(['CustomerId', 'Geography'], axis=1)
    train_dataset = normalize(train_dataset)

    train_dl_batch_size = 100
    test_dl_batch_size = 30
    train_dl, test_dl = prepare_data(
        train_dataset, train_dl_batch_size, test_dl_batch_size)
    model = Net(7)
    print(model)
    train_model(train_dl, model)
    evaluate_model(model, test_dl, test_dl_batch_size)
