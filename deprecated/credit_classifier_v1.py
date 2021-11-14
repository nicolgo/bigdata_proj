import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from sklearn.model_selection import train_test_split


def normalize(dataset):
    dataNorm = ((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataNorm["CreditLevel"] = dataset["CreditLevel"]-1
    return dataNorm


def prepare_data(train_dataset, batch_size, need_unify, is_test):
    x_train = torch.tensor(train_dataset.drop(
        'CreditLevel', axis=1).values.astype('float32'))
    y_values = train_dataset['CreditLevel'].values

    if(need_unify == True):
        y_copy = y_values.copy()
        for i, s in enumerate(y_copy):
            if s < 4:
                y_copy[i] = 0
            elif s > 7:
                y_copy[i] = 2
            else:
                y_copy[i] = 1
        y_train = torch.tensor(y_copy)
    else:
        y_train = torch.tensor(y_values)

    train_tensor = TensorDataset(x_train, y_train)
    if(is_test == True):
        tmp_dl = DataLoader(
            train_tensor, batch_size=batch_size, shuffle=False)
    else:
        tmp_dl = DataLoader(
            train_tensor, batch_size=batch_size, shuffle=True)
    return tmp_dl


class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Net, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, n_inputs*2)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.dropout1 = nn.Dropout(0.3)
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(n_inputs*2, n_outputs*2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.dropout2 = nn.Dropout(0.3)
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(n_outputs*2, n_outputs)
        xavier_uniform_(self.hidden3.weight)
        #self.dropout3 = nn.Dropout(0.3)
        self.act3 = nn.Softmax(dim=1)

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.dropout1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.dropout2(X)
        X = self.act2(X)
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
    number_of_labels = 3
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


def evaluate_model2(model, submodels, test_dl, test_dl_batch_size):
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
            _, rough_predicted = torch.max(outputs, 1)
            for i in range(len(submodels)):
                outputs = submodels[i](inputs)
                if i == 0:
                    _, sub_predicted0 = torch.max(outputs, 1)
                elif i == 1:
                    _, sub_predicted1 = torch.max(outputs, 1)
                else:
                    _, sub_predicted2 = torch.max(outputs, 1)

            predicted = rough_predicted.clone().detach()
            for i in range(len(rough_predicted)):
                if rough_predicted[i] == 0:
                    predicted[i] = sub_predicted0[i]
                elif rough_predicted[i] == 1:
                    predicted[i] = sub_predicted1[i]+4
                else:
                    predicted[i] = sub_predicted2[i]+8
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
    all_dataset = pd.read_csv("BankChurners.csv")
    all_dataset = all_dataset.drop(['CustomerId', 'Geography'], axis=1)
    all_dataset = normalize(all_dataset)
    train_dataset, test_dataset = train_test_split(
        all_dataset, test_size=0.2)
    sub_dataset1 = train_dataset[train_dataset['CreditLevel'] <= 3].copy()

    sub_dataset2 = train_dataset[(train_dataset['CreditLevel'] >= 4) & (
        train_dataset['CreditLevel'] <= 7)].copy()
    sub_dataset2["CreditLevel"] = sub_dataset2["CreditLevel"] - 4

    sub_dataset3 = train_dataset[train_dataset['CreditLevel'] >= 8].copy()
    sub_dataset3["CreditLevel"] = sub_dataset3["CreditLevel"] - 8

    train_dl_batch_size = 200
    test_dl_batch_size = 50
    train_dl = prepare_data(
        train_dataset, train_dl_batch_size, need_unify=True, is_test=False)
    sub_traindl_1 = prepare_data(
        sub_dataset1, train_dl_batch_size, need_unify=False, is_test=False)
    sub_traindl_2 = prepare_data(
        sub_dataset2, train_dl_batch_size, need_unify=False, is_test=False)
    sub_traindl_3 = prepare_data(
        sub_dataset3, train_dl_batch_size, need_unify=False, is_test=False)

    test_dl = prepare_data(test_dataset, test_dl_batch_size,
                           need_unify=False, is_test=True)

    class_model = Net(7, 3)
    sub_model_1 = Net(7, 4)
    sub_model_2 = Net(7, 4)
    sub_model_3 = Net(7, 2)
    train_model(train_dl, class_model)
    train_model(sub_traindl_1, sub_model_1)
    train_model(sub_traindl_2, sub_model_2)
    train_model(sub_traindl_3, sub_model_3)
    sub_models = [sub_model_1, sub_model_2, sub_model_3]
    # evaluate_model(class_model, sub_models, test_dl, test_dl_batch_size)
    evaluate_model2(class_model, sub_models, test_dl, test_dl_batch_size)
