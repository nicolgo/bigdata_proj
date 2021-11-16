from os import device_encoding
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm


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


def get_class_distribution(obj):
    count_dicts = {}
    for i in range(10):
        count_dicts[i] = 0
    for i in obj:
        count_dicts[i] += 1
    return count_dicts


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


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


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def train_model(train_loader, val_loader, model, device, class_weights):
    # define a Loss function and optimizer
    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for e in tqdm(range(1, num_epochs+1)):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        running_loss = 0.0
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(
                device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(
                    device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


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
            # c = (predicted == targets).squeeze()
            # for i in range(test_dl_batch_size):
            #     target = targets[i]
            #     class_correct[target] += c[i].item()
            #     class_total[target] += 1

    accuracy = (100*accuracy/total)
    print(f"Accuracy for all data: {accuracy}")
    # for i in range(number_of_labels):
    #     print('Accuracy of %5d : %2d %%' % (
    #         i, 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    # data preprocessing
    df = pd.read_csv("BankChurners.csv")
    df = df.drop(['CustomerId', 'Geography'], axis=1)
    df["CreditLevel"] = df["CreditLevel"]-1
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=69)
    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    # Train
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(
    ), x="variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(
    ), x="variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Val Set')
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(
    ), x="variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')
    train_dataset = ClassifierDataset(torch.from_numpy(
        X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(
        X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(
        X_test).float(), torch.from_numpy(y_test).long())

    # Weighted Sampling
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)
    target_list = torch.tensor(target_list)
    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    print(class_weights)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    train_batch_size = 100
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=train_batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_batch_size)

    num_feature = len(X.columns)
    num_classes = 10
    model = MulticlassClassification(num_feature, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_model(train_loader, val_loader, model, device, class_weights)
    # evaluate_model(model,test_loader,train_batch_size)

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
