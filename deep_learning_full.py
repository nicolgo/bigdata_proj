import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.utils.data as Data
from sklearn import neighbors
from imblearn.over_sampling import SMOTENC
pd.options.mode.chained_assignment = None


def combine_feature(data_set):
    """
    组合特征
    :param data_set: 数据集
    :return: data_set
    """
    feat1 = []
    feat2 = []
    feat3 = []
    for item in data_set.values:
        products = item[4]
        cr_card = item[5]
        active_member = item[5]
        exit = item[8]
        if cr_card == 0:
            if products == 1:
                feat1.append(1)
            elif products == 2:
                feat1.append(2)
            elif products == 3:
                feat1.append(3)
            else:
                feat1.append(4)

            if active_member == 0:
                feat2.append(1)
            else:
                feat2.append(2)

            if exit == 0:
                feat3.append(1)
            else:
                feat3.append(2)
        else:
            if products == 1:
                feat1.append(5)
            elif products == 2:
                feat1.append(6)
            elif products == 3:
                feat1.append(7)
            else:
                feat1.append(8)

            if active_member == 0:
                feat2.append(3)
            else:
                feat2.append(4)

            if exit == 0:
                feat3.append(3)
            else:
                feat3.append(4)
    data_set['NewFeature1'] = feat1
    data_set['NewFeature2'] = feat2
    data_set['NewFeature3'] = feat3
    return data_set


def predict_balance(data_set):
    """
    将缺失的balance col数据补全
    linear_model.LinearRegression() 线性回归补全
    :param data_set: 读取数据集
    :return: 补全后的data_set
    """
    zero_data = []
    norm_data = []
    for item in data_set.values:
        if item[3] == 0:
            zero_data.append(item)
        else:
            norm_data.append(item)

    train = pd.DataFrame(norm_data)
    test = pd.DataFrame(zero_data)
    test = test.drop(columns=0).drop(columns=1).drop(columns=3)
    x_test = test

    y_train = train[3]
    x_train = train.drop(columns=0).drop(columns=1).drop(columns=3)

    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

    model = neighbors.KNeighborsRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    res = []
    index = 0
    for item in data_set['Balance']:
        if item != 0:
            res.append(item)
        else:
            res.append(y_predict[index])
            index += 1
    data_set['Balance'] = res
    return data_set


def data_basic_clean(data_set):
    """
    数据基础预处理
    :param data_set:
    :return:
    """
    geo = []
    for item in data_set['Geography']:
        if item == 'Spain':
            geo.append(1)
        elif item == 'France':
            geo.append(2)
        else:
            geo.append(0)
    data_set['Geography'] = geo
    return data_set


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(14, 256)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(256, 256)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(256, 256)
        self.relu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(256, 256)
        self.relu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0)
        self.type = 'MLP'

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        out = self.dropout(x)
        return out


def get_bank_dataset():
    data0 = pd.read_csv('BankChurners.csv')
    data = data0.copy()

    data = combine_feature(data_set=data)
    data = predict_balance(data_set=data)

    print(data.isnull().sum())
    print(data.describe())

    data['Tenure^2'] = data['Tenure'] ** 2
    data['Balance^2'] = data['Balance'] ** 2

    data['EstimatedSalary^2'] = data['EstimatedSalary'] ** 2
    data['NumOfProducts^2'] = data['NumOfProducts'] ** 2

    x0 = data.drop(['CreditLevel', 'CustomerId', 'Geography'], axis=1)

    # Feature Scaling / Standard Score
    x0 = (x0 - x0.mean()) / x0.std()

    y0 = data['CreditLevel']
    y0 = y0 - 1  # cater to one-hot

    # balance data
    # smote_nc = SMOTENC(categorical_features=[0, 2, 3, 4, 6], random_state=0)
    # x, y = smote_nc.fit_resample(x0, y0)
    x, y = x0, y0

    x = x.astype('float32')

    x_dataset = torch.from_numpy(x.values)
    y_dataset = torch.from_numpy(y.values)

    dataset = Data.TensorDataset(x_dataset, y_dataset)

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=34)
    return train_dataset, test_dataset


def get_bank_dataloader(train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    return train_loader, test_loader


def create_dataloader():
    data0 = pd.read_csv('BankChurners.csv')
    data = data0.copy()

    print(data.isnull().sum())
    print(data.describe())

    data['Tenure^2'] = data['Tenure'] ** 2
    data['Balance^2'] = data['Balance'] ** 2
    data['EstimatedSalary^2'] = data['EstimatedSalary'] ** 2
    # data['NumOfProducts^2'] = data['NumOfProducts'] ** 2

    x0 = data.drop(['CreditLevel', 'CustomerId', 'Geography'], axis=1)

    # Feature Scaling / Standard Score
    x0 = (x0 - x0.mean()) / x0.std()

    y0 = data['CreditLevel']
    y0 = y0 - 1  # cater to one-hot

    # balance data
    smote_nc = SMOTENC(categorical_features=[0, 2, 3, 4, 6], random_state=0)
    x, y = smote_nc.fit_resample(x0, y0)

    x = x.astype('float32')

    x_dataset = torch.from_numpy(x.values)
    y_dataset = torch.from_numpy(y.values)

    dataset = Data.TensorDataset(x_dataset, y_dataset)

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=34)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    return train_loader, test_loader


def train(train_loader, model, num_epochs):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    correct = 0
    total = 0
    total_step = len(train_loader)
    epoch_loss = []
    for epoch in range(num_epochs):
        batch_loss = []
        for step, (attribute, credit) in enumerate(train_loader):
            # Forward pass
            outputs = model(attribute)
            loss = criterion(outputs, credit.long())

            outputs = model(attribute)
            _, predicted = torch.max(outputs.data, 1)
            total += credit.size(0)
            correct += (predicted == credit).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % total_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
                tst(test_loader, model)
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    # print('Train accuracy is: {} %'.format(100 * correct / total))
    return model.state_dict(), sum(epoch_loss)/len(epoch_loss)


def tst(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for attribute, credit in test_loader:
            outputs = model(attribute)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += credit.size(0)
            correct += (predicted == credit).sum().item()

        print('Accuracy of the network is: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    # step 1: prepare dataset and create dataloader
    train_dataset, test_dataset = get_bank_dataset()
    train_loader, test_loader = get_bank_dataloader(
        train_dataset, test_dataset)
    # train_loader, test_loader = create_dataloader()

    # step 2: instantiate neural network and design model
    model = NeuralNet()

    # step 3: train the model
    train(train_loader, model, num_epochs=75)

    # step 4: test the model
    # tst(test_loader, model)
