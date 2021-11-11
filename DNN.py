import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.utils.data as Data


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(7, 100)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 3)
        self.type = 'MLP'

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


def create_dataloader():

    data = pd.read_csv('../BankChurners.csv')

    # remove balance with value 0
    '''
    data = data[data['Balance'] != 0]
    '''

    # train 5-7 respectively
    '''
    data = data[(data['CreditLevel'] >= 5) & (data['CreditLevel'] <= 7)]
    '''

    x = data.drop(['CreditLevel', 'CustomerId', 'Geography'], axis=1)
    y = data['CreditLevel']

    # 5-7 cater to one hot
    '''
    y[y == 5] = 1
    y[y == 6] = 2
    y[y == 7] = 3
    '''

    y[(y >= 1) & (y <= 4)] = 1
    y[(y >= 5) & (y <= 7)] = 2
    y[(y >= 8) & (y <= 10)] = 3
    y = y-1  # To cater to one hot
    x = x.astype('float32')
    x_dataset = torch.from_numpy(x.values)
    y_dataset = torch.from_numpy(y.values)
    dataset = Data.TensorDataset(x_dataset, y_dataset)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False)

    return train_loader, test_loader


def train(train_loader, model, criterion, optimizer, num_epochs):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for step, (attribute, credit) in enumerate(train_loader):

            # Forward pass
            outputs = model(attribute)
            loss = criterion(outputs, credit.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))


def tst(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for attribute, credit in test_loader:
            outputs = model(attribute)
            _, predicted = torch.max(outputs.data, 1)
            total += credit.size(0)
            correct += (predicted == credit).sum().item()

        print('Accuracy of the network is: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    ### step 1: prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    ### step 2: instantiate neural network and design model
    model = NeuralNet()
    # model = ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ### step 3: train the model
    train(train_loader, model, criterion, optimizer, num_epochs=5)

    ### step 4: test the model
    tst(test_loader, model)

