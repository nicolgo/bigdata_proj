from deep_learning import NeuralNet, get_bank_dataset, train
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context


def dirichlet_partition(training_data, testing_data, alpha, user_num):
    idxs_train = np.arange(len(training_data))
    idxs_valid = np.arange(len(testing_data))

    labels_train = [label for _, label in training_data]
    labels_valid = [label for _, label in testing_data]
    # if hasattr(training_data, 'targets'):
    #     labels_train = training_data.targets
    #     labels_valid = testing_data.targets
    # elif hasattr(training_data, 'img_label'):
    #     labels_train = training_data.img_label
    #     labels_valid = testing_data.img_label

    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
    idxs_labels_valid = np.vstack((idxs_valid, labels_valid))
    idxs_labels_valid = idxs_labels_valid[:, idxs_labels_valid[1, :].argsort()]

    labels = np.unique(labels_train, axis=0)

    data_train_dict = data_organize(idxs_labels_train, labels)
    data_valid_dict = data_organize(idxs_labels_valid, labels)

    data_partition_profile_train = {}
    data_partition_profile_valid = {}

    for i in range(user_num):
        data_partition_profile_train[i] = []
        data_partition_profile_valid[i] = []

    # Distribute rest data
    for label in data_train_dict:
        proportions = np.random.dirichlet(np.repeat(alpha, user_num))
        proportions_train = len(data_train_dict[label]) * proportions
        proportions_valid = len(data_valid_dict[label]) * proportions

        for user in data_partition_profile_train:
            data_partition_profile_train[user] \
                = set.union(set(np.random.choice(data_train_dict[label], int(proportions_train[user]), replace=False)),
                            data_partition_profile_train[user])
            data_train_dict[label] = list(
                set(data_train_dict[label]) - data_partition_profile_train[user])

            data_partition_profile_valid[user] = set.union(set(
                np.random.choice(data_valid_dict[label], int(proportions_valid[user]),
                                 replace=False)), data_partition_profile_valid[user])
            data_valid_dict[label] = list(
                set(data_valid_dict[label]) - data_partition_profile_valid[user])

        while len(data_train_dict[label]) != 0:
            rest_data = data_train_dict[label][0]
            user = np.random.randint(0, user_num)
            data_partition_profile_train[user].add(rest_data)
            data_train_dict[label].remove(rest_data)

        while len(data_valid_dict[label]) != 0:
            rest_data = data_valid_dict[label][0]
            user = np.random.randint(0, user_num)
            data_partition_profile_valid[user].add(rest_data)
            data_valid_dict[label].remove(rest_data)

    for user in data_partition_profile_train:
        data_partition_profile_train[user] = list(
            data_partition_profile_train[user])
        data_partition_profile_valid[user] = list(
            data_partition_profile_valid[user])
        np.random.shuffle(data_partition_profile_train[user])
        np.random.shuffle(data_partition_profile_valid[user])

    return data_partition_profile_train, data_partition_profile_valid


def bank_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def bank_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 150
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = [label for _, label in dataset]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def data_organize(idxs_labels, labels):
    data_dict = {}

    labels = np.unique(labels, axis=0)
    for one in labels:
        data_dict[one] = []

    for i in range(len(idxs_labels[1, :])):
        data_dict[idxs_labels[1, i]].append(idxs_labels[0, i])
    return data_dict


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), label.clone().detach()


def local_trainer(dataloader, model, local_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for iter in range(local_epoch):
        batch_loss = []
        for batch_idx, (image, labels) in enumerate(dataloader):
            images, labels = image.to(device), labels.to(device)
            model.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('| Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(images),
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def inference(model, testloader, device):
    """ Returns the inference accuracy and loss.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    loss, total, correct = 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    loss /= batch_idx
    accuracy = correct / total
    return accuracy, loss


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.1, 1, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


def draw_distribution(user_num, dataset):
    clients = [i for i in range(user_num)]
    global_dict = {}
    for i in range(user_num):
        client_data = dataset[i]
        lables = [label for _, label in client_data]
        labels_array = np.vstack(lables)
        unique, counts = np.unique(labels_array, return_counts=True)
        count_dict = dict(zip(unique, counts))
        full_dict = {}
        for j in range(10):
            if j not in unique:
                full_dict[j] = 0
            else:
                full_dict[j] = count_dict[j]
        data = np.array(list(full_dict.values()))
        global_dict[i] = data

    survey(global_dict, range(10))
    plt.show()


if __name__ == "__main__":
    # prepare the train dataset
    train_dataset, test_dataset = get_bank_dataset()
    # split the dataset with dirichlet distribution
    user_num = 5
    global_rounds = 150
    local_epochs = 2
    alpha_acc = []
    # alpha = 0.1, Non-IID; alpha = 1000, IID
    # alpha = 0.1
    # train_index, test_index = dirichlet_partition(
    #     train_dataset, test_dataset, alpha=alpha, user_num=user_num)
    train_index = bank_iid(train_dataset, user_num)
    test_index = bank_iid(test_dataset, user_num)

    train_data_list = []
    for user_index in range(user_num):
        train_data_list.append(DatasetSplit(
            train_dataset, train_index[user_index]))
    # draw_distribution(user_num, train_data_list)
    # prepare the test data
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # global_model = LeNet(10).to(device)
    global_model = NeuralNet().to(device)
    global_model.train()
    # start federated learning
    global_loss, global_acc = [], []
    for round_idx in range(global_rounds):
        local_weights, local_losses = [], []
        # global_acc = []
        global_model.train()
        for user_index in range(user_num):
            train_dataloader = DataLoader(
                train_data_list[user_index], batch_size=batch_size, shuffle=True)
            # local train
            model_weights, loss, _ = train(train_dataloader, copy.deepcopy(
                global_model), local_epochs)
            local_weights.append(copy.deepcopy(model_weights))
            local_losses.append(copy.deepcopy(loss))

        global_weight = average_weights(local_weights)
        # update the global weights.
        global_model.load_state_dict(global_weight)

        test_acc, test_loss = inference(
            global_model, test_loader, device=device)
        print('Global Round :{}, the global accuracy is {:.3}%, and the global loss is {:.3}.'.format(
            round_idx, 100 * test_acc, test_loss))
        global_acc.append(test_acc)
        global_loss.append(test_loss)

    # plot the image
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('Loss vs Rounds')
    plt.plot(range(len(global_loss)), global_loss, color='r')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')

    plt.figure()
    plt.title('Acc vs Rounds')
    plt.plot(range(len(global_acc)), global_acc, color='r')
    plt.ylabel('Acc')
    plt.xlabel('Rounds')

    # g_acc, g_loss = inference(
    #         global_model, test_loader, device=device)
    # alpha_acc.append(g_acc)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.title('Acc vs Alpha')
    # plt.plot(np.arange(0.1,2,0.5), alpha_acc, color='r')
    # plt.ylabel('Acc')
    # plt.xlabel('Alpha')
