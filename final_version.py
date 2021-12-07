from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import datasets
from xgboost.sklearn import XGBClassifier
import pydotplus
from IPython.display import Image
import pandas as pd

from bigdata.deep_learning import train_model_outer, predict_test_class, predict_output

# 当拓展特征时需要修改此参数 每加一个特征此常量++
drop_col_index = 8 + 3 - 3


def read_data(file_name='BankChurners.csv'):
    """
    读取csv文件数据 数据大小为9000条
    :return: pandas格式数据
    """
    return pd.read_csv(file_name)


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

    model = linear_model.LinearRegression()

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


def predict_balance_output(data_set):
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
    test = test.drop(columns=0).drop(columns=1).drop(columns=3).drop(columns=9)
    x_test = test

    y_train = train[3]
    x_train = train.drop(columns=0).drop(columns=1).drop(columns=3).drop(columns=9)

    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

    model = linear_model.LinearRegression()

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
    # data_set['NewFeature1'] = feat1
    # data_set['NewFeature2'] = feat2
    # data_set['NewFeature3'] = feat3
    return data_set


def data_basic_clean(data_set):
    """
    数据基础预处理
    :param data_set:
    :return:
    """
    data_set = data_set.drop(columns='CustomerId')
    geo = []
    for item in data_set['Geography']:
        if item == 'Spain':
            geo.append(1)
        elif item == 'France':
            geo.append(2)
        else:
            geo.append(0)
    data_set['Geography'] = geo
    res_col = data_set['CreditLevel']
    data_set = data_set.drop(columns='CreditLevel')

    data_set = (data_set - data_set.min()) / (data_set.max() - data_set.min())
    data_set['CreditLevel'] = res_col
    return data_set


def data_split(data_set, ratio=0.2):
    """
    数据划分
    :param ratio: 测试集占比
    :param data_set: 原数据集（全）
    :return: train_data_set test_data_set
    """
    train_data_set = data_set[0: int(len(data_set) * (1 - ratio))]
    test_data_set = data_set[int(len(data_set) * (1 - ratio)):]
    return train_data_set, test_data_set


def pick_credit_data_level(data_set):
    """
    将数据以credit分为三个等级 并加新列 c_level [0 1 2]
    level 1 : 1 2 3
    level 2 : 4 5 6 7 8
    level 3 : 9 10
    :param data_set:
    :return: data_set
    """
    c_level = []
    for item in data_set['CreditLevel']:
        if 9 <= int(item) <= 10:
            c_level.append(2)
        elif 4 <= int(item) <= 8:
            c_level.append(1)
        else:
            c_level.append(0)
    data_set['c_level'] = c_level
    return data_set


def fit_level_classifier(data_set):
    """
    训练3 level分类
    xgboost 算法
    :param data_set: 数据集
    :return: 模型 model
    """
    x_train = data_set.drop("c_level", axis=1).drop(columns='CreditLevel')
    y_train = data_set["c_level"]

    model = XGBClassifier(
        # 树的个数
        n_estimators=100, learning_rate=0.07, max_depth=12, subsample=1, seed=1000,
        # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
        gamma=0,
        # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        reg_lambda=1,
        # 最大增量步长，我们允许每个树的权重估计。
        max_delta_step=0,
        # 生成树时进行的列采样
        colsample_bytree=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 过拟合。
        min_child_weight=1)

    model.fit(x_train, y_train, eval_metric='auc')
    # model.fit(x_train, y_train)
    return model


def test_level_classification(model, test_data):
    """
    测试分类预测 accuracy
    :param model: 训练后的模型
    :param test_data: 测试集
    """
    x_test = test_data.drop("c_level", axis=1).drop(columns='CreditLevel')
    y_test = test_data["c_level"]

    y_predict = model.predict(x_test)
    print("level classifier accuracy: ", accuracy_score(y_test, y_predict))


def predict_level_classification(model, test_data):
    """
    返回分层预测结果
    :param model: 训练后的模型
    :param test_data:
    :return:
    """
    x_test = test_data.drop(columns='CreditLevel')
    y_predict = model.predict(x_test)
    test_data['c_level'] = y_predict
    return test_data


def divide_dataset_2_levels(data_set):
    """
    将集根据level分成三份
    :param data_set: 训练数据集 / 结果集
    :return: 三个level区分的数据集
    """
    c1_dataset = []
    c2_dataset = []
    c3_dataset = []
    for index, item in enumerate(data_set.values):
        if item[-1] == 0:
            c1_dataset.append(item)
        elif item[-1] == 1:
            c2_dataset.append(item)
        else:
            c3_dataset.append(item)
    c1_dataset = pd.DataFrame(c1_dataset)
    c2_dataset = pd.DataFrame(c2_dataset)
    c3_dataset = pd.DataFrame(c3_dataset)
    return c1_dataset, c2_dataset, c3_dataset


def divide_dataset_2_levels_test(data_set):
    """
    将集根据level分成三份
    :param data_set: 训练数据集 / 结果集
    :return: 三个level区分的数据集
    """
    c1_dataset = []
    c2_dataset = []
    c3_dataset = []
    i1, i2, i3 = 0, 0, 0
    dic = {}
    for index, item in enumerate(data_set.values):
        if item[-1] == 0:
            dic['1_' + str(i1)] = index
            i1 += 1
            c1_dataset.append(item)
        elif item[-1] == 1:
            dic['2_' + str(i2)] = index
            i2 += 1
            c2_dataset.append(item)
        else:
            dic['3_' + str(i3)] = index
            i3 += 1
            c3_dataset.append(item)
    c1_dataset = pd.DataFrame(c1_dataset)
    c2_dataset = pd.DataFrame(c2_dataset)
    c3_dataset = pd.DataFrame(c3_dataset)
    return c1_dataset, c2_dataset, c3_dataset, dic


def fit_sub_level(model, train_data_set, xgb=False, l2=False):
    """
    训练子类数据集
    :param l2:
    :param xgb:
    :param model: 模型
    :param train_data_set: 训练集
    :return:
    """
    y_train = train_data_set[drop_col_index]
    x_train = train_data_set.drop(columns=drop_col_index).drop(columns=drop_col_index + 1)
    if l2:
        from imblearn.over_sampling import SVMSMOTE
        smote_nc = SVMSMOTE(sampling_strategy={4: 2000, 5: 2000, 6: 2000, 7: 2000, 8: 2000}, random_state=0)
        x_train, y_train = smote_nc.fit_resample(x_train, y_train)
    if xgb:
        model.fit(x_train, y_train, eval_metric='auc')
    else:
        model.fit(x_train, y_train)
    return model


def predict_sub_credit(model, test_data):
    """
    预测子类别结果
    :param model: 训练后的模型
    :param test_data: 测试数据
    :return: 预测的credit
    """
    return model.predict(test_data)


def collect_data(data):
    res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for item in data:
        if item == 4:
            res[3] += 1
        elif item == 5:
            res[4] += 1
        elif item == 6:
            res[5] += 1
        elif item == 7:
            res[6] += 1
        elif item == 8:
            res[7] += 1
    print(res)
    return res


if __name__ == '__main__':
    d_set = read_data()
    d_set = predict_balance(data_set=d_set)
    d_set = combine_feature(data_set=d_set)
    d_set = data_basic_clean(data_set=d_set)

    d_train_set, d_test_set = data_split(data_set=d_set, ratio=0.1)

    d_train_set = pick_credit_data_level(data_set=d_train_set)

    # l_model = fit_level_classifier(data_set=d_train_set)
    # nn training
    l_model, test_set = train_model_outer()

    l1_dataset, l2_dataset, l3_dataset = divide_dataset_2_levels(data_set=d_train_set)

    # 3 sub-class model init
    l1_model = RandomForestClassifier(n_estimators=400)
    l2_model = RandomForestClassifier(n_estimators=600)
    # l2_model = LogisticRegression()
    # l2_model = LinearSVC()
    l3_model = RandomForestClassifier(n_estimators=400)

    # train
    l1_model = fit_sub_level(model=l1_model, train_data_set=l1_dataset)
    l2_model = fit_sub_level(model=l2_model, train_data_set=l2_dataset, l2=False)
    l3_model = fit_sub_level(model=l3_model, train_data_set=l3_dataset)

    # Estimators = l2_model.estimators_
    # for index, model in enumerate(Estimators):
    #     filename = 'iris_' + str(index) + '.pdf'
    #     dot_data = tree.export_graphviz(model, out_file=None,
    #                                     feature_names=['geo',
    #                                     'tenure', 'balance',
    #                                     'NumOfProducts', 'HasCrCard',
    #                                     'IsActiveMember', 'EstimatedSalary',
    #                                     'Exited'],
    #                                     class_names=['4', '5', '6', '7', '8'],
    #                                     filled=True, rounded=True,
    #                                     special_characters=True)
    #     graph = pydotplus.graph_from_dot_data(dot_data)
    #     # 使用ipython的终端jupyter notebook显示。
    #     Image(graph.create_png())
    #     graph.write_pdf(filename)


    d_test_set['c_level'] = predict_test_class(l_model, test_set)
    t_l1_dataset, t_l2_dataset, t_l3_dataset, dic = divide_dataset_2_levels_test(data_set=d_test_set)

    ################################################################################################
    right_number = 0

    if not t_l1_dataset.empty:
        t_l1_credit_answer = t_l1_dataset[drop_col_index]
        t_l1_dataset = t_l1_dataset.drop(columns=drop_col_index).drop(columns=drop_col_index + 1)
        predict_l1 = predict_sub_credit(model=l1_model, test_data=t_l1_dataset)
        for i in range(0, len(t_l1_credit_answer)):
            if int(t_l1_credit_answer[i]) == int(predict_l1[i]):
                right_number += 1

    data_res = []
    if not t_l2_dataset.empty:
        t_l2_credit_answer = t_l2_dataset[drop_col_index]
        t_l2_dataset = t_l2_dataset.drop(columns=drop_col_index).drop(columns=drop_col_index + 1)
        predict_l2 = predict_sub_credit(model=l2_model, test_data=t_l2_dataset)
        for i in range(0, len(t_l2_credit_answer)):
            data_res.append(int(predict_l2[i]))
            # print(str(t_l2_credit_answer[i]) + " " + str(predict_l2[i]))
            if int(t_l2_credit_answer[i]) == int(predict_l2[i]):
                right_number += 1

    if not t_l3_dataset.empty:
        t_l3_credit_answer = t_l3_dataset[drop_col_index]
        t_l3_dataset = t_l3_dataset.drop(columns=drop_col_index).drop(columns=drop_col_index + 1)
        predict_l3 = predict_sub_credit(model=l3_model, test_data=t_l3_dataset)
        for i in range(0, len(t_l3_credit_answer)):
            if int(t_l3_credit_answer[i]) == int(predict_l3[i]):
                right_number += 1

    sum_number = len(d_test_set)
    print("Accuracy for all data: ", right_number / sum_number)

