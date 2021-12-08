# Big Data Project
## 1 Introduction
The project of big data class.
### 1.1 Tasks
1. Design a deep learning model to predict the credit level of each customer.
2. Distribute the whole dataset into 5 different parties in independent and identical distribution and then adapt a federated learning framework to do model training and aggregation. Show the superiority of collaborative training over individual local training.
### 1.2 Dataset
`BankChurners.csv` contains basic information of 9000 bank’s customers and the target variable is the credit level between 1(bad) to 10(excellent):
- **CustomerId** – unique Ids for bank customer identification.
- **Geography** – the country from which the customer belongs.
- **Tenure** – number of years for which the customer has been with the bank.
- **Balance** – bank balance of the customer.
- **NumOfProducts** – number of bank products the customer is utilizing.
- **HasCrCard** – binary flag for whether the customer holds a credit card or not.
- **IsActiveMember** – binary flag for whether the customer is an active member or not.
- **EstimatedSalary** – estimated salary of the customer in Dollars.
- **Exited** – binary flag 1 if the customer closed an account with the bank and 0 if the customer is retained.
- **CreditLevel** – credit level of the customer

`New_BankChurners.csv` contains basic information of 1000 new bank’s customers and the credit level is unknown.

### 1.3 File Structure
- `data_preprocessing.ipynb`: data preprocessing
- `deep_learning.py`: training with one DNN model,accuracy:about 20%
- `deep_learning_subclass.py`: In this file, we use one subclass model and three clarifiers to train, accuracy about 21.6%
- `ml_with_nn.py`: deep learning + machine learning, 37.22%
- `federated_learning.py`: federated learning module, include iid and non-iid, FedAvg and so on.
- `federated_learning_noiid`: improvements of FL with non-iid data.

## 2 Data Processing/Analytics
The procedure of data processing and analytics can be divided into three parts, including exploratory data analysis, data preprocessing and feature engineering. Details can be seen at `data_processing.ipynb` 

## 3 Model design and implementation
### 3.1 training with one model
### 3.2 1 subclass model + 3 classifiers
The implementation can be seen at `deep_learning_subclass.py` file
### 3.3 deep learning + machine learning
The implementation can be seen at `ml_with_nn.py` file
## 4 Framework of federated learning
Federated Learning process
- divide dataset by IID or non-IID
- foreach round:
     - training on each clients with model and save weights
     - update the model with avg_weig
- predict
### 4.1 Data partition
- Steps to generate result
uncomment the `draw_distribution` function in `federated_learning.py`
1. get IID: using `bank_iid` function or set α to 1000 use `dirichlet_partition` to get dataset
2. get non-IID: set α to 0.1 and use `dirichlet_partition` to get dataset

   run `federated_learning.py`
- IID data
divided dataset randomly
![IID_simple](https://user-images.githubusercontent.com/17155788/145030135-27e57ff4-405b-4731-a830-ff68e98696fa.png)
setting α to 1000
![image](https://user-images.githubusercontent.com/17155788/145030421-3f2516d1-0af6-4d08-b475-6bfa0043831b.png)

- non-IID data(setting α to 0.1)
![image](https://user-images.githubusercontent.com/17155788/145030468-02689710-6bc5-4637-8158-bf1729085b68.png)

### 4.2 Federated Learning Result
- IID result(`federated_learning.py`)
![image](https://user-images.githubusercontent.com/17155788/145030554-a045ac0a-2efd-4969-b4c4-144c5423be56.png)

- non-IID result(`federated_learning.py`) 
![fl_non](https://user-images.githubusercontent.com/17155788/145030613-1122ec3a-d05b-40dd-bc14-464651a07417.png)

- improved non-IID result(`federated_learning_iid.py`)
![non_iid_improve](https://user-images.githubusercontent.com/17155788/145030630-bd6350c6-53ad-4647-88fd-c2cf0f8a2e5f.png)

