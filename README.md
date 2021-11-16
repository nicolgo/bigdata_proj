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
- `deep_learning.py` (accurancy:40%)
- `machine_learning.py`(accurancy:40%)
- `federated_learning.py` (accurancy:46% or above,using deep_learning model)

## 2 Data preprocessing/analytics

## 3 Model design and implementation

## 4 Framework of federated learning
Federated Learning is simply the decentralized form of Machine Learning.
### 4.1 Federated Learning Process

### 4.2 Federated Learning Result

## 5 Summary

## 6 Reference

