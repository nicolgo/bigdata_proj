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
- `deep_learning.py` (accuracy:40%)
- `machine_learning.py`(accuracy:40%)
- `federated_learning.py` (accuracy:46% or above,using deep_learning model)

## 2 Data preprocessing/analytics
The procedure of data processing and analytics can be divided into three parts, including exploratory data analysis, data preprocessing and feature engineering, which is stated as follows.  
### 2.1 Exploratory Data Analysis
By processing and observing the training data BankChurners.csv, it’s clear to show that there’s no missing values and the variable CustomerId contribute nothing to the result CreditLevel in the dataset. Then we drop the whole column of CustomerId. 
<img width="874" alt="截屏2021-11-24 上午12 38 50" src="https://user-images.githubusercontent.com/93926853/143077735-97826a64-3b8f-4d21-98e9-1f295aed714d.png">  
  
There is no missing value, and weak correlation between features and imbalanced sample size of credit level appears according to the drawing subplot of features. Furthermore, many values for the variable Balance is zero in the dataset.   



### 2.2 Data Preprocessing
As Balance is regarded as one of the most important variable for predicting the credit level, values for this feature is hardly possible equal to zero. We use linear regression model to fill up zero values in Balance. We change the feature Geography into dummy variables since it only shows different regions and has no meaning in our model. 
  <img width="559" alt="截屏2021-11-24 上午2 45 25" src="https://user-images.githubusercontent.com/93926853/143085135-94ac6712-f413-419f-8d66-37308fc9b9e9.png">  

### 2.3 Feature Engineering
After CustomerId is dropped, there’re only 8 explanatory features in total. Since the remaining explanatory features have less correlation with CreditLevel, we combine some features together to improve the model prediction accuracy and make it more convincing. 
## 3 Model design and implementation

## 4 Framework of federated learning
Federated Learning is simply the decentralized form of Machine Learning.
### 4.1 Federated Learning Process
- divide dataset by IID or non-IID
- foreach round:
     - training on each clients with model and save weights
     - update the model with avg_weig
- predict
### 4.2 Federated Learning Result
- IID distribution on 3-output and 10-output model
![image](https://user-images.githubusercontent.com/17155788/143258673-c522854c-ea32-4b72-b0e4-6461b1d101e0.png)
- Dirichlet distribution with a=0.1 and a=5 
![image](https://user-images.githubusercontent.com/17155788/143258957-2ce65fe7-d6d0-4684-bf33-144a364e8a38.png)

## 5 Summary

## 6 Reference

