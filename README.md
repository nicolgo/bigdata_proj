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
Limited by data sources and data acquisition methods, data obtained in real life is difficult to deal with because of missing values and outliers. Therefore, to ensure the assumptions of models optimize the model effect, data preprocessing is an essential part before building models.  
In view of the problem to be studied, there are three parts.  
First of all, exploratory data analysis is required to have a general comprehension of data structure of the issue, including the distribution of data, correlation between features and response variable and even some inspiration to model selection.   
Secondly, data preprocessing is to deal with the missing values and outliers to avoid negative influence on model construction and result prediction.   
Finally, features engineering contains many ways to keep and find features most relavent to the dependent variables, like feature selection, feature transformation and so on. In the case, since the given features are not efficient, feature construction is the main method chosen.  
### 2.1 Exploratory Data Analysis
By observing the features of features and response variable, feature CustomerId obviously makes no difference to CreditLevel. Thus, we drop feature CustomerId in the following analysis.  
<img width="874" alt="截屏2021-11-24 上午12 38 50" src="https://user-images.githubusercontent.com/93926853/143077735-97826a64-3b8f-4d21-98e9-1f295aed714d.png">  
  
There is no missing values.  
  <img width="121" alt="截屏2021-11-24 上午2 12 38" src="https://user-images.githubusercontent.com/93926853/143080945-ef7fe092-adca-47cc-aadd-1a0cb270d5ce.png">  

  
The distribution of response variable  
<img width="605" alt="截屏2021-11-24 上午12 39 52" src="https://user-images.githubusercontent.com/93926853/143079198-51c02f6a-9fdd-49b4-86b2-96ecdaef8fcb.png">  
Most customers belong to middle class of credit level, between 4 and 8. The sample size of each credit level class is not equal.  
  
Distributions of continuous variables  
<img width="496" alt="截屏2021-11-24 上午1 06 26" src="https://user-images.githubusercontent.com/93926853/143079301-0d606c1b-be97-4a47-bfd0-d57858b94dcb.png">  
  
Categorical variables  
<img width="653" alt="截屏2021-11-24 上午2 04 29" src="https://user-images.githubusercontent.com/93926853/143079796-f2741a8f-47d8-4d20-8c6b-ad8839e452c6.png">
  
The correlaiton of features and response variable  
<img width="495" alt="截屏2021-11-24 上午1 48 48" src="https://user-images.githubusercontent.com/93926853/143080619-46f88c9f-53dd-46e8-9c71-caa0151cf7bb.png">  
  
Above all, it is obvious that features have weak correlation with the dependent variable. The sample size of each class level of credit is not the same. Furthermore, there are many customers with 0 balance in the dataset, which does not make sense.    



### 2.2 Data Preprocessing

### 2.3 Feature Engineering

## 3 Model design and implementation

## 4 Framework of federated learning
Federated Learning is simply the decentralized form of Machine Learning.
### 4.1 Federated Learning Process

### 4.2 Federated Learning Result

## 5 Summary

## 6 Reference

