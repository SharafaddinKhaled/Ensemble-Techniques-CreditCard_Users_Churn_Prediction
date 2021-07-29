#!/usr/bin/env python
# coding: utf-8

# ## Khaled Sharafaddin - CreditCard Users Churn Prediction
# 
# Description
# Background & Context
# 
# The Thera bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged on every user irrespective of usage, while others are charged under specified circumstances.
# 
# Customers’ leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers’ and identify the customers who will leave their credit card services and reason for same – so that bank could improve upon those areas
# 
# You as a Data scientist at Thera bank need to come up with a classification model that will help bank improve their services so that customers do not renounce their credit cards
# 
# Objective
# 
# Explore and visualize the dataset.
# Build a classification model to predict if the customer is going to churn or not
# Optimize the model using appropriate techniques
# Generate a set of insights and recommendations that will help the bank
# Data Dictionary:
# * CLIENTNUM: Client number. Unique identifier for the customer holding the account
# * Attrition_Flag: Internal event (customer activity) variable - if the account is closed then 1 else 0
# * Customer_Age: Age in Years
# * Gender: Gender of the account holder
# * Dependent_count: Number of dependents
# * Education_Level: Educational Qualification of the account holder
# * Marital_Status: Marital Status of the account holder
# * Income_Category: Annual Income Category of the account holder
# * Card_Category: Type of Card
# * Months_on_book: Period of relationship with the bank
# * Total_Relationship_Count: Total no. of products held by the customer
# * Months_Inactive_12_mon: No. of months inactive in the last 12 months
# * Contacts_Count_12_mon: No. of Contacts in the last 12 months
# * Credit_Limit: Credit Limit on the Credit Card
# * Total_Revolving_Bal: Total Revolving Balance on the Credit Card
# * Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months)
# * Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
# * Total_Trans_Amt: Total Transaction Amount (Last 12 months)
# * Total_Trans_Ct: Total Transaction Count (Last 12 months)
# * Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
# * Avg_Utilization_Ratio: Average Card Utilization Ratio
# 
# 
# 

# ### 1. Load libraries and view the dataset

# In[1]:


# This allows each cell to output everything instead of just the last command
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# 1. Load libraries and dataset
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import metrics
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from os import system 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# 2. Look at the data
BankChurners = pd.read_csv('/Users/khaledsharafaddin/Documents/Univ_Austin_Texas ML_AI/DataSets/BankChurners.csv')
BankChurners.head()
BankChurners.info()

# 3. Check the total null (only Unnamed21 column) and the shape of the data
BankChurners.isnull().sum()/BankChurners.shape[0]*100
BankChurners.shape   # (10127, 22)


# ### 2. Exploratory Data Analysis

# In[2]:


# 2.1 Look at percentages and value counts

cat = ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 
             'Card_Category']
for i in cat:
    print(i)
    BankChurners[i].value_counts()/BankChurners.shape[0]*100


# In[3]:


# 2.2 Bivariate Analysis

# Boxplot for Attrition_Flag and numeric variables to see relationships

# Credit_Limit
print('Credit_Limit')
sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Credit_Limit", data=BankChurners, orient="vertical")
plt.show()

#  Avg_Utilization_Ratio
print('Avg_Utilization_Ratio')
sns.boxplot(x="Attrition_Flag", y="Avg_Utilization_Ratio", data=BankChurners, orient="vertical")
plt.show()


print('Months_Inactive_12_mon')
sns.boxplot(x="Attrition_Flag", y="Months_Inactive_12_mon", data=BankChurners, orient="vertical")
plt.show()

# Total_Revolving_Bal
print('Total_Revolving_Bal')
sns.boxplot(x="Attrition_Flag", y="Total_Revolving_Bal", data=BankChurners, orient="vertical")
plt.show()


# In[4]:


# 2.3 Relationships between Attrition flag and categorical variables:

## Function to plot stacked bar chart
def stacked_plot(x):
    sns.set(palette="nipy_spectral")
    tab1 = pd.crosstab(x, BankChurners["Attrition_Flag"], margins=True)
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(x, BankChurners["Attrition_Flag"], normalize="index")
    tab.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.show()
    
stacked_plot(BankChurners["Income_Category"])

stacked_plot(BankChurners["Card_Category"])


# 2.4 Correlation R_square
BankChurners.corr()

# Attrition_Flag is imbalanced:
sns.countplot(x='Attrition_Flag', data=BankChurners, palette='hls')
plt.show()

# Age historgram
BankChurners.Customer_Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')


# ### EAD Insights:
# - 83.9% of customers are Existing customers, while only 16% are attrited customers. There is a high level of class imbalance
# - The gender category is balanced 
# - Approx. 50% of customers have attended college or higher education, while only 28% are either uneducated or unknown
# - The majority of customers are either married or single
# - 52% of customers make 60,000 dollars or less, while small number make higher than 120,000 dollars in income
# - 93% of customers are in the Blue card category. 
# - Credit limit is slightely lower for attrited customers than existing ones.
# - the Average untilization ratio for existing customers is higher than attrited ones.
# - One year month inactive for existing customers are lower than attrited ones. 
# - total revolving balance for existing customers are higher than attrited ones.
# - We can see that 
# - It seems that there is not a statistically significant relationship between income and attrition category. However, most of the attritied customers have platinum cards. 
# - There is a strong correlation of 79% between customer age and months on book. 
# - There is also  a very strong correlation of 99% between credit limit and Open to Buy Credit Line
# - Customer_Age is uniform and symmetric. 
# 

# ### 3. Data Cleaning and Preperation

# In[5]:


# 1. Remove CLIENTNUM and Unnamed: 21 from the dataset since they are not useful in predictions
BankChurners = BankChurners.drop(['CLIENTNUM','Unnamed: 21'], axis=1)

# 2. Convert Object categories into categorical type
BankChurners[['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 
             'Card_Category']] = BankChurners[['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 
             'Card_Category']].astype('category')


# 3. Hot-Encode caterogical variables 

# Gender, Attrition_Flag, Marital_Status will be hot-encoded since the rank might not be important
# Education_Level, Income_Category, Card_Category will be replaced by numbers since rank is important 

BankChurners = BankChurners.replace({'Education_Level':{'Doctorate':0,'Graduate':1, 'Post-Graduate':2,
'College':3, 'High School':4, 'Unknown':5, 'Uneducated':6},
'Income_Category':{'Less than $40K':0, '$40K - $60K':1, '$80K - $120K':2, '$60K - $80K':3, 'Unknown':4,'$120K +':5},
'Card_Category':{'Blue':0, 'Silver':1, 'Platinum':2,'Gold':3},
'Attrition_Flag':{'Existing Customer':0, 'Attrited Customer':1}})

# 4. These categories are not ordered so we can do dummy variables for them
oneHotCols=['Gender', 'Marital_Status']
BankChurners = pd.get_dummies(BankChurners, columns=oneHotCols)


# ### 4. Split the Dataset

# In[6]:


# Split Data into training and testing

X = BankChurners.drop(["Attrition_Flag"], axis=1)
y = BankChurners["Attrition_Flag"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1, stratify=y
)
print(X_train.shape, X_test.shape)


# ### 5. Over-Sample The Attrition_Flag using SMOTE
# - SMOTE algorithm(Synthetic Minority Oversampling Technique). At a high level, SMOTE:
# - Creates synthetic samples from the minor class (Attrited == 1) 
# - Randomly choose one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations.

# In[7]:


from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number records X_train dataset: ", X_train.shape)
print("Number records y_train dataset: ", y_train.shape)
print("Number records X_test dataset: ", X_test.shape)
print("Number records y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

print("% Before OverSampling, counts of label '1': {}".format(sum(y_train==1)/len(y_train)))
print("% Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)/len(y_train)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("% After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)/len(y_train)))
print("% After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)/len(y_train)))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[8]:


#  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model, flag=True):
    """
    model : classifier to predict values of X

    """
    # defining an empty list to store train and test results
    score_list = []

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_recall = metrics.recall_score(y_train, pred_train)
    test_recall = metrics.recall_score(y_test, pred_test)

    train_precision = metrics.precision_score(y_train, pred_train)
    test_precision = metrics.precision_score(y_test, pred_test)

    score_list.extend(
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
        )
    )

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ", model.score(X_train, y_train))
        print("Accuracy on test set : ", model.score(X_test, y_test))
        print("Recall on training set : ", metrics.recall_score(y_train, pred_train))
        print("Recall on test set : ", metrics.recall_score(y_test, pred_test))
        print(
            "Precision on training set : ", metrics.precision_score(y_train, pred_train)
        )
        print("Precision on test set : ", metrics.precision_score(y_test, pred_test))

    return score_list  # returning the list with train and test scores


## Function to create confusion matrix
def make_confusion_matrix(model, y_actual, labels=[1, 0]):
    """
    model : classifier to predict values of X
    y_actual : ground truth

    """
    y_predict = model.predict(X_test)
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0, 1])
    data_cm = pd.DataFrame(
        cm,
        index=[i for i in ["Actual - No", "Actual - Yes"]],
        columns=[i for i in ["Predicted - No", "Predicted - Yes"]],
    )
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(10, 7))
    sns.heatmap(data_cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### which metric is right for model? 
# - We are interested in finding customers who may churn from using credit cards.
# - The dataset has class imbalance. Approx. 84% are existing customers, and therefore accuracy is not a good measure.
# - Precision is a good measure here, because The Thera bank should aim at  finding customers who might churn.

# ### 1. Logistic Regression with GridSearchCV:

# In[10]:


log_pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))

param_grid ={"logisticregression__C":np.arange(0.001, 1, 0.01),
      "logisticregression__penalty":["l1","l2"]} # l1 lasso l2 ridge

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

log_reg_cv = GridSearchCV(estimator=log_pipe, param_grid=param_grid,scoring=scorer ,cv=10)
log_reg_cv.fit(X_train_res, y_train_res)

# Best Hyperparameter for log_reg
log_reg_best_params = log_reg_cv.best_params_
print('Tuned Hyperparameters for logistic Regression: ', log_reg_cv.best_params_)

log_reg_tuned = make_pipeline(StandardScaler(),LogisticRegression(random_state=1,C= 0.160, penalty='l2')) 
log_reg_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(log_reg_tuned)
make_confusion_matrix(log_reg_tuned, y_test)


# ### 2. Logistic Regression with RandomizedSearchCV

# In[11]:



from scipy.stats import uniform

rand_logistic_pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
param_grid = {
    'logisticregression__C': uniform(loc=0, scale=4), # regularization hyperparameter distribution using uniform distribution
    'logisticregression__penalty':['l1', 'l2']
}

# Create RandomizedSearchCV
log_clf = RandomizedSearchCV(rand_logistic_pipe, param_grid, n_iter=100, cv=10)
log_clf.fit(X_train_res, y_train_res)

# best parameters:
print('Best hyperparameters for the logistic regression: ', log_clf.best_params_)

rand_logistic_tuned_pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, C=2.5792518191315543, penalty= 'l2'))
rand_logistic_tuned_pipe.fit(X_train_res, y_train_res)

# Calculating different metrics
get_metrics_score(rand_logistic_tuned_pipe)
# Creating confusion matrix
make_confusion_matrix(rand_logistic_tuned_pipe, y_test)



# ### 3. Decision tree using Pipeline and GridSearchCV 

# In[12]:



Dtree_pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=1))

parameters = {'decisiontreeclassifier__max_depth': np.arange(1,10), 
              'decisiontreeclassifier__min_samples_leaf': [1, 2, 5, 7, 10,15,20],
              'decisiontreeclassifier__max_leaf_nodes' : [2, 3, 5, 10],
              'decisiontreeclassifier__min_impurity_decrease': [0.001,0.01,0.1]
             }

scorer = metrics.make_scorer(metrics.precision_score)

Dtree_cv = GridSearchCV(estimator=Dtree_pipe, param_grid=parameters,scoring=scorer ,cv=10)
Dtree_cv.fit(X_train_res, y_train_res)


# Best Hyperparameter for DTree
Dtree_best_params = Dtree_cv.best_params_
estimator = Dtree_cv.best_estimator_
print('Tuned Hyperparameters for DTree Classifier: ', estimator)

# Tune the model with best estimator
Dtree_tuned = make_pipeline(StandardScaler(),estimator) 
Dtree_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(Dtree_tuned)
make_confusion_matrix(Dtree_tuned, y_test)



# ### 4. Decision tree using Pipeline and GridSearchCV 

# In[13]:



from scipy.stats import randint
rand_Dtree_pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=1))

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"decisiontreeclassifier__max_depth": [3, None],
              "decisiontreeclassifier__max_features": randint(1, 9),
              "decisiontreeclassifier__min_samples_leaf": randint(1, 9),
              "decisiontreeclassifier__criterion": ["gini", "entropy"]}

scorer = metrics.make_scorer(metrics.precision_score)

rand_Dtree_cv = RandomizedSearchCV(rand_Dtree_pipe, param_dist, n_iter=50, cv=10)

rand_Dtree_cv.fit(X_train_res, y_train_res)


# Best Hyperparameter for DTree
rand_Dtree_best_params = rand_Dtree_cv.best_estimator_
print('Tuned Hyperparameters for RandomizedSearchCV DTree Classifier: ', rand_Dtree_best_params)

# Tune the model with best estimator
rand_Dtree_tuned = make_pipeline(StandardScaler(),rand_Dtree_best_params) 
rand_Dtree_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(rand_Dtree_tuned)
make_confusion_matrix(rand_Dtree_tuned, y_test)


# ### 5. Random Forest with pipeline and GridSearchCV

# In[14]:



rf_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1))

# Must include RandomForestClassifier__ before parameter names when doing gridsearch cv or randomizedcv
parameters = {"randomforestclassifier__n_estimators": [150,200,250],
    "randomforestclassifier__min_samples_leaf": np.arange(5, 10),
    "randomforestclassifier__max_features": np.arange(0.2, 0.7, 0.1)}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.precision_score)

# Run the grid search
rf_grid_obj = GridSearchCV(rf_pipe, parameters, scoring=acc_scorer,cv=5)
rf_grid_obj = rf_grid_obj.fit(X_train_res, y_train_res)

# Set the clf to the best combination of parameters
rf_estimator = rf_grid_obj.best_estimator_

print('Tuned Hyperparameters for random forest Classifier: ', rf_estimator)

# Tune the model with best estimator
rf_tuned = make_pipeline(StandardScaler(),rf_estimator) 
rf_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(rf_tuned)
make_confusion_matrix(rf_tuned, y_test)



# ### Random Forest Feature Importance

# In[67]:


# rf_estimator feature importance

feature_importances = rf_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 6. Random Forest with pipeline and RandomizedSearchCV

# In[15]:



from scipy.stats import truncnorm, uniform

rand_rf_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1))

# This must be a distribution, not random
model_params = {
    # randomly sample numbers from 4 to 204 estimators
    'randomforestclassifier__n_estimators': randint(4,200),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'randomforestclassifier__max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'randomforestclassifier__min_samples_split': uniform(0.01, 0.199)
}

# Type of scoring used to compare parameter combinations
acc_scorer =  metrics.make_scorer(metrics.precision_score)

# Run the grid search
rand_rf = RandomizedSearchCV(rand_rf_pipe, model_params, scoring=acc_scorer,n_iter=50, cv=5)
rand_rf = rand_rf.fit(X_train_res, y_train_res)

# Set the clf to the best combination of parameters
rand_rf_estimator = rand_rf.best_estimator_

print('Tuned Hyperparameters for random forest Classifier: ', rand_rf_estimator)

# Tune the model with best estimator
rand_rf_tuned = make_pipeline(StandardScaler(),rand_rf_estimator) 
rand_rf_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(rand_rf_tuned)
make_confusion_matrix(rand_rf_tuned, y_test)


# ### RandomizedSearchCV Random Forest Feature Importance

# In[90]:


# rand_rf_estimator feature importance

feature_importances = rand_rf_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='brown', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 7. Bagging Classifier with pipeline and GridSearchCV

# In[16]:



baggingClf_pipe = make_pipeline(StandardScaler(), BaggingClassifier(random_state=1))

# This must be a distribution, not random
bc_params = {"base_estimator__max_depth": [3,5,10,20],
          "base_estimator__max_features": [None, "auto"],
          "base_estimator__min_samples_leaf": [1, 3, 5, 7, 10],
          "base_estimator__min_samples_split": [2, 5, 7],
          'bootstrap_features': [False, True],
          'max_features': [0.5, 0.7, 1.0],
          'max_samples': [0.5, 0.7, 1.0],
          'n_estimators': [2, 5, 10, 20],
}

# Type of scoring used to compare parameter combinations
acc_scorer =  metrics.make_scorer(metrics.precision_score)

# Run the grid search
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

baggingClf = GridSearchCV(baggingClf_pipe, bc_params, scoring=acc_scorer, cv=cv)
baggingClf = rand_rf.fit(X_train_res, y_train_res)

# Set the clf to the best combination of parameters
bc_best_estimator = baggingClf.best_estimator_

print('Tuned Hyperparameters for random forest Classifier: ', bc_best_estimator)

# Tune the model with best estimator
baggingClf_tuned = make_pipeline(StandardScaler(),bc_best_estimator) 
baggingClf_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(baggingClf_tuned)
make_confusion_matrix(baggingClf_tuned, y_test)



# ### Bagging Classifier feature_importances

# In[83]:


# Bagging Classifier feature_importances

feature_importances = bc_best_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 8. Bagging Classifier with pipeline and RandomizedSearchCV

# In[17]:



Rand_baggingClf_pipe = make_pipeline(StandardScaler(), BaggingClassifier(random_state=1))

# This must be a distribution, not random
bc_params = {"base_estimator__max_depth": [3,5,10,20],
          "base_estimator__max_features": [None, "auto"],
          "base_estimator__min_samples_leaf": [1, 3, 5, 7, 10],
          "base_estimator__min_samples_split": [2, 5, 7],
          'bootstrap_features': [False, True],
          'max_features': [0.5, 0.7, 1.0],
          'max_samples': [0.5, 0.7, 1.0],
          'n_estimators': [2, 5, 10, 20],
}

param_dist = {"baggingclassifier__max_depth": [3, None],
              "baggingclassifier__max_features": randint(1, 9),
              "baggingclassifier__min_samples_leaf": randint(1, 9),
              "baggingclassifier__criterion": ["gini", "entropy"]}


# Type of scoring used to compare parameter combinations
acc_scorer =  metrics.make_scorer(metrics.precision_score)

# Run the grid search
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

rand_baggingClf = RandomizedSearchCV(Rand_baggingClf_pipe, param_dist, scoring=acc_scorer,n_iter=50, cv=cv)
rand_baggingClf = rand_rf.fit(X_train_res, y_train_res)

# Set the clf to the best combination of parameters
rand_bg_best_estimator = rand_baggingClf.best_estimator_

print('Tuned Hyperparameters for baggining Classifier: ', rand_bg_best_estimator)

# Tune the model with best estimator
rand_baggingClf_tuned = make_pipeline(StandardScaler(),bc_best_estimator) 
rand_baggingClf_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(rand_baggingClf_tuned)
make_confusion_matrix(rand_baggingClf_tuned, y_test)


# ### RandomizedCV Bagging Classifier feature_importances

# In[84]:


# Randomized Bagging Classifier feature_importances

feature_importances = rand_bg_best_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 9. Adaboost Classifier with pipeline and GridSearchCV

# In[19]:



# Creating pipeline
Adaboostpipe = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=1))

# Parameter grid to pass in GridSearchCV
param_grid = {
    "adaboostclassifier__n_estimators": np.arange(10, 100, 10),
    "adaboostclassifier__learning_rate": [0.1, 0.01, 1],
    "adaboostclassifier__base_estimator": [
        DecisionTreeClassifier(max_depth=3, random_state=1),
        DecisionTreeClassifier(max_depth=5, random_state=1),
    ],
}

# Type of scoring used to compare parameter combinations
scorer =  metrics.make_scorer(metrics.precision_score)

# Calling GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

Adaboostpipegrid_cv = GridSearchCV(estimator=Adaboostpipe, param_grid=param_grid, scoring=scorer, cv=cv)

# Fitting parameters in GridSeachCV
Adaboostpipegrid_cv.fit(X_train_res, y_train_res)


# Set the clf to the best combination of parameters
adaboost_best_estimator = Adaboostpipegrid_cv.best_estimator_

print('Tuned Hyperparameters for Adaboost Classifier: ', adaboost_best_estimator)

# Tune the model with best estimator
Adaboost_tuned = make_pipeline(StandardScaler(),adaboost_best_estimator) 
Adaboost_tuned.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(Adaboost_tuned)
make_confusion_matrix(Adaboost_tuned, y_test)


# ### Adaboost Feature Importance 
# 

# In[85]:


# adaboost feature importance 
feature_importances = adaboost_best_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 10. Adaboost Classifier with pipeline and RandomizedSearchCV

# In[21]:



# Creating pipeline
rand_adaboost_pipe = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=1))

# Parameter grid to pass in GridSearchCV
param_grid = {
    "adaboostclassifier__n_estimators": np.arange(10, 100, 10),
    "adaboostclassifier__learning_rate": [0.1, 0.01, 1],
    "adaboostclassifier__base_estimator": [
        DecisionTreeClassifier(max_depth=1, random_state=1),
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],  
}
# Type of scoring used to compare parameter combinations
scorer =  metrics.make_scorer(metrics.precision_score)

#Calling RandomizedSearchCV
rand_adaboost_pipe_tuned = RandomizedSearchCV(estimator=rand_adaboost_pipe, param_distributions=param_grid, n_iter=50, scoring=scorer, cv=cv, random_state=1)
rand_adaboost_pipe_tuned.fit(X_train_res, y_train_res)

adaboost_best_estimator = rand_adaboost_pipe_tuned.best_estimator_


#Fitting parameters in RandomizedSearchCV
adaboost_best_estimator.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(adaboost_best_estimator)
make_confusion_matrix(adaboost_best_estimator, y_test)


# ### 11. XGBClassifier Classifier with pipeline and GridSearchCV

# In[27]:



XGBpipe=make_pipeline(StandardScaler(), XGBClassifier(random_state=1,eval_metric='logloss'))

param_grid={'xgbclassifier__n_estimators':np.arange(50,200,50),
            'xgbclassifier__learning_rate':[0.01,0.1,1], 
            'xgbclassifier__gamma':[0,1,3],
            'xgbclassifier__subsample':[0.2,0.6,1]}

# Type of scoring used to compare parameter combinations
scorer =  metrics.make_scorer(metrics.precision_score)

#Calling GridSearchCV
XGBgrid_cv = GridSearchCV(estimator=XGBpipe, param_grid=param_grid, scoring=scorer, cv=5)

#Fitting parameters in GridSeachCV
XGBgrid_cv.fit(X_train_res, y_train_res)

XGBbest_estimator = XGBgrid_cv.best_estimator_

#Fitting parameters in RandomizedSearchCV
XGBbest_estimator.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(XGBbest_estimator)
make_confusion_matrix(XGBbest_estimator, y_test)



# ###  XGB feature importance 

# In[86]:


#  XGB feature importance 
feature_importances = XGBbest_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='yellow', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 12. XGBClassifier Classifier with pipeline and RandomizedSearchCV

# In[29]:



rand_XGBpipe=make_pipeline(StandardScaler(),XGBClassifier(random_state=1,eval_metric='logloss', n_estimators = 10))

param_grid={'xgbclassifier__n_estimators':np.arange(50,200,50),
            'xgbclassifier__scale_pos_weight':[0,1,2],
            'xgbclassifier__learning_rate':[0.01,0.1,1],
            'xgbclassifier__gamma':[0,1,5],
            'xgbclassifier__subsample':[0.1,0.5,1],
           'xgbclassifier__max_depth':np.arange(1,5,1),
            'xgbclassifier__reg_lambda':[1,5,10]}


# Type of scoring used to compare parameter combinations
scorer =  metrics.make_scorer(metrics.precision_score)

#Calling RandomizedSearchCV
XGBrandomized_cv = RandomizedSearchCV(estimator=rand_XGBpipe, param_distributions=param_grid, n_iter=10, scoring=scorer, cv=cv, random_state=1)

#Fitting parameters in RandomizedSearchCV
XGBrandomized_cv.fit(X_train_res, y_train_res)

rand_XGBbest_estimator = XGBrandomized_cv.best_estimator_

#Fitting parameters in RandomizedSearchCV
rand_XGBbest_estimator.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(rand_XGBbest_estimator)
make_confusion_matrix(rand_XGBbest_estimator, y_test)


# ###  RandXGB feature importance 

# In[87]:


#  RandXGB feature importance 
feature_importances = rand_XGBbest_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### 13. GradientBoostingClassifier Classifier with pipeline and GridSearchCV
# 

# In[40]:



GradBoostingpipe=make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1))

parameters={
    "gradientboostingclassifier__learning_rate": [0.01, 0.001, 0.1, 1],
    "gradientboostingclassifier__min_samples_split": np.linspace(0.1, 0.5, 5),
    "gradientboostingclassifier__min_samples_leaf": np.linspace(0.1, 0.5, 5),
    "gradientboostingclassifier__max_depth":[3, 8],
    "gradientboostingclassifier__max_features":["log2","sqrt"],
    "gradientboostingclassifier__subsample":[0.5, 1.0],
    "gradientboostingclassifier__n_estimators":[10]
    }

# Type of scoring used to compare parameter combinations
scorer =  metrics.make_scorer(metrics.precision_score)

#Calling GridSearchCV
GradBoosting_cv = GridSearchCV(estimator=GradBoostingpipe, param_grid=parameters,
                               scoring=scorer, cv=5)

#Fitting parameters in GridSeachCV
GradBoosting_cv.fit(X_train_res, y_train_res)

GradBoosting_estimator = GradBoosting_cv.best_estimator_

#Fitting parameters in RandomizedSearchCV
GradBoosting_estimator.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(GradBoosting_estimator)
make_confusion_matrix(GradBoosting_estimator, y_test)


# ### 14. GradientBoostingClassifier Classifier with pipeline and RandomizedSearchCV
# 

# In[49]:



from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
    
RanGradBoostingpipe=make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1))

parameters = {'gradientboostingclassifier__learning_rate': sp_randFloat(),
                  'gradientboostingclassifier__subsample'    : sp_randFloat(),
                  'gradientboostingclassifier__n_estimators' : sp_randInt(100, 1000),
                  'gradientboostingclassifier__max_depth'    : sp_randInt(4, 10) 
                 }

# Type of scoring used to compare parameter combinations
scorer =  metrics.make_scorer(metrics.precision_score)

#Calling RandomizedSearchCV
RanGradBoosting_cv = RandomizedSearchCV(estimator=RanGradBoostingpipe, param_distributions=parameters, 
                               scoring=scorer, cv=5)

#Fitting parameters in GridSeachCV
RanGradBoosting_cv.fit(X_train_res, y_train_res)

RandGradBoosting_estimator = RanGradBoosting_cv.best_estimator_

#Fitting parameters in RandomizedSearchCV
RandGradBoosting_estimator.fit(X_train_res, y_train_res)

# Scores and Confusion matrix
get_metrics_score(RandGradBoosting_estimator)
make_confusion_matrix(RandGradBoosting_estimator, y_test)


# ### GradientBoostingClassifier feature importance 
# 
# 

# In[89]:


feature_importances = RandGradBoosting_estimator.steps[1][1].feature_importances_
indices  = np.argsort(feature_importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), feature_importances[indices], color='green', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### Comparing Models Performances

# In[102]:



all_models = [log_reg_tuned, rand_logistic_tuned_pipe,
              Dtree_tuned, rand_Dtree_tuned,
            rf_tuned,  rand_rf_tuned, 
            baggingClf_tuned, rand_baggingClf_tuned, 
             Adaboost_tuned,adaboost_best_estimator, 
              XGBbest_estimator,
             rand_XGBbest_estimator,
             GradBoosting_estimator,
            RandGradBoosting_estimator]

# defining empty lists to add train and test results
acc_train = []
acc_test = []
recall_train = []
recall_test = []
precision_train = []
precision_test = []

# looping through all the models to get the accuracy, precall and precision scores
for model in all_models:
    j = get_metrics_score(model,False)
    acc_train.append(np.round(j[0],2))
    acc_test.append(np.round(j[1],2))
    recall_train.append(np.round(j[2],2))
    recall_test.append(np.round(j[3],2))
    precision_train.append(np.round(j[4],2))
    precision_test.append(np.round(j[5],2))

    
# Put everything in a dataframe
comprison = pd.DataFrame(
{
    'Model':['Logistic GridSearchCV', 'Logistic RandomizedGridSearchCV',
             'DTree GridSearchCV', 'DTree RandomizedGridSearchCV',
             'RF GridSearchCV','RF RandomizedGridSearchCV',
             'Bagging GridSearchCV','Bagging RandomizedGridSearchCV',
             'Adaboost GridSearchCV', 'Adaboost RandomizedGridSearchCV',
             'XGB GridSearchCV', 'XGB RandomizedGridSearchCV',
             'GradBoosting GridSearchCV','GradBoosting RandomizedGridSearchCV'
             ],
    'Train_Accuracy': acc_train,'Test_Accuracy': acc_test,
                                          'Train_Recall':recall_train,'Test_Recall':recall_test,
                                          'Train_Precision':precision_train,'Test_Precision':precision_test})

coparison_final = comprison.sort_values(["Test_Precision"], ascending = False)
coparison_final


#  ### Insights:
#  
#  - Test accuracy metric seems to perform well for all models, ranging from 0.76 to 0.89. The high level of accuracy ,however, is not a good measure of identifying cutsomers due to the severe class imbalance in the target variable customers. 
#   
# - Precision is a good measure here, because The Thera bank should aim at  finding customers who might churn. 
# 
# - The Precision metric for the training set overfits the data for GradBoosting, XGB, and Adaboost models. At the same time, it performs poorly on both the training and testing datasets for Decision Tree, Logistic regression, and Gradient Boost models. 
# 
# 
# - The Precision metric for Random forest RandomizedGridSearchCV model performs well for both precision in the training and the testing dataset, with 0.80 and 0.74 respectivley. The precision recall does not overfit a lot and the test precision is high. Similarly, Bagging RandomizedGridSearchCV performs really well on both test and train precisions. 
# 
# 
# - The analysis suggests that The Thera bank can use Random forest RandomizedGridSearchCV and Bagging RandomizedGridSearchCV models to predict whether a customer will churn.
# 
# 
# - The company should take into consideration the following factors when determining advertising costs and customer segmentation: transaction amount and total revolving balance, relationship counts, and average utilization ratio.
# 
# 
# - The compnay should also consider trying additional models tuning analysis or add new models such as logistic regression, or neural networks for future insights. 

# In[ ]:





# In[ ]:




