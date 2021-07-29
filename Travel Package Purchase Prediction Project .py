#!/usr/bin/env python
# coding: utf-8

# ## 1. Load all libraries:

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from os import system 
# from sklearn.preprocessing import Imputer
import xgboost as xgb


# ## 2. Load the data

# In[2]:


tourism = pd.read_csv('/Users/khaledsharafaddin/Documents/Univ_Austin_Texas ML_AI/DataSets/Tourism.csv')
tourism.head()
tourism.info()
tourism.shape  # (4888, 20)


# In[3]:


tourism.head()


# ## 3. Perform an Exploratory Data Analysis
# 

# In[4]:


# a. ProductPitched Segmentation
tourism['ProductPitched'].value_counts()/tourism.shape[0]*100

# b. Male to Female ratio
tourism['Gender'].value_counts()/tourism.shape[0]*100
sns.countplot(x ='Gender', data = tourism) 

# c. Histogram for distribution of numeric values 
columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisited', 'PitchSatisfactionScore','MonthlyIncome']
tourism[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2))

# d. Correlations among variables 
tourism.corr()

# e. Ratio of those who took products 
tourism['ProdTaken'].value_counts()/tourism.shape[0]*100

# f. Those who have higher income tend to purchase high-end products such as super deluxe and king
sns.catplot(x="ProdTaken", y="MonthlyIncome", kind="box",hue="ProductPitched", data=tourism)

# g. Female to Male by Age Categories 
sns.barplot(x="Gender", y="Age", data=tourism)

# h. How many customers have travel documents  
tourism['Passport'].value_counts()/tourism.shape[0]*100

# i. Designation by Gender
ax = sns.catplot(y="Gender", hue="Designation", kind="count",
            palette="pastel", edgecolor=".6",
            data=tourism)
ax.fig.suptitle('Designation by Gender')


# #### EDA Insights:
# - There are 4888 rows and 20 columns in the dataset
# - Approximately 73% of customers use Basic or the deluxe packages. 
# - Male customers are slightely higher than females 
# - The distribution of customer age is symmetric with mean 37 and median 36.
# - The median monthly income of a customer is $22347
# - There is a weak correlation between age and income of 0.4 
# - Approx. 81% of customers have not taken the product in comparison to only 18.8% who did. There seems to be  a class imbalance here.
# - Those who have higher income tend to purchase high-end products such as super deluxe and king
# - 71% of customers don't have passports, which might result in less purchase and less traveling
# - 37.9% of customers do not own a car, which could result in them not traveling as often as someone who has a car
# - There are more Male Managers and executives than females. 

# ## 4. Data Cleaning, pre-processing and Handling Missing Values

# In[5]:


# a. Gender needs to be standarized into two categories
tourism['Gender'] = tourism['Gender'].replace('Fe Male', 'Female')

# b. Customer_ID is not useful. Will be dropped 
tourism = tourism.drop(['CustomerID'], axis=1)

# c. There are several Missing values:

# c.1 Age will be replaced with the mean 
tourism['Age'] = tourism['Age'].fillna((tourism['Age'].mean()))

# c.2 Type of Contact NaN will be replaced with a third category called 'Unspecified'
tourism['TypeofContact'] = np.where(tourism['TypeofContact'].isnull(),"Unspecified",tourism['TypeofContact'])

# c.3 DurationOfPitch, numberoffollowups, and NumberOfTrips will be replaced with the mean
tourism['DurationOfPitch'] = tourism['DurationOfPitch'].fillna((tourism['DurationOfPitch'].mean()))
tourism['NumberOfFollowups'] = tourism['NumberOfFollowups'].fillna((tourism['NumberOfFollowups'].mean()))
tourism['NumberOfTrips'] = tourism['NumberOfTrips'].fillna((tourism['NumberOfTrips'].mean()))

# c.4 NumberOfChildrenVisited will be replaced with 0 
tourism['NumberOfChildrenVisited'] = tourism['NumberOfChildrenVisited'].fillna(0)

# c.5 PreferredPropertyStar is a category and will be replaced with 0 
tourism['PreferredPropertyStar'] = tourism['PreferredPropertyStar'].fillna(0.0)

# c.6 MonthlyIncome will be replaced with the median income 
tourism['MonthlyIncome'] = tourism['MonthlyIncome'].fillna((tourism['MonthlyIncome'].median()))


# d Replacing categorical ordered variables to numeric values 
replaceCat = {
    'TypeofContact':{'Unspecified': 0,'Self Enquiry':1, 'Company Invited':2},
    'Occupation':{'Salaried':1, 'Small Business':2, 'Large Business':3,'Free Lancer':4},
    'ProductPitched':{'Basic':0, 'Standard':1,'Deluxe':2,'Super Deluxe':3, 'King':4},
    'Designation': {'Executive':0, 'VP':1, 'AVP':2, 'Senior Manager':3,'Manager':4}
    }

tourism = tourism.replace(replaceCat)


# e Create dummy variables for MaritalStatus and Gender
onehotencoder = ['MaritalStatus', 'Gender']
tourism = pd.get_dummies(tourism, columns = onehotencoder)


# f. Convert all object types to categorical
for feature in tourism.columns:
    if tourism[feature].dtype=='object':
        tourism[feature] = pd.Categorical(tourism[feature])
       


# ## 5. Prepare Data for Modeling

# In[6]:


# 5.1 Create a train and test sets
X = tourism.drop(['ProdTaken'], axis=1)  
y = tourism['ProdTaken']

# 5.2 Split into train and test, with stratify because there is class imbalance
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# ### which metric is right for model? 
# - We are interested in finding customers who may take new offering of packages (ProdTaken=1)
# - The travel package purchase dataset has class imbalance. Approx. 81% of the product taken categories are no (0), and therefore accuracy is not a good measure.
# - Recall: aims to find the proportion of actual positives was identified correctly. 
# - Therefore, recall is a good measure here, because the marketing cost of identifying customers who might NOT take the product is costly.

# ## 6. Function to produce metrics such as accuracy, precision, and recall for train and test sets and Confusion Matrix:
# 

# In[7]:



# Create a confusion Matrix 
def make_confusion_matrix(model, y_actual, labels=[1,0]):
    y_predict = model.predict(x_test)
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0,1])
    df_cm  = pd.DataFrame(cm, index=[i for i in ['Actual-No', 'Actual-Yes']],
                          columns =[i for i in ['Predicted-No', 'Predicted-Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels= np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=labels, fmt='')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


 # Create function to calculate all scores and prints them nicely:

def get_metrics_score(model, flag=True):
    # empty list to store results
    score_list = []
    
    # Predict on train and test
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    
    # Accuracy
    acc_train = model.score(x_train, y_train)
    acc_test = model.score(x_test, y_test)
    
    # Recall
    recall_train = metrics.recall_score(y_train, pred_train)
    recall_test = metrics.recall_score(y_test, pred_test)
    
    # Percision
    perc_train = metrics.precision_score(y_train, pred_train)
    perc_test = metrics.precision_score(y_test, pred_test)
    
    score_list.extend((acc_train, acc_test, recall_train, recall_test, 
                   perc_train,   perc_test ))
    
    if flag == True: 
        print("Accuracy on training set : ",model.score(x_train,y_train))
        print("Accuracy on test set : ",model.score(x_test,y_test))
        print("Recall on training set : ",metrics.recall_score(y_train,pred_train))
        print("Recall on test set : ",metrics.recall_score(y_test,pred_test))
        print("Precision on training set : ",metrics.precision_score(y_train,pred_train))
        print("Precision on test set : ",metrics.precision_score(y_test,pred_test))
    
    return score_list # returning the list with train and test scores


# ## 7. Model building - Bagging 
# 

# In[8]:


# Bagging Classfier

baggining_estimator = BaggingClassifier(random_state=1)
baggining_estimator.fit(x_train, y_train)

# Bagging Metrics and Confusion Matrix
make_confusion_matrix(baggining_estimator, y_test)
get_metrics_score(baggining_estimator)


# In[9]:


# 7.2 Bagging Classfier with Grid Search 

baggining_estimator_tuned = BaggingClassifier(random_state=1)

# Grid search 
parameters = {
    'max_samples': [0.6,0.7, 0.8, 0.9, 1], # 70% to 100% of the data
    'max_features': [0.6,0.7, 0.8, 0.9, 1], 
    'n_estimators': [10,20, 30, 40, 50, 100] # number of trees in the forest
}

# Type of scoring used to compare parameter combination
acct_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(baggining_estimator_tuned, parameters, scoring=acct_scorer, cv=5)
grid_obj= grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters automatically:
baggining_estimator_tuned = grid_obj.best_estimator_

# fit the best algorithm to the data
baggining_estimator_tuned.fit(x_train, y_train)

# Bagging Metrics and Confusion Matrix
make_confusion_matrix(baggining_estimator_tuned, y_test)
get_metrics_score(baggining_estimator_tuned)


# In[10]:


### Bagging Insights: 
# The training sets seem to overfit 
# Precision on the test set performs really well with 0.95 precision score.
# Recall on the test set performs poorly with 0.49 recall score. 


# In[11]:


## 7.3 Random Forest 

rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(x_train, y_train)

# Bagging Metrics and Confusion Matrix
make_confusion_matrix(rf_estimator, y_test)
get_metrics_score(rf_estimator)


# In[12]:


# 7.4 Random Forest Model Tuned:

rf_estimator_tuned = RandomForestClassifier(random_state=1)

# Note: class_weight says: put more emphasis on the customers who will take products 1 by 70%. 
parameters = {"n_estimators": [150,200,250],
    "min_samples_leaf": np.arange(5, 10),
    "max_features": np.arange(0.2, 0.7, 0.1),
              "class_weight": [{0:0.3, 1:0.7}] 
             }

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
rf_estimator_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
rf_estimator_tuned.fit(x_train, y_train)


# In[13]:


# RF Metrics and Confusion Matrix
make_confusion_matrix(rf_estimator_tuned, y_test)
get_metrics_score(rf_estimator_tuned)

# Important Features: 
importances = rf_estimator_tuned.feature_importances_
indices = np.argsort(importances)
feature_names = X.columns

plt.figure(figsize=(10,10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[14]:


# Random Forest Insights: 
# The training sets seem to overfit the data 
# Precision on the test set performs really well with 0.95 precision score.
# Recall on the test set performs poorly with 0.49 recall score.


# In[15]:


# 7.5 Decision Tree 



dt_estimator = DecisionTreeClassifier(random_state=1)
dt_estimator.fit(x_train, y_train)

# Bagging Metrics and Confusion Matrix
make_confusion_matrix(dt_estimator, y_test)
get_metrics_score(dt_estimator)


# In[16]:


# 7.6 Decision Tree Tuned

Decision_Tree_estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from

parameters = {'max_depth': np.arange(1,10), 
              'min_samples_leaf': [1, 2, 5, 7, 10,15,20],
              'max_leaf_nodes' : [2, 3, 5, 10],
              'min_impurity_decrease': [0.001,0.01,0.1]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(Decision_Tree_estimator, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
Decision_Tree_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
Decision_Tree_estimator.fit(x_train, y_train)


# Important Features: 
importances = Decision_Tree_estimator.feature_importances_
indices = np.argsort(importances)
feature_names = X.columns

plt.figure(figsize=(10,10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ## 8. Model Building - Boosting

# In[17]:


# 8.1 AdaBoost

adaboost = AdaBoostClassifier(random_state=1)
adaboost.fit(x_train, y_train)

# Matrix and Scores:
get_metrics_score(adaboost)
make_confusion_matrix(adaboost, y_test)


# In[18]:


# 8.2 AdaBoost Tuned
adaboost_tuned = AdaBoostClassifier(random_state=1)

# base estimator can be decision trees with multiple depths
parameters = {
    'base_estimator':[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),DecisionTreeClassifier(max_depth=3),DecisionTreeClassifier(max_depth=4),DecisionTreeClassifier(max_depth=5)],
    'n_estimators':np.arange(10, 110, 10),
    'learning_rate': np.arange(0.1, 2, 0.1)
}
acct_score = metrics.make_scorer(metrics.recall_score)

# Grid search
grid_search = GridSearchCV(adaboost_tuned, parameters, scoring=acct_score, cv=5)
grid_search.fit(x_train, y_train)

adaboost_tuned = grid_search.best_estimator_

adaboost_tuned.fit(x_train, y_train)

# get score info for adaboost tuned 
get_metrics_score(adaboost_tuned)
make_confusion_matrix(adaboost_tuned, y_test)

# Important Features 
importance = adaboost_tuned.feature_importances_
indices  = np.argsort(importance)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importance[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[19]:


# 8.3 Gradient Boosting Classifier

gboosting = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1), random_state=1)
gboosting.fit(x_train, y_train)

# See results 
get_metrics_score(gboosting)
make_confusion_matrix(gboosting, y_test)


# In[20]:


# 8.4 Gradient Boosting Classifier Tuned


gboosting_tuned = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1), random_state=1)

# Grid search 
parameters = {
    'n_estimators': [100, 150,200, 250],  # num of trees
    'max_features': [0.7, 0.8, 0.9, 1], 
    'max_features': [0.7, 0.8, 0.9, 1] 
}

acc_scorer = metrics.make_scorer(metrics.recall_score)
grid_obj = GridSearchCV(gboosting_tuned, parameters,scoring=acc_scorer, cv=5)
grid_obj.fit(x_train, y_train)

gboosting_tuned = grid_obj.best_estimator_

# Fit on best estimators
gboosting_tuned.fit(x_train, y_train)

# See results 
get_metrics_score(gboosting_tuned)
make_confusion_matrix(gboosting_tuned, y_test)

# Important Features:
importance = gboosting_tuned.feature_importances_
indices  = np.argsort(importance)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importance[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[21]:


# 8.5 XGBOOST Classifier 

xgb_classifier = xgb.XGBClassifier(random_state=1, eval_metric='logloss')

# Fit the best algorithm to the data.
xgb_classifier.fit(x_train, y_train)

get_metrics_score(xgb_classifier)
make_confusion_matrix(xgb_classifier, y_test)


# In[22]:


# 8.6 XGBOOST Classifier TUNED

xgb_tuned = xgb.XGBClassifier(random_state=1, eval_metric='logloss')


parameters = {
    "n_estimators": [10,50,100],
    "scale_pos_weight":[1,2,5],
    "subsample":[0.5,1],
    "learning_rate":[0.01,0.1,1],
    "gamma":[0,1,3],
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(xgb_tuned, parameters,scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(x_train, y_train)

get_metrics_score(xgb_tuned)
make_confusion_matrix(xgb_tuned, y_test)


# Important Features:
importance = xgb_tuned.feature_importances_
indices  = np.argsort(importance)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importance[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ## 9. Actionable Insights & Recommendations
# 
# 

# In[24]:




all_models = [baggining_estimator, baggining_estimator_tuned,
              rf_estimator, rf_estimator_tuned,
              dt_estimator, Decision_Tree_estimator,
              adaboost, adaboost_tuned, 
              gboosting, gboosting_tuned,
              xgb_classifier, xgb_tuned]

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
    'Model':["baggining_estimator", "baggining_estimator_tuned",
              "rf_estimator", "rf_estimator_tuned",
              "dt_estimator", "Decision_Tree_estimator",
              "adaboost", "adaboost_tuned", 
              "gboosting", "gboosting_tuned",
              "xgb_classifier", "xgb_tuned"],
    'Train_Accuracy': acc_train,'Test_Accuracy': acc_test,
                                          'Train_Recall':recall_train,'Test_Recall':recall_test,
                                          'Train_Precision':precision_train,'Test_Precision':precision_test})


coparison_final = comprison.sort_values(["Test_Recall"], ascending = True)
coparison_final


#  ### Insights:
#  
#  - Test accuracy metric seems to perform well for all models, ranging from 0.75 to 0.92. The high level of accuracy ,however, is not a good measure of identifying cutsomers due to the severe class imbalance in the target variable prod_taken. 
#   
# - Recall is a good measure here because Visit with us company. Recall is a good measure here, because the marketing cost of identifying customers who might NOT take the product is high.
# 
# 
# - The recall metric for the training set overfits the data for Random Forest, Adaboost Tuned, Bagging and Decision Tree Tuned models. At the same time, it performs poorly on both the training and testing datasets for Adaboost, Decision Tree, xgboost, and Gradient Boost models. 
# 
# 
# - The Recall metric for Tuned xgboost model performs well for both recall in the training and the testing dataset, with 0.82 and 0.77 respectivley. The train recall does not overfit a lot and the test recall is high. 
# 
# 
# - The analysis suggests that Visit with us travel agency can use xgb_tuned model to predict whether a customer will take the travel package. 
# 
# 
# - The company should take into consideration the following factors when determining advertising costs and customer segmentation: customers who own a passport, designation, age and monthly income. 
# 
# 
# - The compnay should also consider trying additional models tuning analysis or add new models such as logistic regression, or neural networks for future insights. 

# In[ ]:





# In[ ]:




