#!/usr/bin/env python
# coding: utf-8

# # Import libraries necessary for this project

# In[1]:


import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# # Loading data 

# In[2]:


data= pd.read_csv("C:\\Users\\Sumit Ranjan\\Desktop\\pgdm\\semester 5\\loan.csv")


# # Introduction To The Data

# In[3]:


data.head()


# In[4]:


new_data = data[['member_id', 'loan_amnt', 'funded_amnt', 'addr_state', 'funded_amnt_inv', 'grade', 'term', 'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'loan_status']]


# In[5]:


new_data.head()


# # Data Cleaning

# In[6]:


new_data['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
new_data['term'] = pd.to_numeric(new_data['term'], errors='coerce')


# In[7]:


new_data['emp_length'].replace('n/a', '0', inplace=True)
new_data['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
new_data['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
new_data['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
new_data['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
new_data['emp_length'] = pd.to_numeric(new_data['emp_length'], errors='coerce')


# In[8]:


new_data['grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
new_data['grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
new_data['grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
new_data['grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
new_data['grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
new_data['grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
new_data['grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
new_data['grade'] = pd.to_numeric(new_data['grade'], errors='coerce')


# In[9]:


new_data.isnull().sum().sort_values(ascending=False).head()


# In[10]:


cols = ['term', 'loan_amnt', 'funded_amnt', 'int_rate', 'grade', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']
for col in cols:
    new_data[col].fillna(new_data[col].median(), inplace=True)


# In[11]:


cols = ['acc_now_delinq', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med']
for col in cols:
    new_data[col].fillna(0, inplace=True)


# #  Loan status columns into two categories i.e. default and fully paid

# In[12]:


default=['Charged Off','Late (31-120 days)','Default','Late (16-30 days)','In Grace Period']
paid=['Fully Paid']
new_data['loan_status']=new_data['loan_status'].apply(lambda x: 'Default' if x in default else x)
new_data['loan_status']=new_data['loan_status'].apply(lambda x: 'Fully Paid' if x in paid else x)
new_data=new_data[(new_data['loan_status']=='Default') | (new_data['loan_status']=='Fully Paid')]
target_variable=new_data['loan_status'].apply(lambda x: 1 if x=='Default' else 0)
new_data.drop('loan_status',axis=1,inplace=True)


# In[13]:


selected_cols = ['member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'grade', 'int_rate', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'mths_since_last_major_derog', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt', 'recoveries', 'collection_recovery_fee', 'term', 'acc_now_delinq', 'collections_12_mths_ex_med']
new_data_2=new_data[selected_cols]


# # Creating new variable for better training of data

# # Loan to income ratio

# In[14]:


new_data_2['loan_to_income_ratio'] = new_data_2['annual_inc']/new_data_2['funded_amnt_inv']


# In[15]:


new_data_2.head()


# In[16]:


new_data_2['avl_lines'] = new_data_2['total_acc'] - new_data_2['open_acc']


# # Total Interest paid

# In[17]:


new_data_2['int_paid'] = new_data_2['total_rec_int'] + new_data_2['total_rec_late_fee']


# # Total repayment 

# In[18]:


new_data_2['total_repayment'] =  ((new_data_2['recoveries']/new_data_2['funded_amnt_inv']) * 100)


# In[19]:


new_data_2.shape


# # Splitting the data as train and test

# In[24]:


X_train,X_test,y_train,y_test = train_test_split(np.array(new_data_2),target_variable,test_size=.30,random_state=1) 
X_train.shape
X_test.shape


# In[27]:


eval_set=[(X_test, y_test)]


# In[30]:


import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
# fit model no training data
model = XGBClassifier()
 
from sklearn import tree
model.fit(X_train,y_train)
df_test_output= model.predict(X_test)
import pandas as pd
import numpy as np
np.unique(y_test, return_counts=True)
np.unique(df_test_output, return_counts=True)
 


# In[31]:


accuracy_score(y_test, df_test_output)


# # Accuracy calculation

# In[32]:


accuracy = accuracy_score(np.array(y_test).flatten(),df_test_output)
print("Accuracy: %.10f%%" % (accuracy * 100.0))


# # Confusion matrix

# In[36]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test
predicted = df_test_output
results = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results) 
 
print (classification_report(actual, predicted))


# In[ ]:




