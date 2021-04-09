#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# The classification of clients applying for loan into bad clients and good clients with respect to the various details regarding the client provided to the bank so the bank could make informative decision to avoid risk of non-repayment of loan and hence reduce liquid damage to the bank.<br>
# 
# 1. Month: Month of loan applied (1-12) 
# 2. Credit Amount: Balance Amount 
# 3. Credit Term: No.of months due
# 4. Age: Age of the client
# 5. Sex : Gender (Male/Female)
# 6. Education: Types  of education 
# 7. Product type: Type of product for loan
# 8. Having_children: If the client has children (0/1)
# 9. Region: Region from which client comes from 
# 10. Income: Monthly Salary of the client
# 11. Family_status: Whether the client is married or not.
# 12. Phone operator: Type of phone operator used.
# 13. Is_client: Has an existing loan
# 
# The dependent/target variable is:
# 14. Bad_client: The client is considered a bad client to give a loan to. Good client (0) or bad client (1)
# 

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, cohen_kappa_score, accuracy_score, roc_auc_score, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier


# In[5]:


plt.rcParams['figure.figsize'] = [15,8]


# In[6]:


client = pd.read_csv('clients.csv')


# # Preliminary Investigation

# In[7]:


client.head()


# In[8]:


client.tail()


# In[9]:


client.shape


# **Comments:**<br>
# Total 1723 records, 13 features and 1 target variable

# In[10]:


client.info()


# Comments
# 1. There are 10 numerical columns
# 2. And 4 categorical columns

# In[11]:


client['month'] = client['month'].astype('O')
client['credit_term'] = client['credit_term'].astype('O')
client['having_children_flg'] = client['having_children_flg'].astype('O')
client['region'] = client['region'].astype('O')
client['phone_operator'] = client['phone_operator'].astype('O')
client['phone_operator'] = client['phone_operator'].astype('O')
client['is_client'] = client['is_client'].astype('O')


# In[12]:


client.describe()


# **Comments**
# 1. Credit card is positiveky skewed.
# 2. Age is near normally distributed.
# 3. Income is also positiveky skewed.
# 4. Target variable is skewed as well.<br>
# `Need for scaling`

# # Exploratory Data Analysis

# ## 1. Target Distribution

# In[13]:


sns.countplot(x='bad_client_target', data=client)
plt.show()


# In[14]:


df_num = client.select_dtypes(np.number)
df_num.drop(columns=['bad_client_target'], inplace=True)
df_num.columns


# In[15]:


df_cat = client.select_dtypes(include='object')
df_cat.columns


# ## 2. Univariate analysis of Categorical features

# In[16]:


fig, axes = plt.subplots(5, 2, figsize=(18,30))
axes = [ax for axes_rows in axes for ax in axes_rows]

for i, c in enumerate(df_cat.columns):
    client[c].value_counts()[::-1].plot(kind='pie',
                                          ax=axes[i],
                                          title=c,
                                          autopct='%.0f%%',
                                          fontsize=12)
    axes[i].set_ylabel('')


# In[17]:


fig, axes = plt.subplots(5, 2, figsize=(18,30))
axes = [ax for axes_rows in axes for ax in axes_rows]

for i, c in enumerate(df_cat.columns):
    client[c].value_counts()[::-1].plot(kind='barh',
                                          ax=axes[i],
                                          title=c,
                                          fontsize=12)


# **Comments**
# 1. More number of clients during the end of the calender year.
# 2. Highest No.of months due (credit term) is 12.
# 3. Male and Female ratio is also equal.
# 4. Most of the loan applicants are highly educated.
# 5. Surprisingly, more than 60% of the loan taken are for non essential items like cell phone, home appliences and computers and very less for essential types like medical.
# 6. More than 80% of the customers are from region 2. Need to focus more on region 1 and 0.

# ## 3. Bivariate analysis of Categorical features

# In[18]:


fig, axes = plt.subplots(5, 2, figsize=(16,24))
axes = [ax for axes_rows in axes for ax in axes_rows]

for i, c in enumerate(df_cat.columns):
    fltr = client['bad_client_target']==0
    
    vc_a=client[fltr][c].value_counts(normalize=True).reset_index().rename({'index':c,c:'count'}, axis=1)
    
    vc_b=client[~fltr][c].value_counts(normalize=True).reset_index().rename({'index':c,c:'count'}, axis=1)
    
    vc_a['bad_client_target']=0
    vc_b['bad_client_target']=1
    
    df = pd.concat([vc_a, vc_b]).reset_index(drop=True)
    
    sns.barplot(y=c, x='count', data=df, hue='bad_client_target', ax=axes[i], orient='h')


# **Comments**
# 1. Most of the client who applied loan in the begining or end of the year are bad clients.
# 2. Credit term's with multiples of 6 i.e, half yearly, yearly and one and a half yearly have high number of bad clients.
# 3. High no. of bad clients are females.
# 4. Surprisingly, highly educated clients are more bad clients.
# 5. Loan application for cell phone attract more no.of bad clients.

# ## 4. Univariate analysis of Numerical features

# In[19]:


fig, axes = plt.subplots(2, 2,figsize=(15,10))
y = 0
for c in df_num.columns:
    i, j = divmod(y, 2)
    if c!='bad_client_target':
        sns.boxplot(x = c, data=client, orient='h', ax=axes[i, j])
        y = y + 1


# **Comments**<br>
# Credit amount and income have highest no.of outliers on the positive side, meaning, more variations in high income clients. 

# ## 5. Bivariate analysis of Numerical features

# In[20]:


#sns.set(font_scale=1.3)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(df_num):
    client.groupby('bad_client_target')[c].median().plot(kind = 'barh', title=f'Median_{c}', ax=axes[i])


# ## 6. Correlation analysis

# In[21]:


client['month'] = client['month'].astype('int64')
client['credit_term'] = client['credit_term'].astype('int64')
client['having_children_flg'] = client['having_children_flg'].astype('int64')
client['region'] = client['region'].astype('int64')
client['phone_operator'] = client['phone_operator'].astype('int64')
client['phone_operator'] = client['phone_operator'].astype('int64')
client['is_client'] = client['is_client'].astype('int64')


# In[22]:


figs = plt.figure(figsize=(15, 10))
sns.heatmap(client.corr(), annot=True)
plt.show()


# **Comments**
# 1. Region and income are highly negative correlated.
# 2. Region and credit term have almost no correlation.
# 3. Phone operator and income have almost no correlation.
# 4. There also no correlation between month and phone operator.<br>
# `To conclude we do not see any multicollinearity`

# # Encoding

# In[23]:


df_num = client.select_dtypes(np.number)
df_num.drop(columns=['bad_client_target'], inplace=True)
df_num.columns


# In[24]:


df_cat = client.select_dtypes(include='object')
df_cat.columns


# In[25]:


dummy_var = pd.get_dummies(df_cat, drop_first=True)
dummy_var.head(2)


# # Function definitions

# In[26]:


def plot_confusion_matrix(model, cutoff):
    y_pred_prob = model.predict(X_test)
    y_pred = [ 0 if x < cutoff else 1 for x in y_pred_prob]
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])

    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cbar = False, linewidths = 0.1, annot_kws = {'size':25})
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()

    
def get_test_report(model, cutoff):
    y_pred_prob = model.predict(X_test)
    y_pred = [ 0 if x < cutoff else 1 for x in y_pred_prob]
    print(classification_report(y_test, y_pred))


# In[27]:


score_card = pd.DataFrame(columns=['Model Name','Probability Cutoff', 'AUC Score', 'Precision Score', 'Recall Score',
                                       'Accuracy Score', 'Kappa Score', 'f1-score'])

def update_score_card(Model_name, model, cutoff='-'):
    y_pred_prob = model.predict(X_test)
    y_pred = [ 0 if x < cutoff else 1 for x in y_pred_prob]
    global score_card
    score_card = score_card.append({'Model Name':Model_name,
                                    'Probability Cutoff': cutoff,
                                    'AUC Score' : metrics.roc_auc_score(y_test, y_pred_prob),
                                    'Precision Score': metrics.precision_score(y_test, y_pred),
                                    'Recall Score': metrics.recall_score(y_test, y_pred),
                                    'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                                    'Kappa Score':metrics.cohen_kappa_score(y_test, y_pred),
                                    'f1-score': metrics.f1_score(y_test, y_pred)}, 
                                    ignore_index = True)
    
    return score_card


# # Scaling

# In[28]:


sc = StandardScaler()

num_scaled = sc.fit_transform(df_num)
df_num_scaled = pd.DataFrame(num_scaled, columns = df_num.columns)

X = pd.concat([df_num_scaled,dummy_var], axis=1)
Y = client['bad_client_target']

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 10, test_size = 0.3)


# **Comments**<br>
# As we saw in the EDA most of the features are skewed and range of values are differ by big margin. Scaling is required.

# # Base Model

# In[29]:


logreg_full = sm.Logit(y_train,X_train).fit(method='bfgs')
print(logreg_full.summary())


# **Comments**<br>
# Very little difference between log-likelihood and log-null-likelihood, meaning, no much significance of features in predicting the target variable.

# In[30]:


df_odds = pd.DataFrame(np.exp(logreg_full.params), columns= ['Odds']) 
df_odds


# **How to interpret above table**
# 1. Odds of bad client is 0.059.
# 2. Month = 1.839477, it implies that the odds of getting bad client increases by a factor of 1.017969 due to one unit increase in the month, keeping other variables constant.
# 3. credit_amount = 1.192460, it implies that the odds of getting bad client increases by a factor of 1.192460 due to one unit increase in the credit_amount, keeping other variables constant.
# 4. credit_term = 1.401019, it implies that the odds of getting bad client increases by a factor of 1.401019 due to one unit increase in the credit_term, keeping other variables constant.
# 5. having_children_flg = 1.035859, it implies that the odds of getting bad client increases by a factor of 1.035859 due to one unit increase in the ahaving_children_flg, keeping other variables constant.
# 6. region = 0.998598, it implies that the odds of getting bad client increases by a factor of 0.998598 due to one unit increase in the region, keeping other variables constant.
# 7. income = 0.832696, it implies that the odds of getting bad client increases by a factor of 0.832696 due to one unit increase in the income, keeping other variables constant.
# 8. sex_male = 1.342753, it implies that the odds of getting bad client increases by a factor of 1.342753 due to one unit increase in the sex_male, keeping other variables constant.
# 9. education_Incomplete higher education = 0.45755, it implies that the odds of getting bad client increases by a factor of 0.45755 due to one unit increase in the education_Incomplete higher education, keeping other variables constant.<br>
# 
# and so on...
# 

# In[31]:


y_pred_prob = logreg_full.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
youdens_table = pd.DataFrame({'TPR': tpr,
                             'FPR': fpr,
                             'Threshold': thresholds})
youdens_table['Difference'] = youdens_table.TPR - youdens_table.FPR
youdens_table = youdens_table.sort_values('Difference', ascending = False).reset_index(drop = True)
youdens_table.head()


# **Comments**<br>
# Optimal threshold is 0.076979

# In[32]:


plot_confusion_matrix(logreg_full, 0.07)


# **Comments**
# 1. True negative socre is 246.
# 2. True positive socre is 50.
# 3. False positive socre is 214.
# 4. False negative socre is 7.<br>
# 
# 246 of class 0 are correctly classified and 214 instances of class 0 are incorrectly classified as class 1.<br>
# 7 instances of the class 1 are incorrectly classified as class 0 and 50 instances are correctly classified as class 1.
# 

# In[33]:


get_test_report(logreg_full, 0.07)


# In[34]:


y_pred_prob = logreg_full.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC curve', fontsize = 15)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)
plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(y_test, y_pred_prob),4)))
plt.grid(True)


# In[35]:


update_score_card('Logistic Regression Fullmodel',logreg_full, cutoff=0.07)


# # Improving base model

# ## 1. Data balancing

# In[36]:


Y.value_counts()


# In[37]:


smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, Y)
X_sm = pd.DataFrame(X_sm, columns=X.columns)
y_sm = pd.DataFrame(y_sm, columns=['bad_client_target'])
y_sm.value_counts()


# ## 2. Function definitions

# In[38]:


def plot_roc(model):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1],'r--')
    plt.title('ROC curve ', fontsize = 15)
    plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
    plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)
    plt.text(x = 0.82, y = 0.3, s = ('AUC Score:',round(roc_auc_score(y_test, y_pred_prob),4)))
    plt.grid(True)
    
def plot_confusion_matrix(model):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cbar = False, 
                linewidths = 0.1, annot_kws = {'size':25})
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()
    
def get_train_report(model):
    train_pred = model.predict(X_train)
    return(classification_report(y_train, train_pred))

def get_test_report(model):
    test_pred = model.predict(X_test)
    return(classification_report(y_test, test_pred))

def update_score_card(Model_name,model,cutoff="-"):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    global score_card
    
    score_card = score_card.append({'Model Name':Model_name,
                                    'Probability Cutoff': cutoff,
                                    'AUC Score' : metrics.roc_auc_score(y_test, y_pred_prob),
                                    'Precision Score': metrics.precision_score(y_test, y_pred),
                                    'Recall Score': metrics.recall_score(y_test, y_pred),
                                    'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                                    'Kappa Score':metrics.cohen_kappa_score(y_test, y_pred),
                                    'f1-score': metrics.f1_score(y_test, y_pred)}, 
                                    ignore_index = True)

    return score_card


# ## 2. Scaling

# In[39]:


sc = StandardScaler()
num_scaled = sc.fit_transform(df_num)
df_num_scaled = pd.DataFrame(num_scaled, columns = df_num.columns)

X = pd.concat([df_num_scaled,dummy_var], axis=1)
Y = client['bad_client_target']
smote = SMOTE(random_state=42)
X_sm, Y_sm = smote.fit_resample(X, Y)
X_sm = pd.DataFrame(X_sm, columns=X.columns)
y_sm = pd.DataFrame(y_sm, columns=['bad_client_target'])


# # Logistic regression (Improved)

# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, random_state = 10, test_size = 0.3)

lr = LogisticRegression()
lr_full = lr.fit(X_train, y_train)


# In[41]:


y_pred_prob = lr_full.predict_proba(X_test)[:,1:]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
youdens_table = pd.DataFrame({'TPR': tpr,
                             'FPR': fpr,
                             'Threshold': thresholds})
youdens_table['Difference'] = youdens_table.TPR - youdens_table.FPR
youdens_table = youdens_table.sort_values('Difference', ascending = False).reset_index(drop = True)
youdens_table.head()


# **Comments**<br>
# Optimal threshold is 0.395

# In[42]:


plot_confusion_matrix(lr_full)


# **Comments**
# 1. True negative socre is 338.
# 2. True positive socre is 358.
# 3. False positive socre is 122.
# 4. False negative socre is 99.<br>
# 
# 338 of class 0 are correctly classified and 122 instances of class 0 are incorrectly classified as class 1.<br>
# 99 instances of the class 1 are incorrectly classified as class 0 and 358 instances are correctly classified as class 1.

# In[43]:


print(get_test_report(lr_full))


# In[44]:


plot_roc(lr_full)


# In[45]:


update_score_card('Logistic Regression(balanced data)',lr_full,0.39)


# # Feature Selection

# In[46]:


X = pd.concat([df_num_scaled,dummy_var], axis=1)
Y = client['bad_client_target']
smote = SMOTE(random_state=42)
X_sm, Y_sm = smote.fit_resample(X, Y)
X_sm = pd.DataFrame(X_sm, columns=X.columns)
y_sm = pd.DataFrame(y_sm, columns=['bad_client_target'])
X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, random_state = 10, test_size = 0.3)

from sklearn.feature_selection import RFE
accuracy_score=[]
for i in range(1,39):
    X_train_rfe = X_train
    X_test_rfe = X_test
    logreg = LogisticRegression()
    rfe_model = RFE(estimator = logreg, n_features_to_select = i)
    rfe_model = rfe_model.fit(X_train_rfe, y_train)
    feat_index = pd.Series(data = rfe_model.ranking_, index = X_train_rfe.columns)
    signi_feat_rfe = feat_index[feat_index==1].index
    accuracy_score.append(rfe_model.score(X_train_rfe,y_train))

lis_acc={(i+1,np.round(accuracy_score[i],4))  for i in range(0,38)}
lis_acc


# # Logistic regression (RFE)

# In[47]:


logreg = LogisticRegression()
rfe_model = RFE(estimator = logreg, n_features_to_select = 33)
rfe_model = rfe_model.fit(X_train, y_train)
feat_index = pd.Series(data = rfe_model.ranking_, index = X_train.columns)
signi_feat_rfe = feat_index[feat_index==1].index
print(signi_feat_rfe)


# In[48]:


X_signi =X[signi_feat_rfe]
Y = client['bad_client_target']
smote = SMOTE(random_state=42)
X_sm, Y_sm = smote.fit_resample(X_signi, Y)
X_sm = pd.DataFrame(X_sm, columns=X_signi.columns)
y_sm = pd.DataFrame(y_sm, columns=['bad_client_target'])
X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, random_state = 10, test_size = 0.3)

lr = LogisticRegression()
lr_signi = lr.fit(X_train, y_train) 


# In[49]:


plot_confusion_matrix(lr_signi)


# **Comments**
# 1. True negative socre is 336.
# 2. True positive socre is 354.
# 3. False positive socre is 124.
# 4. False negative socre is 103.<br>
# 
# 336 of class 0 are correctly classified and 124 instances of class 0 are incorrectly classified as class 1.<br>
# 103 instances of the class 1 are incorrectly classified as class 0 and 354 instances are correctly classified as class 1.

# In[50]:


print(get_test_report(lr_signi))


# In[51]:


plot_roc(lr_signi)


# In[52]:


update_score_card('Logistic RFE Regression',lr_signi)


# # Naive Bayes

# In[53]:


X = pd.concat([df_num_scaled,dummy_var], axis=1)
Y = client['bad_client_target']
smote = SMOTE(random_state=42)
X_sm, Y_sm = smote.fit_resample(X, Y)
X_sm = pd.DataFrame(X_sm, columns=X.columns)
y_sm = pd.DataFrame(y_sm, columns=['bad_client_target'])
X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, random_state = 10, test_size = 0.3)


gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)


# In[54]:


plot_confusion_matrix(gnb_model)


# **Comments**
# 1. True negative socre is 28.
# 2. True positive socre is 456.
# 3. False positive socre is 432.
# 4. False negative socre is 1.<br>
# 
# 28 of class 0 are correctly classified and 432 instances of class 0 are incorrectly classified as class 1.<br>
# 1 instance of the class 1 are incorrectly classified as class 0 and 456 instances are correctly classified as class 1.

# In[55]:


print(get_test_report(gnb_model))


# In[56]:


plot_roc(gnb_model)


# In[57]:


update_score_card('Gaussian Naive Bayes',gnb_model)


# # K-Nearest Neighbor

# In[58]:


tuned_paramaters = [{'metric': ['euclidean', 'minkowski'],
                     'n_neighbors': [2, 3, 4, 5, 6, 8, 10]}]
 
knn_classification = KNeighborsClassifier()

rf_grid = GridSearchCV(estimator = knn_classification, 
                       param_grid = tuned_paramaters, 
                       cv = 5)

rf_grid_model = rf_grid.fit(X_train, y_train)
rf_grid_model.best_params_


# In[59]:


knn_classification = KNeighborsClassifier(metric = 'euclidean', n_neighbors = 2)
knn = knn_classification.fit(X_train, y_train)
plot_confusion_matrix(knn)


# **Comments**
# 1. True negative socre is 275.
# 2. True positive socre is 453.
# 3. False positive socre is 185.
# 4. False negative socre is 4.<br>
# 
# 275 of class 0 are correctly classified and 211854 instances of class 0 are incorrectly classified as class 1.<br>
# 4 instances of the class 1 are incorrectly classified as class 0 and 453 instances are correctly classified as class 1.

# In[60]:


print(get_test_report(knn))


# In[61]:


plot_roc(knn)


# In[62]:


update_score_card('KNN classifier',knn)


# # Decision Tree

# In[63]:


decision_tree_classification  = DecisionTreeClassifier(criterion = 'gini',
                                  max_depth = 5,
                                  min_samples_split = 5,
                                  max_leaf_nodes = 6,
                                  random_state = 10)

decision_tree = decision_tree_classification.fit(X_train, y_train)

train_report = get_train_report(decision_tree)
print('Train data:\n', train_report)

test_report = get_test_report(decision_tree)
print('Test data:\n', test_report)


# tuned_paramaters = [{'criterion': ['entropy', 'gini'], 
#                      'max_depth': range(2, 10),
#                      'min_samples_split': range(2,10),
#                      'max_leaf_nodes': range(1, 10)}]
# 
# decision_tree_classification = DecisionTreeClassifier(random_state = 10)
# tree_grid = GridSearchCV(estimator = decision_tree_classification, 
#                          param_grid = tuned_paramaters, 
#                          cv = 5)
# 
# 
# tree_grid_model = tree_grid.fit(X_train, y_train)
# print('Best parameters for decision tree classifier: ', tree_grid_model.best_params_, '\n')

# In[64]:


X = pd.concat([df_num_scaled,dummy_var], axis=1)
X_signi = X[signi_feat_rfe]
Y = client['bad_client_target']
smote = SMOTE(random_state=42)
X_sm, Y_sm = smote.fit_resample(X_signi, Y)
X_sm = pd.DataFrame(X_sm, columns=X_signi.columns)
y_sm = pd.DataFrame(y_sm, columns=['bad_client_target'])
X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, random_state = 10, test_size = 0.3)

decision_tree_classification  = DecisionTreeClassifier(criterion = 'gini',
                                  max_depth = 6,
                                  min_samples_split = 2,
                                  max_leaf_nodes = 8,
                                  random_state = 10)

decision_tree = decision_tree_classification.fit(X_train, y_train)

train_report = get_train_report(decision_tree)
print('Train data:\n', train_report)

test_report = get_test_report(decision_tree)
print('Test data:\n', test_report)


# In[65]:


labels = X_train.columns
from sklearn import tree
tree.plot_tree(decision_tree,filled=True, feature_names = labels, class_names = ["No","Yes"])
plt.show()


# In[66]:


plot_confusion_matrix(decision_tree)


# **Comments**
# 1. True negative socre is 333.
# 2. True positive socre is 402.
# 3. False positive socre is 127.
# 4. False negative socre is 55.<br>
# 
# 333 of class 0 are correctly classified and 127 instances of class 0 are incorrectly classified as class 1.<br>
# 55 instances of the class 1 are incorrectly classified as class 0 and 402 instances are correctly classified as class 1.

# In[67]:


plot_roc(decision_tree)


# In[68]:


update_score_card('Decision Tree',decision_tree)


# # Random Forest

# In[69]:


rf_classification = RandomForestClassifier(n_estimators = 10, random_state = 10)
rf_model = rf_classification.fit(X_train, y_train)

train_report = get_train_report(rf_model)
print('Train data:\n', train_report)

test_report = get_test_report(rf_model)
print('Test data:\n', test_report)


# tuned_paramaters = [{'criterion': ['entropy', 'gini'],
#                      'n_estimators': [10, 30, 50, 70, 90],
#                      'max_depth': [10, 15, 20],
#                      'max_features': ['sqrt', 'log2'],
#                      'min_samples_split': [2, 5, 8, 11],
#                      'min_samples_leaf': [1, 5, 9],
#                      'max_leaf_nodes': [2, 5, 8, 11]}]
# 
# random_forest_classification = RandomForestClassifier(random_state = 10)
# 
# rf_grid = GridSearchCV(estimator = random_forest_classification, 
#                        param_grid = tuned_paramaters, 
#                        cv = 5)
# 
# 
# rf_grid_model = rf_grid.fit(X_train, y_train)
# 
# print('Best parameters for random forest classifier: ', rf_grid_model.best_params_, '\n')

# In[70]:


rf_model = RandomForestClassifier(criterion = 'entropy', 
                                  n_estimators = 90,
                                  max_depth = 10,
                                  max_features = 'sqrt',
                                  max_leaf_nodes = 11,
                                  min_samples_leaf = 1,
                                  min_samples_split = 8,
                                  random_state = 10)

rf_model = rf_model.fit(X_train, y_train)
train_report = get_train_report(rf_model)
print('Train data:\n', train_report)

test_report = get_test_report(rf_model)
print('Test data:\n', test_report)


# In[71]:


plot_confusion_matrix(rf_model)


# **Comments**
# 1. True negative socre is 375.
# 2. True positive socre is 388.
# 3. False positive socre is 85.
# 4. False negative socre is 69.<br>
# 
# 375 of class 0 are correctly classified and 85 instances of class 0 are incorrectly classified as class 1.<br>
# 69 instances of the class 1 are incorrectly classified as class 0 and 388 instances are correctly classified as class 1.

# In[72]:


plot_roc(rf_model)


# In[73]:


update_score_card('Random Forest',rf_model)


# # Boosting: XGBoost

# In[74]:


xgb_model = XGBClassifier(max_depth = 10, gamma = 1)

xgb_model.fit(X_train, y_train)

train_report = get_train_report(xgb_model)
print('Train data:\n', train_report)

test_report = get_test_report(xgb_model)
print('Test data:\n', test_report)


# tuning_parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#                      'max_depth': range(3,10),
#                      'gamma': [0, 1, 2, 3, 4]}
# 
# xgb_model = XGBClassifier()
# xgb_grid = GridSearchCV(estimator = xgb_model, param_grid = tuning_parameters, cv = 3, scoring = 'roc_auc')
# 
# xgb_grid.fit(X_train, y_train)
# 
# print('Best parameters for XGBoost classifier: ', xgb_grid.best_params_, '\n')

# In[75]:


xgb_grid_model = XGBClassifier(learning_rate = 0.3,
                               max_depth = 4,
                              gamma = 0)

xgb_model = xgb_grid_model.fit(X_train, y_train)

train_report = get_train_report(xgb_model)
print('Train data:\n', train_report)

test_report = get_test_report(xgb_model)
print('Test data:\n', test_report)


# In[76]:


plot_confusion_matrix(xgb_model)


# **Comments**
# 1. True negative socre is 433.
# 2. True positive socre is 416.
# 3. False positive socre is 27.
# 4. False negative socre is 41.<br>
# 
# 433 of class 0 are correctly classified and only 27 instances of class 0 are incorrectly classified as class 1.<br>
# Onlt 41 instances of the class 1 are incorrectly classified as class 0 and 416 instances are correctly classified as class 1.

# In[77]:


plot_roc(xgb_model)


# In[78]:


update_score_card('XGBoost',xgb_model)


# # Stacking

# In[79]:


base_learners = [('Decision Tree Model', DecisionTreeClassifier(criterion = 'gini',
                                  max_depth = 6,
                                  min_samples_split = 2,
                                  max_leaf_nodes = 8,
                                  random_state = 10)),
                 ('Random Forest model',RandomForestClassifier(criterion = 'entropy', 
                                  n_estimators = 90,
                                  max_depth = 10,
                                  min_samples_split = 8,
                                  random_state = 10))]

stack_model = StackingClassifier(estimators = base_learners, final_estimator = XGBClassifier())
stack_model.fit(X_train, y_train)

train_report = get_train_report(stack_model)
print('Train data:\n', train_report)

test_report = get_test_report(stack_model)
print('Test data:\n', test_report)


# In[80]:


plot_confusion_matrix(stack_model)


# **Comments**
# 1. True negative socre is 424.
# 2. True positive socre is 423.
# 3. False positive socre is 36.
# 4. False negative socre is 34.<br>
# 
# 424 of class 0 are correctly classified and 36 instances of class 0 are incorrectly classified as class 1.<br>
# 34 instances of the class 1 are incorrectly classified as class 0 and 423 instances are correctly classified as class 1.

# In[81]:


plot_roc(stack_model)


# In[82]:


update_score_card('Stacked model with final estimator as XGBoost',stack_model)


# In[83]:


score_card.style.highlight_max(['AUC Score'], 'gray')


# # Conclusion
# As per the table above XG Boost model gives the best results with an AUC score of 97% and an accuracy score of 92% . However, scores are not only the criteria to decide the best model we also have to take overfitting/underfitting into consideration. XGBoost of course gives best score but is slightly overfitted. Stacked model being 1% less accurate is better balanced than XGBoost and hence, best model is concluded as Stacked model with final estimator as XGBoost.

# In[83]:




