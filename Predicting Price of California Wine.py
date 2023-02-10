#!/usr/bin/env python
# coding: utf-8

# ## Knowledge Discovery and Data Mining
# 
# # Project: Predicting the Price of California Wine
# #### Course : CS-513 A

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, roc_auc_score
from mlxtend.plotting import category_scatter, plot_learning_curves, plot_decision_regions


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv("Californa_Wine_Production_1980_2020.csv")


# In[4]:


data


# In[5]:


data.rename(columns = {'Yield(Unit/Acre)':'Yield', 'Price(Dollars/Unit)':'Price', 'Value(Dollars)':'Value'}, inplace = True)


# In[6]:


data.Unit = "TONS"


# In[7]:


data


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.describe(include='all')


# In[11]:


data.dtypes


# In[12]:


data = data.drop(columns=['CommodityCode','CropName','Unit','Value'])


# In[13]:


data.duplicated().any()


# In[14]:


data.duplicated().value_counts()


# In[15]:


data.isnull().sum()


# In[16]:


data.columns[data.isnull().any()]


# In[17]:


data.isnull().value_counts()


# In[18]:


data[data.HarvestedAcres.isna()].sort_values(['County','Year'])


# In[19]:


data[data.Yield.isna()].sort_values(['County','Year'])


# In[20]:


data[data.Production.isna()].sort_values(['County','Year'])


# In[21]:


data[data.Price.isna()].sort_values(['County','Year'])


# In[22]:


data[data['County'] == 'Nevada']


# In[23]:


data[data['County'] == 'SanMateo']


# In[24]:


data[data['County'] == 'Trinity']


# In[25]:


data.dropna(inplace=True)


# In[26]:


data.isnull().sum()


# In[27]:


data.info()


# <b>Discretizing the data into classes. </b>

# In[28]:


data


# In[29]:


price_data = data.Price


# In[30]:


# len(price_data)
price_data.size


# In[31]:


price_data.sort_values(ascending=True)


# In[32]:


data.sort_values('Price')


# In[33]:


data['Price'].value_counts()


# In[34]:


data['Price'].value_counts().sort_index()


# In[35]:


plt.figure(figsize=(200, 6))
price_data.value_counts().plot(kind = 'bar', title = 'Counts')


# In[36]:


data['Price_Classification'] = pd.cut(x=data['Price'], bins=[0, 250, 1000, 50000], labels=[0, 1, 2])
data['Price_Categories'] = pd.cut(x=data['Price'], bins=[0, 250, 1000, 50000], labels=["Low", "Medium", "High"])


# In[37]:


data.Price_Categories.size


# In[38]:


data.Price_Categories.value_counts()


# In[39]:


data.Price_Categories.value_counts().plot(kind='bar', title='Price Categories')
plt.xticks(rotation=0)
# plt.grid(True)


# In[40]:


data.Price_Categories.value_counts().plot(kind='pie', title='Price Categories')


# In[41]:


data.info()


# Year (From 1980 to 2020) <br>
# CountyCode <br>
# County <br>
# HarvestedAcres <br>
# Yield (Tons/Acres) <br>
# Production (Tons) <br>
# Price (Dollar/Ton) <br>

# $Formula: $ <br>
# $Yield = \frac{Production}{HarvestedAcres}$ <br>

# In[42]:


plt.figure(figsize=(12, 6))
data.boxplot()


# In[43]:


plt.figure(figsize=(12, 6))
sns.boxplot(data)


# In[44]:


data.plot(kind='box', subplots=True, layout=(3, 3), figsize=(20, 18))


# In[45]:


sns.heatmap(data.corr(), annot=True)
plt.tight_layout()


# In[46]:


sns.pairplot(data, hue='Price_Categories', palette='tab10')


# In[47]:


year_group_df = data.groupby(['Year']).mean()


# In[48]:


year_group_df


# In[49]:


year_group_df.index


# # Exploratory Data Analysis:

# In[50]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=data.groupby(['Year']).count().index, y=data.groupby(['Year']).count()['County'])
plt.ylabel(ylabel='Number of counties producing wine')
plt.title(label='Fluctuation in the number of wine-producing counties across time')
plt.show()


# In[51]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=year_group_df.index, y=year_group_df['Production'])
plt.title(label='Fluctuations in Production over Time')
plt.show()


# In[52]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='Year', y='Production')
plt.title(label='Fluctuations in Production over Time')
plt.show()


# In[53]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='Year', y='Production')
plt.xticks(rotation=90)
plt.title(label='Annual Grapes Production in Tons')
plt.show()


# In[54]:


data.sort_values('Production').tail(5)


# In[55]:


sns.barplot(x='Year', y='Production', data=data.sort_values('Production').tail(5))
plt.title(label='Top Five Wine Production Years')


# In[56]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=year_group_df.index, y=year_group_df['HarvestedAcres'])
plt.title(label='Fluctuations in Harvested Acres over Time')
plt.show()


# In[57]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='Year', y='HarvestedAcres')
plt.title(label='Fluctuations in Harvested Acres over Time')
plt.show()


# In[58]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='Year', y='HarvestedAcres')
plt.xticks(rotation=90)
plt.title(label='Annual Grapes Harvested Area in Acres')
plt.show()


# In[59]:


data.sort_values('HarvestedAcres').tail(5)


# In[60]:


sns.barplot(x='Year', y='HarvestedAcres', data=data.sort_values('HarvestedAcres').tail(5))
plt.title(label='Top Five Grapes Harvest Years')


# In[61]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=year_group_df.index, y=year_group_df['Yield'])
plt.title(label='Fluctuations in Yield over Time')
plt.show()


# In[62]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='Year', y='Yield')
plt.title(label='Fluctuations in Yield over Time')
plt.show()


# In[63]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='Year', y='Yield')
plt.xticks(rotation=90)
plt.title(label='Annual Grapes Yield in Tons per Acres')
plt.show()


# In[64]:


data.sort_values('Yield').tail(5)


# In[65]:


sns.barplot(x='Year', y='Yield', data=data.sort_values('Yield').tail(5))
plt.title(label='Top Five Wine Yield Years')


# In[66]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=year_group_df.index, y=year_group_df['Price'])
plt.title(label='Fluctuations in Price over Time')
plt.show()


# In[67]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='Year', y='Price')
plt.title(label='Fluctuations in Price over Time')
plt.show()


# In[68]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='Year', y='Price')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Price per Ton over Time')
plt.show()


# In[69]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Price_Categories', data=data)
plt.xticks(rotation=90)
plt.title(label='Price Categories over Time')
plt.show()


# In[70]:


import plotly.express as px
fig = px.line(data, x='Year', y='Price', color='Price_Categories', symbol="Price_Categories")
fig.show()


# In[71]:


data.sort_values('Price').tail(5)


# In[72]:


sns.barplot(x='Year', y='Price', data=data.sort_values('Price').tail(5))
plt.title(label='Top Five Wine Price Years')


# In[73]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='County', y='Production')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Production over County')
plt.show()


# In[74]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='County', y='Production')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Production over County')
plt.show()


# In[75]:


data.sort_values('Production').tail(1)


# In[76]:


sns.barplot(x='County', y='Production', data=data.sort_values('Production').tail(1))
plt.title(label='Top Wine Production County')


# In[77]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='County', y='HarvestedAcres')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in HarvestedAcres over County')
plt.show()


# In[78]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='County', y='HarvestedAcres')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in HarvestedAcres over County')
plt.show()


# In[79]:


data.sort_values('HarvestedAcres').tail(1)


# In[80]:


sns.barplot(x='County', y='HarvestedAcres', data=data.sort_values('HarvestedAcres').tail(1))
plt.title(label='Top Grapes Harvest County')


# In[81]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='County', y='Yield')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Yield over County')
plt.show()


# In[82]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='County', y='Yield')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Yield over County')
plt.show()


# In[83]:


data.sort_values('Yield').tail(1)


# In[84]:


sns.barplot(x='County', y='Yield', data=data.sort_values('Yield').tail(1))
plt.title(label='Top Wine Yield County')


# In[85]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x='County', y='Price')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Price over County')
plt.show()


# In[86]:


plt.figure(figsize=(12, 6))
sns.color_palette("tab10")
sns.barplot(data, x='County', y='Price')
plt.xticks(rotation=90)
plt.title(label='Fluctuations in Price over County')
plt.show()


# In[87]:


plt.figure(figsize=(12, 6))
sns.countplot(x='County', hue='Price_Categories', data=data)
plt.xticks(rotation=90)
plt.title(label='Price Categories over County')
plt.show()


# In[88]:


data.sort_values('Price').tail(1)


# In[89]:


sns.barplot(x='County', y='Price', data=data.sort_values('Price').tail(1))
plt.title(label='Top Wine Price County')


# In[90]:


fig = category_scatter(x='Yield', y='Production', label_col='Price_Categories', data=data, legend_loc='upper left')
plt.xlabel('Yield')
plt.ylabel('Production')
plt.title(label='Price Categories over Yield and Production')
# plt.grid(True)
plt.show()


# In[91]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data, x='Yield', y='Production', hue="Price_Categories")
plt.title(label='Price Categories over Yield and Production')


# In[92]:


sns.scatterplot(data, x='CountyCode', y='Year', hue="Price_Categories")
plt.title(label='Price Categories over CountyCode and Year')


# In[93]:


plt.figure(figsize=(12, 6))
sns.lineplot(data, x="CountyCode", y="Year", hue="Price_Categories")
plt.title(label='Price Categories over CountyCode and Year')


# <b>Setting up the target variable. </b>

# In[94]:


data.info()


# In[95]:


features = ['Year', 'CountyCode', 'HarvestedAcres', 'Yield', 'Production']
x = data[features]
y = data.Price_Classification


# In[96]:


x


# In[97]:


x.info()


# In[98]:


y = y.astype('int64')
y.info()


# In[99]:


y.value_counts()


# In[100]:


y.value_counts().plot(kind = 'bar', title = 'Price Classification Counts')
plt.xticks(rotation=0)


# In[101]:


sns.heatmap(x.corr(), annot=True)
plt.tight_layout()


# In[102]:


scaler = MinMaxScaler()
# scaler.fit(x)
# scaler.transform(x)
x = scaler.fit_transform(x)


# In[103]:


x


# In[104]:


sns.boxplot(x)


# <b>Split dataset into 80% for training and 20% for testing. </b>

# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[106]:


print("Shape of original dataset:", data.shape)
print("Shape of input training set:", x_train.shape)
print("Shape of output training set:", y_train.shape)
print("Shape of input testing set:", x_test.shape)
print("Shape of output testing set:", y_test.shape)


# In[107]:


y_test.size


# In[108]:


y_test.value_counts()


# <b>Performance Functions for Models</b>

# In[109]:


true_class_names = ["True Low", "True Medium", "True High"]
predicted_class_names = ["Predicted Low", "Predicted Medium", "Predicted High"]


# In[110]:


def Confusion_Matrix_Plotter(cm, dtype):
    cm_df = pd.DataFrame(cm, index = true_class_names, columns = predicted_class_names)
    if dtype == 1:
        sns.heatmap(cm_df, annot=True, fmt="d")
        plt.title('Confusion Matrix')
    else:
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix Percentage')
    plt.tight_layout()


# In[111]:


def Compute_Error(cm):
    n11, n12, n13, n21, n22, n23, n31, n32, n33 = cm.ravel()
    TP_C1 = n11
    TN_C1 = n22 + n33
    FP_C1 = n21 + n31
    FN_C1 = n12 + n13
    Type1_Error_C1 = FP_C1
    Type2_Error_C1 = FN_C1
    print("Type1_Error_LowPrice:", Type1_Error_C1)
    print("Type2_Error_LowPrice:", Type2_Error_C1)
    TP_C2 = n22
    TN_C2 = n11 + n33
    FP_C2 = n12 + n32
    FN_C2 = n21 + n23
    Type1_Error_C2 = FP_C2
    Type2_Error_C2 = FN_C2
    print("Type1_Error_MediumPrice:", Type1_Error_C2)
    print("Type2_Error_MediumPrice:", Type2_Error_C2)
    TP_C3 = n33
    TN_C3 = n11 + n22
    FP_C3 = n13 + n23
    FN_C3 = n31 + n32
    Type1_Error_C3 = FP_C3
    Type2_Error_C3 = FN_C3
    print("Type1_Error_HighPrice:", Type1_Error_C3)
    print("Type2_Error_HighPrice:", Type2_Error_C3)
    return Type1_Error_C1, Type2_Error_C1, Type1_Error_C2, Type2_Error_C2, Type1_Error_C3, Type2_Error_C3


# In[112]:


def Compute_Sensitivity(TP, FN):
    sensitivity_test = (TP / (TP + FN))
    return sensitivity_test


# In[113]:


def Compute_Specificity(TN, FP):
    specificity_test = (TN / (FP + TN))
    return specificity_test


# ### Naive Bayes

# In[114]:


nb = GaussianNB()
nb.fit(x_train, y_train)


# In[115]:


# Model Scores on training and test set
print("Training Set score:", nb.score(x_train, y_train))
print("Test Set score:", nb.score(x_test, y_test))


# In[116]:


# Prediction on Testing Data
y_pred_nb = nb.predict(x_test)
nb_accuracy = metrics.accuracy_score(y_test, y_pred_nb)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_nb))


# In[117]:


# Prediction on Training Data
y_pred2_nb = nb.predict(x_train)
nb_taccuracy = metrics.accuracy_score(y_train, y_pred2_nb)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_nb))


# In[118]:


confusion_matrix_nb = metrics.confusion_matrix(y_test, y_pred_nb)
confusion_matrix_nb


# In[119]:


Confusion_Matrix_Plotter(confusion_matrix_nb, 1)


# In[120]:


confusion_matrix_nb_percent = confusion_matrix_nb.astype('float') / confusion_matrix_nb.sum(axis=1)[:, np.newaxis]
confusion_matrix_nb_percent


# In[121]:


Confusion_Matrix_Plotter(confusion_matrix_nb_percent, 0)


# In[122]:


print(classification_report(y_test, y_pred_nb, target_names=["Low Price", "Medium Price", "High Price"]))


# In[123]:


nb_t1_l, nb_t2_l, nb_t1_m, nb_t2_m, nb_t1_h, nb_t2_h = Compute_Error(confusion_matrix_nb)


# In[124]:


nb_pl, nb_pm, nb_ph = precision_score(y_test, y_pred_nb, average=None)


# In[125]:


nb_rl, nb_rm, nb_rh = recall_score(y_test, y_pred_nb, average=None)


# In[126]:


nb_fl, nb_fm, nb_fh = f1_score(y_test, y_pred_nb, average=None)


# In[127]:


cv_nb = cross_val_score(nb, x_train, y_train, cv = 10, scoring='accuracy')
cv_nb


# In[128]:


cv_nb_m = cv_nb.mean()
print("Cross Validation Score:", cv_nb_m)


# In[129]:


plot_learning_curves(x_train, y_train, x_test, y_test, nb)
plt.show()


# ### Support Vector Classification

# In[130]:


svc = SVC()
svc.fit(x_train, y_train)


# In[131]:


# Model Scores on training and test set
print("Training Set score:", svc.score(x_train, y_train))
print("Test Set score:", svc.score(x_test, y_test))


# In[132]:


# Prediction on Testing Data
y_pred_svc = svc.predict(x_test)
svc_accuracy = metrics.accuracy_score(y_test, y_pred_svc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svc))


# In[133]:


# Prediction on Training Data
y_pred2_svc = svc.predict(x_train)
svc_taccuracy = metrics.accuracy_score(y_train, y_pred2_svc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_svc))


# In[134]:


confusion_matrix_svc = metrics.confusion_matrix(y_test, y_pred_svc)
confusion_matrix_svc


# In[135]:


Confusion_Matrix_Plotter(confusion_matrix_svc, 1)


# In[136]:


confusion_matrix_svc_percent = confusion_matrix_svc.astype('float') / confusion_matrix_svc.sum(axis=1)[:, np.newaxis]
confusion_matrix_svc_percent


# In[137]:


Confusion_Matrix_Plotter(confusion_matrix_svc_percent, 0)


# In[138]:


print(classification_report(y_test, y_pred_svc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[139]:


svc_t1_l, svc_t2_l, svc_t1_m, svc_t2_m, svc_t1_h, svc_t2_h = Compute_Error(confusion_matrix_svc)


# In[140]:


svc_pl, svc_pm, svc_ph = precision_score(y_test, y_pred_svc, average=None)

svc_rl, svc_rm, svc_rh = recall_score(y_test, y_pred_svc, average=None)

svc_fl, svc_fm, svc_fh = f1_score(y_test, y_pred_svc, average=None)


# In[141]:


cv_svc = cross_val_score(svc, x_train, y_train, cv = 10, scoring='accuracy')
cv_svc


# In[142]:


cv_svc_m = cv_svc.mean()
print("Cross Validation Score:", cv_svc_m)


# In[143]:


plot_learning_curves(x_train, y_train, x_test, y_test, svc)
plt.show()


# <b>Applying GridSearchCV. </b>

# In[144]:


svc.get_params()


# In[145]:


param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 
              'degree': [1, 2, 3, 4, 5], 'gamma': [1, 0.1, 0.01, 0.001]}


# In[146]:


param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'sigmoid'], 
              'degree': [1, 2, 3], 'gamma': [1, 0.1, 0.01]}


# In[147]:


svc_gscv = GridSearchCV(SVC(), param_grid, scoring = 'accuracy', cv = 10, refit=True, verbose=1)


# In[148]:


svc_gscv.fit(x_train, y_train)


# In[149]:


# Model Scores on training and test set
print("Training Set score:", svc_gscv.score(x_train, y_train))
print("Test Set score:", svc_gscv.score(x_test, y_test))


# In[150]:


#printing best parameter after tuning
print("GridSearch CV Best Parameters:", svc_gscv.best_params_) 

#printing how our model looks after hyper-parameter tuning
print("\nGridSearch CV Best Estimator:", svc_gscv.best_estimator_)

# best score achieved during the GridSearchCV
print("\nGridSearch CV Best score:", svc_gscv.best_score_)

cv_svc_gscv_b = svc_gscv.best_score_


# In[151]:


# Prediction on Testing Data
y_pred_svc = svc_gscv.predict(x_test)
svc_gscv_accuracy = metrics.accuracy_score(y_test, y_pred_svc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svc))


# In[152]:


# Prediction on Training Data
y_pred2_svc = svc_gscv.predict(x_train)
svc_gscv_taccuracy = metrics.accuracy_score(y_train, y_pred2_svc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_svc))


# In[153]:


confusion_matrix_svc = metrics.confusion_matrix(y_test, y_pred_svc)
confusion_matrix_svc


# In[154]:


Confusion_Matrix_Plotter(confusion_matrix_svc, 1)


# In[155]:


confusion_matrix_svc_percent = confusion_matrix_svc.astype('float') / confusion_matrix_svc.sum(axis=1)[:, np.newaxis]
confusion_matrix_svc_percent


# In[156]:


Confusion_Matrix_Plotter(confusion_matrix_svc_percent, 0)


# In[157]:


print(classification_report(y_test, y_pred_svc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[158]:


svc_gs_t1_l, svc_gs_t2_l, svc_gs_t1_m, svc_gs_t2_m, svc_gs_t1_h, svc_gs_t2_h = Compute_Error(confusion_matrix_svc)


# In[159]:


svc_gs_pl, svc_gs_pm, svc_gs_ph = precision_score(y_test, y_pred_svc, average=None)

svc_gs_rl, svc_gs_rm, svc_gs_rh = recall_score(y_test, y_pred_svc, average=None)

svc_gs_fl, svc_gs_fm, svc_gs_fh = f1_score(y_test, y_pred_svc, average=None)


# ### Logistic Regression

# In[160]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[161]:


# Model Scores on training and test set
print("Training Set score:", logreg.score(x_train, y_train))
print("Test Set score:", logreg.score(x_test, y_test))


# In[162]:


# Prediction on Testing Data
y_pred_lr = logreg.predict(x_test)
lr_accuracy = metrics.accuracy_score(y_test, y_pred_lr)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_lr))


# In[163]:


# Prediction on Training Data
y_pred2_lr = logreg.predict(x_train)
lr_taccuracy = metrics.accuracy_score(y_train, y_pred2_lr)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_lr))


# In[164]:


confusion_matrix_lr = metrics.confusion_matrix(y_test, y_pred_lr)
confusion_matrix_lr


# In[165]:


Confusion_Matrix_Plotter(confusion_matrix_lr, 1)


# In[166]:


confusion_matrix_lr_percent = confusion_matrix_lr.astype('float') / confusion_matrix_lr.sum(axis=1)[:, np.newaxis]
confusion_matrix_lr_percent


# In[167]:


Confusion_Matrix_Plotter(confusion_matrix_lr_percent, 0)


# In[168]:


print(classification_report(y_test, y_pred_lr, target_names=["Low Price", "Medium Price", "High Price"]))


# In[169]:


lr_t1_l, lr_t2_l, lr_t1_m, lr_t2_m, lr_t1_h, lr_t2_h = Compute_Error(confusion_matrix_lr)


# In[170]:


lr_pl, lr_pm, lr_ph = precision_score(y_test, y_pred_lr, average=None)

lr_rl, lr_rm, lr_rh = recall_score(y_test, y_pred_lr, average=None)

lr_fl, lr_fm, lr_fh = f1_score(y_test, y_pred_lr, average=None)


# In[171]:


cv_lr = cross_val_score(logreg, x_train, y_train, cv = 10, scoring='accuracy')
cv_lr


# In[172]:


cv_lr_m = cv_lr.mean()
print("Cross Validation Score:", cv_lr_m)


# In[173]:


plot_learning_curves(x_train, y_train, x_test, y_test, logreg)
plt.show()


# <b>Applying GridSearchCV. </b>

# In[174]:


logreg.get_params()


# In[175]:


parameters = {'penalty': ['l1', 'l2'], 'C': np.logspace(-3, 3, 7), 'solver': ['newton-cg', 'lbfgs', 'liblinear']}


# In[176]:


lr_gs = GridSearchCV(estimator = logreg, param_grid = parameters, scoring = 'accuracy', cv = 10, verbose=1)


# In[177]:


lr_gs.fit(x_train, y_train)


# In[178]:


print("GridSearch CV Best Parameters:", lr_gs.best_params_) 

print("\nGridSearch CV Best Estimator:", lr_gs.best_estimator_)

print("\nGridSearch CV Best score:", lr_gs.best_score_)

cv_lr_gs_b = lr_gs.best_score_


# In[179]:


# Prediction on Testing Data
y_pred_lr = lr_gs.predict(x_test)
lr_gs_accuracy = metrics.accuracy_score(y_test, y_pred_lr)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_lr))


# In[180]:


# Prediction on Training Data
y_pred2_lr = lr_gs.predict(x_train)
lr_gs_taccuracy = metrics.accuracy_score(y_train, y_pred2_lr)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_lr))


# In[181]:


confusion_matrix_lr = metrics.confusion_matrix(y_test, y_pred_lr)
confusion_matrix_lr


# In[182]:


Confusion_Matrix_Plotter(confusion_matrix_lr, 1)


# In[183]:


confusion_matrix_lr_percent = confusion_matrix_lr.astype('float') / confusion_matrix_lr.sum(axis=1)[:, np.newaxis]
confusion_matrix_lr_percent


# In[184]:


Confusion_Matrix_Plotter(confusion_matrix_lr_percent, 0)


# In[185]:


print(classification_report(y_test, y_pred_lr, target_names=["Low Price", "Medium Price", "High Price"]))


# In[186]:


lr_gs_t1_l, lr_gs_t2_l, lr_gs_t1_m, lr_gs_t2_m, lr_gs_t1_h, lr_gs_t2_h = Compute_Error(confusion_matrix_lr)


# In[187]:


lr_gs_pl, lr_gs_pm, lr_gs_ph = precision_score(y_test, y_pred_lr, average=None)

lr_gs_rl, lr_gs_rm, lr_gs_rh = recall_score(y_test, y_pred_lr, average=None)

lr_gs_fl, lr_gs_fm, lr_gs_fh = f1_score(y_test, y_pred_lr, average=None)


# In[188]:


plot_learning_curves(x_train, y_train, x_test, y_test, lr_gs)
plt.show()


# ### AdaBoost Classifier

# In[189]:


abc = AdaBoostClassifier()
abc.fit(x_train, y_train)


# In[190]:


# Model Scores on training and test set
print("Training Set score:", abc.score(x_train, y_train))
print("Test Set score:", abc.score(x_test, y_test))


# In[191]:


# Prediction on Testing Data
y_pred_abc = abc.predict(x_test)
abc_accuracy = metrics.accuracy_score(y_test, y_pred_abc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_abc))


# In[192]:


# Prediction on Training Data
y_pred2_abc = abc.predict(x_train)
abc_taccuracy = metrics.accuracy_score(y_train, y_pred2_abc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_abc))


# In[193]:


confusion_matrix_abc = metrics.confusion_matrix(y_test, y_pred_abc)
confusion_matrix_abc


# In[194]:


Confusion_Matrix_Plotter(confusion_matrix_abc, 1)


# In[195]:


confusion_matrix_abc_percent = confusion_matrix_abc.astype('float') / confusion_matrix_abc.sum(axis=1)[:, np.newaxis]
confusion_matrix_abc_percent


# In[196]:


Confusion_Matrix_Plotter(confusion_matrix_abc_percent, 0)


# In[197]:


print(classification_report(y_test, y_pred_abc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[198]:


abc_t1_l, abc_t2_l, abc_t1_m, abc_t2_m, abc_t1_h, abc_t2_h = Compute_Error(confusion_matrix_abc)


# In[199]:


abc_pl, abc_pm, abc_ph = precision_score(y_test, y_pred_abc, average=None)

abc_rl, abc_rm, abc_rh = recall_score(y_test, y_pred_abc, average=None)

abc_fl, abc_fm, abc_fh = f1_score(y_test, y_pred_abc, average=None)


# In[200]:


cv_abc = cross_val_score(abc, x_train, y_train, cv = 10, scoring='accuracy')
cv_abc


# In[201]:


cv_abc_m = cv_abc.mean()
print("Cross Validation Score:", cv_abc_m)


# In[202]:


plot_learning_curves(x_train, y_train, x_test, y_test, abc)
plt.show()


# ### K-Nearest Neighbours

# In[203]:


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)


# <b>Applying GridSearchCV. </b>

# In[204]:


knn.get_params()


# In[205]:


k_range = list(range(1, 31))


# In[206]:


grid_params = {'n_neighbors': k_range, 'weights': ['uniform', 'distance'], 
               'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']}


# In[207]:


knn_gscv = GridSearchCV(KNeighborsClassifier(), grid_params, cv=10)


# In[208]:


knn_gscv.fit(x_train, y_train)


# In[209]:


# Model Scores on training and test set
print("Training Set score:", knn_gscv.score(x_train, y_train))
print("Test Set score:", knn_gscv.score(x_test, y_test))


# In[210]:


print("GridSearch CV Best Parameters:", knn_gscv.best_params_) 

print("\nGridSearch CV Best Estimator:", knn_gscv.best_estimator_)

print("\nGridSearch CV Best score:", knn_gscv.best_score_)

cv_knn_b = knn_gscv.best_score_


# In[211]:


# Prediction on Testing Data
y_pred_knn = knn_gscv.predict(x_test)
knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))


# In[212]:


# Prediction on Training Data
y_pred2_knn = knn_gscv.predict(x_train)
knn_taccuracy = metrics.accuracy_score(y_train, y_pred2_knn)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_knn))


# In[213]:


confusion_matrix_knn = metrics.confusion_matrix(y_test, y_pred_knn)
confusion_matrix_knn


# In[214]:


Confusion_Matrix_Plotter(confusion_matrix_knn, 1)


# In[215]:


confusion_matrix_knn_percent = confusion_matrix_knn.astype('float') / confusion_matrix_knn.sum(axis=1)[:, np.newaxis]
confusion_matrix_knn_percent


# In[216]:


Confusion_Matrix_Plotter(confusion_matrix_knn_percent, 0)


# In[217]:


print(classification_report(y_test, y_pred_knn, target_names=["Low Price", "Medium Price", "High Price"]))


# In[218]:


knn_t1_l, knn_t2_l, knn_t1_m, knn_t2_m, knn_t1_h, knn_t2_h = Compute_Error(confusion_matrix_knn)


# In[219]:


knn_pl, knn_pm, knn_ph = precision_score(y_test, y_pred_knn, average=None)

knn_rl, knn_rm, knn_rh = recall_score(y_test, y_pred_knn, average=None)

knn_fl, knn_fm, knn_fh = f1_score(y_test, y_pred_knn, average=None)


# In[220]:


plot_learning_curves(x_train, y_train, x_test, y_test, knn_gscv)
plt.show()


# ### Decision Tree

# In[221]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)


# In[222]:


# Model Scores on training and test set
print("Training Set score:", dtc.score(x_train, y_train))
print("Test Set score:", dtc.score(x_test, y_test))


# In[223]:


def Plotter(df):
    plt.figure(figsize = (100, 35))
    print_tree = tree.plot_tree(df,
                       feature_names = features,
                       class_names = ['1','2','3'],
                       rounded = True,
                       filled = True)
    plt.show()


# In[224]:


Plotter(dtc)


# In[225]:


dtc_print_tree = tree.export_text(dtc, feature_names = features)
# print_tree


# In[226]:


# Prediction on Testing Data
y_pred_dtc = dtc.predict(x_test)
dtc_accuracy = metrics.accuracy_score(y_test, y_pred_dtc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_dtc))


# In[227]:


# Prediction on Training Data
y_pred2_dtc = dtc.predict(x_train)
dtc_taccuracy = metrics.accuracy_score(y_train, y_pred2_dtc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_dtc))


# In[228]:


confusion_matrix_dtc = metrics.confusion_matrix(y_test, y_pred_dtc)
confusion_matrix_dtc


# In[229]:


Confusion_Matrix_Plotter(confusion_matrix_dtc, 1)


# In[230]:


confusion_matrix_dtc_percent = confusion_matrix_dtc.astype('float') / confusion_matrix_dtc.sum(axis=1)[:, np.newaxis]
confusion_matrix_dtc_percent


# In[231]:


Confusion_Matrix_Plotter(confusion_matrix_dtc_percent, 0)


# In[232]:


print(classification_report(y_test, y_pred_dtc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[233]:


dt_t1_l, dt_t2_l, dt_t1_m, dt_t2_m, dt_t1_h, dt_t2_h = Compute_Error(confusion_matrix_dtc)


# In[234]:


dt_pl, dt_pm, dt_ph = precision_score(y_test, y_pred_dtc, average=None)

dt_rl, dt_rm, dt_rh = recall_score(y_test, y_pred_dtc, average=None)

dt_fl, dt_fm, dt_fh = f1_score(y_test, y_pred_dtc, average=None)


# In[235]:


cv_dt = cross_val_score(dtc, x_train, y_train, cv = 10, scoring='accuracy')
cv_dt


# In[236]:


cv_dt_m = cv_dt.mean()
print("Cross Validation Score:", cv_dt_m)


# In[237]:


feature_importance = pd.Series(dtc.feature_importances_, index = features).sort_values(ascending = False)

sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.tight_layout()


# In[238]:


plot_learning_curves(x_train, y_train, x_test, y_test, dtc)
plt.show()


# <b>Finding the best parameter max_leaf_nodes using GridSearchCV()</b>

# In[239]:


dtc.get_params()


# In[240]:


leaf_nodes_list = list(range(1, 16))


# In[241]:


parameters = {'max_leaf_nodes': leaf_nodes_list, 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5, 10], 
              'min_samples_leaf': [1, 2, 3, 4, 5, 6]}
dt_gscv = GridSearchCV(estimator = dtc, param_grid = parameters, scoring = 'accuracy', cv = 10)
dt_gscv = dt_gscv.fit(x_train, y_train)
print("Best Parameters:", dt_gscv.best_params_)


# In[242]:


print("GridSearch CV Best Parameters:", dt_gscv.best_params_) 

print("\nGridSearch CV Best Estimator:", dt_gscv.best_estimator_)

print("\nGridSearch CV Best score:", dt_gscv.best_score_)

cv_dt_gscv_b = dt_gscv.best_score_


# In[243]:


nleaf_list = []
score_list = []
for i in range(2, 16):
    nleaf_list.append(i)
    parameters = {'max_leaf_nodes': [i]}
    grid_search = GridSearchCV(estimator = dtc, param_grid = parameters, scoring = 'accuracy', cv = 10)
    grid_search = grid_search.fit(x_train, y_train)
    score_list.append(grid_search.best_score_)

# Plot of tree sizes VS classification rate.
plt.plot(nleaf_list, score_list)
plt.scatter(nleaf_list, score_list)
plt.title("Plot of Tree Size VS Classification Rate")
# plt.grid(True)


# <b>Plotting the Pruned Tree</b>

# In[244]:


dt_gscv.fit(x_train, y_train)


# In[245]:


dtc_pt = dt_gscv


# In[246]:


dtc_pt2 = DecisionTreeClassifier(max_leaf_nodes = 11, min_samples_leaf = 6)
dtc_pt2.fit(x_train, y_train)
Plotter(dtc_pt2)


# In[247]:


# Prediction on Testing Data
y_pred_dtc_pt = dtc_pt.predict(x_test)
dtc_pt_accuracy = metrics.accuracy_score(y_test, y_pred_dtc_pt)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_dtc_pt))


# In[248]:


# Prediction on Training Data
y_pred2_dtc_pt = dtc_pt.predict(x_train)
dtc_pt_taccuracy = metrics.accuracy_score(y_train, y_pred2_dtc_pt)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_dtc_pt))


# In[249]:


confusion_matrix_dtc_pt = metrics.confusion_matrix(y_test, y_pred_dtc_pt)
confusion_matrix_dtc_pt


# In[250]:


# cm_df = pd.DataFrame(confusion_matrix_dtc_pt)
# sns.heatmap(cm_df, annot=True, fmt="d")
# plt.tight_layout()


# In[251]:


Confusion_Matrix_Plotter(confusion_matrix_dtc_pt, 1)


# In[252]:


confusion_matrix_dtc_pt_percent = confusion_matrix_dtc_pt.astype('float') / confusion_matrix_dtc_pt.sum(axis=1)[:, np.newaxis]
confusion_matrix_dtc_pt_percent


# In[253]:


Confusion_Matrix_Plotter(confusion_matrix_dtc_pt_percent, 0)


# In[254]:


print(classification_report(y_test, y_pred_dtc_pt, target_names=["Low Price", "Medium Price", "High Price"]))


# In[255]:


dtp_t1_l, dtp_t2_l, dtp_t1_m, dtp_t2_m, dtp_t1_h, dtp_t2_h = Compute_Error(confusion_matrix_dtc_pt)


# In[256]:


dtp_pl, dtp_pm, dtp_ph = precision_score(y_test, y_pred_dtc_pt, average=None)

dtp_rl, dtp_rm, dtp_rh = recall_score(y_test, y_pred_dtc_pt, average=None)

dtp_fl, dtp_fm, dtp_fh = f1_score(y_test, y_pred_dtc_pt, average=None)


# In[257]:


feature_importance = pd.Series(dtc_pt2.feature_importances_, index = features).sort_values(ascending = False)

sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.tight_layout()


# In[258]:


plot_learning_curves(x_train, y_train, x_test, y_test, dtc_pt2) #dtc_pt2 dt_gscv
plt.show()


# ### Random Forest

# <b>Training a Random Forest with the best parameter max_leaf_nodes using RandomForestClassifier()</b>

# In[259]:


rfc = RandomForestClassifier(max_leaf_nodes = 11, n_estimators = 100)
rfc.fit(x_train, y_train)


# In[260]:


# Model Scores on training and test set
print("Training Set score:", rfc.score(x_train, y_train))
print("Test Set score:", rfc.score(x_test, y_test))


# In[261]:


# Prediction on Testing Data
y_pred_rfc = rfc.predict(x_test)
rfc_accuracy = metrics.accuracy_score(y_test, y_pred_rfc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rfc))


# In[262]:


# Prediction on Training Data
y_pred2_rfc = rfc.predict(x_train)
rfc_taccuracy = metrics.accuracy_score(y_train, y_pred2_rfc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_rfc))


# In[263]:


confusion_matrix_rfc = metrics.confusion_matrix(y_test, y_pred_rfc)
confusion_matrix_rfc


# In[264]:


Confusion_Matrix_Plotter(confusion_matrix_rfc, 1)


# In[265]:


confusion_matrix_rfc_percent = confusion_matrix_rfc.astype('float') / confusion_matrix_rfc.sum(axis=1)[:, np.newaxis]
confusion_matrix_rfc_percent


# In[266]:


Confusion_Matrix_Plotter(confusion_matrix_rfc_percent, 0)


# In[267]:


print(classification_report(y_test, y_pred_rfc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[268]:


rfc_t1_l, rfc_t2_l, rfc_t1_m, rfc_t2_m, rfc_t1_h, rfc_t2_h = Compute_Error(confusion_matrix_rfc)


# In[269]:


rfc_pl, rfc_pm, rfc_ph = precision_score(y_test, y_pred_rfc, average=None)

rfc_rl, rfc_rm, rfc_rh = recall_score(y_test, y_pred_rfc, average=None)

rfc_fl, rfc_fm, rfc_fh = f1_score(y_test, y_pred_rfc, average=None)


# In[270]:


cv_rfc = cross_val_score(rfc, x_train, y_train, cv = 10, scoring='accuracy')
cv_rfc


# In[271]:


cv_rfc_m = cv_rfc.mean()
print("Cross Validation Score:", cv_rfc_m)


# In[272]:


feature_importance = pd.Series(rfc.feature_importances_, index = features).sort_values(ascending = False)

sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.tight_layout()


# In[273]:


plot_learning_curves(x_train, y_train, x_test, y_test, rfc)
plt.show()


# ### Random Forest with Randomized Search CV

# In[274]:


rfc.get_params()


# In[275]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
n_estimators


# In[276]:


# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
max_depth


# In[277]:


leaf_nodes_list = list(range(1, 16))


# In[278]:


random_grid = {'n_estimators': n_estimators, 'max_features': ['auto', 'sqrt'],
               'max_depth': max_depth, 'max_leaf_nodes': leaf_nodes_list,
               'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 3, 4, 5, 6], 'bootstrap': [True, False]}


# In[279]:


rfc_rscv = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, cv = 10, 
                              verbose=1, n_jobs = -1)


# In[280]:


rfc_rscv.fit(x_train, y_train)


# In[281]:


print("GridSearch CV Best Parameters:", rfc_rscv.best_params_) 

print("\nGridSearch CV Best Estimator:", rfc_rscv.best_estimator_)

print("\nGridSearch CV Best score:", rfc_rscv.best_score_)

cv_rfc_rscv_b = rfc_rscv.best_score_


# In[282]:


# Model Scores on training and test set
print("Training Set score:", rfc_rscv.score(x_train, y_train))
print("Test Set score:", rfc_rscv.score(x_test, y_test))


# In[283]:


# Prediction on Testing Data
y_pred_rfc_rscv = rfc_rscv.predict(x_test)
rfc_rscv_accuracy = metrics.accuracy_score(y_test, y_pred_rfc_rscv)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rfc_rscv))


# In[284]:


# Prediction on Training Data
y_pred2_rfc_rscv = rfc_rscv.predict(x_train)
rfc_rscv_taccuracy = metrics.accuracy_score(y_train, y_pred2_rfc_rscv)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_rfc_rscv))


# In[285]:


confusion_matrix_rfc_rscv = metrics.confusion_matrix(y_test, y_pred_rfc_rscv)
confusion_matrix_rfc_rscv


# In[286]:


Confusion_Matrix_Plotter(confusion_matrix_rfc_rscv, 1)


# In[287]:


confusion_matrix_rfc_rscv_percent = confusion_matrix_rfc_rscv.astype('float') / confusion_matrix_rfc_rscv.sum(axis=1)[:, np.newaxis]
confusion_matrix_rfc_rscv_percent


# In[288]:


Confusion_Matrix_Plotter(confusion_matrix_rfc_rscv_percent, 0)


# In[289]:


print(classification_report(y_test, y_pred_rfc_rscv, target_names=["Low Price", "Medium Price", "High Price"]))


# In[290]:


rfc_rscv_t1_l, rfc_rscv_t2_l, rfc_rscv_t1_m, rfc_rscv_t2_m, rfc_rscv_t1_h, rfc_rscv_t2_h = Compute_Error(confusion_matrix_rfc_rscv)


# In[291]:


rfc_rscv_pl, rfc_rscv_pm, rfc_rscv_ph = precision_score(y_test, y_pred_rfc_rscv, average=None)

rfc_rscv_rl, rfc_rscv_rm, rfc_rscv_rh = recall_score(y_test, y_pred_rfc_rscv, average=None)

rfc_rscv_fl, rfc_rscv_fm, rfc_rscv_fh = f1_score(y_test, y_pred_rfc_rscv, average=None)


# In[292]:


rfc2 = RandomForestClassifier(max_leaf_nodes=14, min_samples_leaf=5, min_samples_split=10, n_estimators=400)
rfc2.fit(x_train, y_train)


# In[293]:


feature_importance = pd.Series(rfc2.feature_importances_, index = features).sort_values(ascending = False)

sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.tight_layout()


# In[294]:


# plot_learning_curves(x_train, y_train, x_test, y_test, rfc_rscv)
# plt.show()


# ### Artificial Neural Networks 

# hidden_layer_sizes : This parameter allows us to set the number of layers and the number of nodes we wish to have in the Neural Network Classifier. Each element in the tuple represents the number of nodes at the ith position where i is the index of the tuple. Thus the length of tuple denotes the total number of hidden layers in the network.

# In[295]:


mlp = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(48, 24, 12), alpha=1e-06, max_iter=5000)


# In[296]:


mlp.fit(x_train, y_train)


# In[297]:


# Prediction on Testing Data
y_pred_mlp = mlp.predict(x_test)
# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP) 
mlp_accuracy = metrics.accuracy_score(y_test, y_pred_mlp)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_mlp))


# In[298]:


# Prediction on Training Data
y_pred2_mlp = mlp.predict(x_train)
mlp_taccuracy = metrics.accuracy_score(y_train, y_pred2_mlp)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_mlp))


# In[299]:


# Mean accuracy on the given test data and label
mlp.score(x_test, y_pred_mlp)


# In[300]:


# Model Scores on training and test set
print("Training Set score:", mlp.score(x_train, y_train))
print("Test Set score:", mlp.score(x_test, y_test))


# In[301]:


y_pred_mlp


# In[302]:


print("Number of layers:", mlp.n_layers_)
print("Number of iterations the solver has run:", mlp.n_iter_)
print("Computed Loss:", mlp.loss_)
print("Minimum loss reached by the solver throughout fitting:", mlp.best_loss_)
print("Number of features seen during fit:", mlp.n_features_in_)
print("Output activation function:", mlp.out_activation_) #logistic sigmoid function: returns f(x) = 1 / (1 + exp(-x)).


# In[303]:


confusion_matrix_mlp = metrics.confusion_matrix(y_test, y_pred_mlp)
confusion_matrix_mlp


# In[304]:


Confusion_Matrix_Plotter(confusion_matrix_mlp, 1)


# In[305]:


confusion_matrix_mlp_percent = confusion_matrix_mlp.astype('float') / confusion_matrix_mlp.sum(axis=1)[:, np.newaxis]
confusion_matrix_mlp_percent


# In[306]:


Confusion_Matrix_Plotter(confusion_matrix_mlp_percent, 0)


# In[307]:


print(classification_report(y_test, y_pred_mlp, target_names=["Low Price", "Medium Price", "High Price"]))


# In[308]:


mlp1_t1_l, mlp1_t2_l, mlp1_t1_m, mlp1_t2_m, mlp1_t1_h, mlp1_t2_h = Compute_Error(confusion_matrix_mlp)


# In[309]:


mlp1_pl, mlp1_pm, mlp1_ph = precision_score(y_test, y_pred_mlp, average=None)

mlp1_rl, mlp1_rm, mlp1_rh = recall_score(y_test, y_pred_mlp, average=None)

mlp1_fl, mlp1_fm, mlp1_fh = f1_score(y_test, y_pred_mlp, average=None)


# In[310]:


plt.plot(mlp.loss_curve_)
plt.title("Loss Curve")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# <b>Neural Network with a different network structure</b>

# In[311]:


mlp2 = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(500, 250), alpha=1e-08, max_iter=5000)
mlp2.fit(x_train, y_train)


# In[312]:


# Prediction on Testing Data
y_pred_mlp2 = mlp2.predict(x_test)
mlp2_accuracy = metrics.accuracy_score(y_test, y_pred_mlp2)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_mlp2))


# In[313]:


# Prediction on Training Data
y_pred2_mlp2 = mlp2.predict(x_train)
mlp2_taccuracy = metrics.accuracy_score(y_train, y_pred2_mlp2)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_mlp2))


# In[314]:


# Mean accuracy on the given test data and label
mlp2.score(x_test, y_pred_mlp2)


# In[315]:


# Model Scores on training and test set
print("Training Set score:", mlp2.score(x_train, y_train))
print("Test Set score:", mlp2.score(x_test, y_test))


# In[316]:


y_pred_mlp2


# In[317]:


print("Number of layers:", mlp2.n_layers_)
print("Number of iterations the solver has run:", mlp2.n_iter_)
print("Computed Loss:", mlp2.loss_)
print("Minimum loss reached by the solver throughout fitting:", mlp2.best_loss_)
print("Number of features seen during fit:", mlp2.n_features_in_)
print("Output activation function:", mlp2.out_activation_)


# In[318]:


confusion_matrix_mlp2 = metrics.confusion_matrix(y_test, y_pred_mlp2)
confusion_matrix_mlp2


# In[319]:


Confusion_Matrix_Plotter(confusion_matrix_mlp2, 1)


# In[320]:


confusion_matrix_mlp2_percent = confusion_matrix_mlp2.astype('float') / confusion_matrix_mlp2.sum(axis=1)[:, np.newaxis]
confusion_matrix_mlp2_percent


# In[321]:


Confusion_Matrix_Plotter(confusion_matrix_mlp2_percent, 0)


# In[322]:


print(classification_report(y_test, y_pred_mlp2, target_names=["Low Price", "Medium Price", "High Price"]))


# In[323]:


mlp2_t1_l, mlp2_t2_l, mlp2_t1_m, mlp2_t2_m, mlp2_t1_h, mlp2_t2_h = Compute_Error(confusion_matrix_mlp2)


# In[324]:


mlp2_pl, mlp2_pm, mlp2_ph = precision_score(y_test, y_pred_mlp2, average=None)

mlp2_rl, mlp2_rm, mlp2_rh = recall_score(y_test, y_pred_mlp2, average=None)

mlp2_fl, mlp2_fm, mlp2_fh = f1_score(y_test, y_pred_mlp2, average=None)


# In[325]:


plt.plot(mlp2.loss_curve_)
plt.title("Loss Curve")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# In[326]:


improvement_mlp = mlp2_accuracy - mlp_accuracy
improvement_mlp * 100


# ### XGBoost

# In[327]:


from xgboost import XGBClassifier


# In[328]:


xgbc = XGBClassifier()


# In[329]:


print(xgbc)


# In[330]:


xgbc.fit(x_train, y_train)


# In[331]:


# Model Scores on training and test set
print("Training Set score:", xgbc.score(x_train, y_train))
print("Test Set score:", xgbc.score(x_test, y_test))


# In[332]:


# Prediction on Testing Data
y_pred_xgbc = xgbc.predict(x_test)
xgbc_accuracy = metrics.accuracy_score(y_test, y_pred_xgbc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_xgbc))


# In[333]:


# Prediction on Training Data
y_pred2_xgbc = xgbc.predict(x_train)
xgbc_taccuracy = metrics.accuracy_score(y_train, y_pred2_xgbc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_xgbc))


# In[334]:


confusion_matrix_xgbc = metrics.confusion_matrix(y_test, y_pred_xgbc)
confusion_matrix_xgbc


# In[335]:


Confusion_Matrix_Plotter(confusion_matrix_xgbc, 1)


# In[336]:


confusion_matrix_xgbc_percent = confusion_matrix_xgbc.astype('float') / confusion_matrix_xgbc.sum(axis=1)[:, np.newaxis]
confusion_matrix_xgbc_percent


# In[337]:


Confusion_Matrix_Plotter(confusion_matrix_xgbc_percent, 0)


# In[338]:


print(classification_report(y_test, y_pred_xgbc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[339]:


xgbc_t1_l, xgbc_t2_l, xgbc_t1_m, xgbc_t2_m, xgbc_t1_h, xgbc_t2_h = Compute_Error(confusion_matrix_xgbc)


# In[340]:


xgbc_pl, xgbc_pm, xgbc_ph = precision_score(y_test, y_pred_xgbc, average=None)

xgbc_rl, xgbc_rm, xgbc_rh = recall_score(y_test, y_pred_xgbc, average=None)

xgbc_fl, xgbc_fm, xgbc_fh = f1_score(y_test, y_pred_xgbc, average=None)


# In[341]:


cv_xgbc = cross_val_score(xgbc, x_train, y_train, cv = 10, scoring='accuracy')
cv_xgbc


# In[342]:


cv_xgbc_m = cv_xgbc.mean()
print("Cross Validation Score:", cv_xgbc_m)


# In[343]:


plot_learning_curves(x_train, y_train, x_test, y_test, xgbc)
plt.show()


# ### Stochastic Gradient Descent 

# In[344]:


from sklearn.linear_model import SGDClassifier


# In[345]:


sgd = SGDClassifier()


# In[346]:


sgd.get_params()


# In[347]:


sgd.fit(x_train, y_train)


# In[348]:


# Model Scores on training and test set
print("Training Set score:", sgd.score(x_train, y_train))
print("Test Set score:", sgd.score(x_test, y_test))


# In[349]:


# Prediction on Testing Data
y_pred_sgd = sgd.predict(x_test)
sgd_accuracy = metrics.accuracy_score(y_test, y_pred_sgd)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_sgd))


# In[350]:


# Prediction on Training Data
y_pred2_sgd = sgd.predict(x_train)
sgd_taccuracy = metrics.accuracy_score(y_train, y_pred2_sgd)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_sgd))


# In[351]:


confusion_matrix_sgd = metrics.confusion_matrix(y_test, y_pred_sgd)
confusion_matrix_sgd


# In[352]:


Confusion_Matrix_Plotter(confusion_matrix_sgd, 1)


# In[353]:


confusion_matrix_sgdc_percent = confusion_matrix_sgd.astype('float') / confusion_matrix_sgd.sum(axis=1)[:, np.newaxis]
confusion_matrix_sgdc_percent


# In[354]:


Confusion_Matrix_Plotter(confusion_matrix_sgdc_percent, 0)


# In[355]:


print(classification_report(y_test, y_pred_sgd, target_names=["Low Price", "Medium Price", "High Price"]))


# In[356]:


sgd_t1_l, sgd_t2_l, sgd_t1_m, sgd_t2_m, sgd_t1_h, sgd_t2_h = Compute_Error(confusion_matrix_sgd)


# In[357]:


sgd_pl, sgd_pm, sgd_ph = precision_score(y_test, y_pred_sgd, average=None)

sgd_rl, sgd_rm, sgd_rh = recall_score(y_test, y_pred_sgd, average=None)

sgd_fl, sgd_fm, sgd_fh = f1_score(y_test, y_pred_sgd, average=None)


# In[358]:


cv_sgd = cross_val_score(sgd, x_train, y_train, cv = 10, scoring='accuracy')
cv_sgd


# In[359]:


cv_sgd_m = cv_sgd.mean()
print("Cross Validation Score:", cv_sgd_m)


# In[360]:


plot_learning_curves(x_train, y_train, x_test, y_test, sgd)
plt.show()


# ### Gradient Boosting Classifier

# In[361]:


from sklearn.ensemble import GradientBoostingClassifier


# In[362]:


gbc = GradientBoostingClassifier()


# In[363]:


gbc.get_params()


# In[364]:


gbc.fit(x_train, y_train)


# In[365]:


# Model Scores on training and test set
print("Training Set score:", gbc.score(x_train, y_train))
print("Test Set score:", gbc.score(x_test, y_test))


# In[366]:


# Prediction on Testing Data
y_pred_gbc = gbc.predict(x_test)
gbc_accuracy = metrics.accuracy_score(y_test, y_pred_gbc)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_gbc))


# In[367]:


# Prediction on Training Data
y_pred2_gbc = gbc.predict(x_train)
gbc_taccuracy = metrics.accuracy_score(y_train, y_pred2_gbc)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred2_gbc))


# In[368]:


confusion_matrix_gbc = metrics.confusion_matrix(y_test, y_pred_gbc)
confusion_matrix_gbc


# In[369]:


Confusion_Matrix_Plotter(confusion_matrix_gbc, 1)


# In[370]:


confusion_matrix_gbc_percent = confusion_matrix_gbc.astype('float') / confusion_matrix_gbc.sum(axis=1)[:, np.newaxis]
confusion_matrix_gbc_percent


# In[371]:


Confusion_Matrix_Plotter(confusion_matrix_gbc_percent, 0)


# In[372]:


print(classification_report(y_test, y_pred_gbc, target_names=["Low Price", "Medium Price", "High Price"]))


# In[373]:


gbc_t1_l, gbc_t2_l, gbc_t1_m, gbc_t2_m, gbc_t1_h, gbc_t2_h = Compute_Error(confusion_matrix_gbc)


# In[374]:


gbc_pl, gbc_pm, gbc_ph = precision_score(y_test, y_pred_gbc, average=None)

gbc_rl, gbc_rm, gbc_rh = recall_score(y_test, y_pred_gbc, average=None)

gbc_fl, gbc_fm, gbc_fh = f1_score(y_test, y_pred_gbc, average=None)


# In[375]:


cv_gbc = cross_val_score(gbc, x_train, y_train, cv = 10, scoring='accuracy')
cv_gbc


# In[376]:


cv_gbc_m = cv_gbc.mean()
print("Cross Validation Score:", cv_sgd_m)


# In[377]:


plot_learning_curves(x_train, y_train, x_test, y_test, gbc)
plt.show()


# ## Comparing Performance of Various Models

# In[378]:


models_error = [('Naive Bayes', nb_t1_l, nb_t2_l, nb_t1_m, nb_t2_m, nb_t1_h, nb_t2_h),
('Support Vector Classification', svc_t1_l, svc_t2_l, svc_t1_m, svc_t2_m, svc_t1_h, svc_t2_h),
('Support Vector Classification with GridSearchCV', svc_gs_t1_l, svc_gs_t2_l, svc_gs_t1_m, svc_gs_t2_m, svc_gs_t1_h, svc_gs_t2_h),
('Logistic Regression', lr_t1_l, lr_t2_l, lr_t1_m, lr_t2_m, lr_t1_h, lr_t2_h),
('Logistic Regression with GridSearchCV', lr_gs_t1_l, lr_gs_t2_l, lr_gs_t1_m, lr_gs_t2_m, lr_gs_t1_h, lr_gs_t2_h),
('AdaBoost Classifier', abc_t1_l, abc_t2_l, abc_t1_m, abc_t2_m, abc_t1_h, abc_t2_h),
('K-Nearest Neighbors with GridSearchCV', knn_t1_l, knn_t2_l, knn_t1_m, knn_t2_m, knn_t1_h, knn_t2_h),
('Decision Trees', dt_t1_l, dt_t2_l, dt_t1_m, dt_t2_m, dt_t1_h, dt_t2_h),
('Decision Trees with GridSearchCV', dtp_t1_l, dtp_t2_l, dtp_t1_m, dtp_t2_m, dtp_t1_h, dtp_t2_h),
('Random Forest Classifier', rfc_t1_l, rfc_t2_l, rfc_t1_m, rfc_t2_m, rfc_t1_h, rfc_t2_h),
('Random Forest with RandomizedSearchCV', rfc_rscv_t1_l, rfc_rscv_t2_l, rfc_rscv_t1_m, rfc_rscv_t2_m, rfc_rscv_t1_h, rfc_rscv_t2_h),
('Neural Network (48, 24, 12)', mlp1_t1_l, mlp1_t2_l, mlp1_t1_m, mlp1_t2_m, mlp1_t1_h, mlp1_t2_h),
('Neural Network (500, 250)', mlp2_t1_l, mlp2_t2_l, mlp2_t1_m, mlp2_t2_m, mlp2_t1_h, mlp2_t2_h),
('XGBoost Classifier', xgbc_t1_l, xgbc_t2_l, xgbc_t1_m, xgbc_t2_m, xgbc_t1_h, xgbc_t2_h),
('Stochastic Gradient Descent', sgd_t1_l, sgd_t2_l, sgd_t1_m, sgd_t2_m, sgd_t1_h, sgd_t2_h),
('Gradient Boosting Classifier', gbc_t1_l, gbc_t2_l, gbc_t1_m, gbc_t2_m, gbc_t1_h, gbc_t2_h)]


# In[379]:


error_data = pd.DataFrame(data = models_error, columns=['Model', 'Type1 Error Low Price', 'Type2 Error Low Price', 
                                                        'Type1 Error Medium Price', 'Type2 Error Medium Price', 
                                                        'Type1 Error High Price', 'Type2 Error High Price'])
error_data


# In[380]:


models_score = [('Naive Bayes', nb_pl, nb_pm, nb_ph, nb_rl, nb_rm, nb_rh, nb_fl, nb_fm, nb_fh),
                ('Support Vector Classification', svc_pl, svc_pm, svc_ph, svc_rl, svc_rm, svc_rh, svc_fl, svc_fm, svc_fh),
                ('Support Vector Classification with GridSearchCV', svc_gs_pl, svc_gs_pm, svc_gs_ph, svc_gs_rl, svc_gs_rm, svc_gs_rh, svc_gs_fl, svc_gs_fm, svc_gs_fh),
                ('Logistic Regression', lr_pl, lr_pm, lr_ph, lr_rl, lr_rm, lr_rh, lr_fl, lr_fm, lr_fh),
                ('Logistic Regression with GridSearchCV', lr_gs_pl, lr_gs_pm, lr_gs_ph, lr_gs_rl, lr_gs_rm, lr_gs_rh, lr_gs_fl, lr_gs_fm, lr_gs_fh),
                ('AdaBoost Classifier', abc_pl, abc_pm, abc_ph, abc_rl, abc_rm, abc_rh, abc_fl, abc_fm, abc_fh),
                ('K-Nearest Neighbors with GridSearchCV', knn_pl, knn_pm, knn_ph, knn_rl, knn_rm, knn_rh, knn_fl, knn_fm, knn_fh),
                ('Decision Trees', dt_pl, dt_pm, dt_ph, dt_rl, dt_rm, dt_rh, dt_fl, dt_fm, dt_fh),
                ('Decision Trees with GridSearchCV', dtp_pl, dtp_pm, dtp_ph, dtp_rl, dtp_rm, dtp_rh, dtp_fl, dtp_fm, dtp_fh),
                ('Random Forest Classifier', rfc_pl, rfc_pm, rfc_ph, rfc_rl, rfc_rm, rfc_rh, rfc_fl, rfc_fm, rfc_fh),
                ('Random Forest with RandomizedSearchCV', rfc_rscv_pl, rfc_rscv_pm, rfc_rscv_ph, rfc_rscv_rl, rfc_rscv_rm, rfc_rscv_rh, rfc_rscv_fl, rfc_rscv_fm, rfc_rscv_fh),
                ('Neural Network (48, 24, 12)', mlp1_pl, mlp1_pm, mlp1_ph, mlp1_rl, mlp1_rm, mlp1_rh, mlp1_fl, mlp1_fm, mlp1_fh),
                ('Neural Network (500, 250)', mlp2_pl, mlp2_pm, mlp2_ph, mlp2_rl, mlp2_rm, mlp2_rh, mlp2_fl, mlp2_fm, mlp2_fh),
                ('XGBoost Classifier', xgbc_pl, xgbc_pm, xgbc_ph, xgbc_rl, xgbc_rm, xgbc_rh, xgbc_fl, xgbc_fm, xgbc_fh),
                ('Stochastic Gradient Descent', sgd_pl, sgd_pm, sgd_ph, sgd_rl, sgd_rm, sgd_rh, sgd_fl, sgd_fm, sgd_fh),
                ('Gradient Boosting Classifier', gbc_pl, gbc_pm, gbc_ph, gbc_rl, gbc_rm, gbc_rh, gbc_fl, gbc_fm, gbc_fh)]


# In[381]:


score_performance = pd.DataFrame(data=models_score,
                                 columns=['Model', 'Precision Score Low Price', 'Precision Score Medium Price', 'Precision Score High Price',
                                          'Recall Score Low Price', 'Recall Score Medium Price', 'Recall Score High Price',
                                          'F1 Score Low Price', 'F1 Score Medium Price', 'F1 Score High Price'])
score_performance


# In[382]:


models = [('Naive Bayes', nb_accuracy, nb_taccuracy, cv_nb_m),
          ('Support Vector Classification', svc_accuracy, svc_taccuracy, cv_svc_m),
          ('Support Vector Classification with GridSearchCV', svc_gscv_accuracy, svc_gscv_taccuracy, cv_svc_gscv_b),
          ('Logistic Regression', lr_accuracy, lr_taccuracy, cv_lr_m),
          ('Logistic Regression with GridSearchCV', lr_gs_accuracy, lr_gs_taccuracy, cv_lr_gs_b),
          ('AdaBoost Classifier', abc_accuracy, abc_taccuracy, cv_abc_m),
          ('K-Nearest Neighbors with GridSearchCV', knn_accuracy, knn_taccuracy, cv_knn_b),
          ('Decision Trees', dtc_accuracy, dtc_taccuracy, cv_dt_m),
          ('Decision Trees with GridSearchCV', dtc_pt_accuracy, dtc_pt_taccuracy, cv_dt_gscv_b),
          ('Random Forest Classifier', rfc_accuracy, rfc_taccuracy, cv_rfc_m),
          ('Random Forest with RandomizedSearchCV', rfc_rscv_accuracy, rfc_rscv_taccuracy, cv_rfc_rscv_b),
          ('XGBoost Classifier', xgbc_accuracy, xgbc_taccuracy, cv_xgbc_m),
          ('Stochastic Gradient Descent', sgd_accuracy, sgd_taccuracy, cv_sgd_m),
          ('Gradient Boosting Classifier', gbc_accuracy, gbc_taccuracy, cv_gbc_m),
          ('Neural Network (48, 24, 12)', mlp_accuracy, mlp_taccuracy, "None"),
          ('Neural Network (500, 250)', mlp2_accuracy, mlp2_taccuracy, "None")]


# In[383]:


performance = pd.DataFrame(data=models, columns=['Model', 'Accuracy(Test Set)', 'Accuracy(Training Set)', 'Cross-Validation'])
performance


# In[384]:


performance.info()


# In[385]:


performance['Cross-Validation'][:14]


# In[386]:


performance['Cross-Validation'] = performance['Cross-Validation'][:14].astype('float64')


# In[387]:


f, axe = plt.subplots(1, 1, figsize=(10, 6))
performance.sort_values(by=['Cross-Validation'][:14], ascending=False, inplace=True)
sns.barplot(x='Cross-Validation', y='Model', data=performance[:14], ax=axe)
axe.set_xlabel('Cross-Validaton Score', size=14)
axe.set_ylabel('Models', size=14)
axe.set_xlim(0, 1.0)
axe.set_xticks(np.arange(0, 1.1, 0.1))
# plt.title("Cross-Validaton Score Plot")
plt.tight_layout()


# In[388]:


f, axes = plt.subplots(2, 1, figsize=(12, 10))

performance.sort_values(by=['Accuracy(Training Set)'], ascending=False, inplace=True)
sns.barplot(x='Accuracy(Training Set)', y='Model', data=performance, palette='Blues_d', ax=axes[0])
axes[0].set_xlabel('Accuracy (Training Set)', size=14)
axes[0].set_ylabel('Model', size=14)
axes[0].set_xlim(0, 1.0)
axes[0].set_xticks(np.arange(0, 1.1, 0.1))

performance.sort_values(by=['Accuracy(Test Set)'], ascending=False, inplace=True)
sns.barplot(x='Accuracy(Test Set)', y='Model', data=performance, palette='Reds_d', ax=axes[1])
axes[1].set_xlabel('Accuracy (Test Set)', size=14)
axes[1].set_ylabel('Model', size=14)
axes[1].set_xlim(0, 1.0)
axes[1].set_xticks(np.arange(0, 1.1, 0.1))

# plt.title("Accuracy Plot")
plt.tight_layout()


# In[389]:


# Sorted based on Accuracy(Test Set)
performance.sort_values(by=['Accuracy(Test Set)'], ascending=False, inplace=True)
performance


# In[390]:


# Sorted based on Accuracy(Training Set)
performance.sort_values(by=['Accuracy(Training Set)'], ascending=False, inplace=True)
performance


#  
