import os
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
print("dataset before!\n",dataset.head())
print(dataset.shape)

#print(dataset['gender'].unique())
#print(dataset.isna().any()) # is there a nan value  or not

# Replace Nan Value in "BMI" with The Mean
dataset['bmi'].fillna(value=dataset['bmi'].mean() , inplace=True)

dataset=dataset.replace({
    'gender':{'Other' : 'Male'},
})

dataset=dataset.replace({
    'gender':{'Male' : 1 , 'Female' : 0},
    'ever_married' :{'Yes':1, 'No':0},
    'Residence_type':{'Urban':1, 'Rural':0 }
})


dataset=pd.concat([dataset,pd.get_dummies(dataset['work_type'],prefix='WT', drop_first=True)],axis=1)
dataset.drop(['work_type'],axis=1, inplace = True)

dataset=pd.concat([dataset,pd.get_dummies(dataset['smoking_status'], drop_first=True)],axis=1)
dataset.drop(['smoking_status'],axis=1, inplace = True)

# print(dataset.head())
#print(dataset['avg_glucose_level'].skew())
#print(dataset['avg_glucose_level'].describe())


j = dataset['avg_glucose_level'].quantile(0.20)
k = dataset['avg_glucose_level'].quantile(0.80)


# print(dataset['avg_glucose_level'].skew())
Q1 = dataset['avg_glucose_level'].quantile(0.25)
Q3 = dataset['avg_glucose_level'].quantile(0.75)
IQR = Q3-Q1
MIN = Q1 - 1.5*IQR
MAX = Q3 + 1.5*IQR

# print(MAX)
# print(MIN)
dataset['avg_glucose_level'] = np.where(dataset['avg_glucose_level'] < MIN, j , dataset['avg_glucose_level'])
dataset['avg_glucose_level'] = np.where(dataset['avg_glucose_level'] > MAX, k , dataset['avg_glucose_level'])
# print(dataset['avg_glucose_level'].skew())


# print(plt.boxplot(dataset['avg_glucose_level']))

n = dataset['bmi'].quantile(0.20)
m = dataset['bmi'].quantile(0.80)

# print(dataset['bmi'].describe())

# print(dataset['bmi'].skew())
Q1_2 = dataset['bmi'].quantile(0.25)
Q3_2 = dataset['bmi'].quantile(0.75)
IQR_2 = Q3_2-Q1_2
MIN2 = Q1_2 - 1.5*IQR_2
MAX2 = Q3_2 + 1.5*IQR_2

# print(MAX2)
# print(MIN2)
dataset['bmi'] = np.where(dataset['bmi'] < MIN2, n , dataset['bmi'])
dataset['bmi'] = np.where(dataset['bmi'] > MAX2, m , dataset['bmi'])

# print(dataset['bmi'].skew())
# print(pit.boxplot(dataset['bmi']))
# print(dataset.corr())

# fig, (ax1) = plt.subplots(ncols=1, figsize=(10,8))
# ax1.set_title('Original Distributation')

# sns.kdeplot(dataset['age'],ax=ax1)


# Normaliazation
col_names = list(dataset.columns)
mm_scaler = preprocessing.MinMaxScaler()
dataset_mm = mm_scaler.fit_transform(dataset)
dataset_mm = pd.DataFrame(dataset_mm, columns=col_names)


dataset = dataset_mm
print(dataset.head())


#fig, (ax1) = plt.subplots(ncols=1, figsize=(10,8))
#ax1.set_title('Original Distributation')
#sns.kdeplot(dataset['age'],ax=ax1)


#dataset.shape

y = dataset.stroke
x = dataset.drop(['stroke'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

print(x_train.head())
print(x_train.shape)

print(x_test.head())
print(x_test.shape)


#SVC Model
lsvc = LinearSVC(verbose=0)
print(lsvc)
lsvc.fit(x_train, y_train)
score = lsvc.score(x_train, y_train)
print("svc Accuracy: ", score*100)

#Logistic Regression Model
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
Ypred = logistic_regression.predict(x_test)
testAccuracy = accuracy_score(y_test, Ypred)
print("Logistic Regression Accuracy: ", testAccuracy*100)