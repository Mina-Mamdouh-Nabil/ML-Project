import os
import numpy as np 
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
import warnings

dataset = pd.read_csv('D:\Projects\ML\Data Sets\healthcare-dataset-stroke-data\healthcare-dataset-stroke-data.csv')

dataset_orig = dataset.copy()


#print(dataset.dtypes)
#print(dataset.info()) 
#print(dataset.isna().any()) # is there a nan value  or not
#print(dataset.columns)
#print(dataset.shape)

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


#print(dataset.isna().any()) # is there a nan value  or not


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


fig, (ax1) = plt.subplots(ncols=1, figsize=(10,8))
ax1.set_title('Original Distributation')

sns.kdeplot(dataset['age'],ax=ax1)






