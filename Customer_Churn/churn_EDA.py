import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

telco_base_data = pd.read_csv('/content/Telco_Customer_Chrun.csv')
telco_base_data.head()

print('SHAPE OF DF')
print(telco_base_data.shape)
print('----' * 20)
print('\nCOLUMN NAMES')
print(telco_base_data.columns.values)
print('----' * 20)
print('\nDATA TYPES')
print(telco_base_data.dtypes)
print('----' * 20)
print('\nDESCRIBED')
print(telco_base_data.describe())
print('----' * 20)

telco_base_data['Churn'].value_counts().plot(kind = 'barh', figsize = (8,6))
plt.xlabel('Target Variable', labelpad=10)
plt.ylabel('Count', labelpad=10)
plt.title('Count of Target Variable per Category')

100 * telco_base_data['Churn'].value_counts() / len(telco_base_data['Churn'])

telco_base_data['Churn'].value_counts()

telco_base_data.info(verbose=True)

missing = pd.DataFrame((telco_base_data.isnull().sum())*100/telco_base_data.shape[0]).reset_index()
plt.figure(figsize=(12,5))
ax = sns.pointplot(x = 'index',y = 0,data = missing)
plt.xticks(rotation = 90, fontsize = 7 )
plt.title("Percentage of Missing Values")
plt.ylabel('%')
plt.show()

telco_data  = telco_base_data.copy() 

telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors= 'coerce')

telco_data.isnull().sum()

telco_data.loc[telco_data['TotalCharges'].isnull() == True]

telco_data.dropna(how = 'any', inplace=True)
telco_data.shape

telco_data['tenure'].max()

labels = [f"{i} - {i + 11}" for i in range(1,72,12)]
telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1,80, 12), right = False, labels = labels)
telco_data['tenure_group']

telco_data.drop(columns = ['customerID', 'tenure'], axis = 1, inplace = True)
telco_data.head()

for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
  plt.figure(i)
  sns.countplot(telco_data, x=predictor, hue = 'Churn')

telco_data['Churn'] = np.where(telco_data['Churn'] == 'Yes', 1, 0)
telco_data.head()

telco_data_dummies = pd.get_dummies(telco_data)
telco_data_dummies.head()

sns.lmplot(telco_data_dummies, x = 'MonthlyCharges', y = 'TotalCharges', fit_reg = False)

mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[telco_data_dummies['Churn']==0], color='red', fill = True)
mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[telco_data_dummies['Churn']==1], color='blue', fill = True, ax = mth)
mth.legend(['No Chrun','Chrun'], loc = 'upper right')
mth.set_title('Churn by Monthly Charges')
mth.set_ylabel('Density')
mth.set_xlabel('Monthly Charges')

tol = sns.kdeplot(telco_data_dummies.TotalCharges[telco_data_dummies['Churn']==0], color='red', fill = True)
tol = sns.kdeplot(telco_data_dummies.TotalCharges[telco_data_dummies['Churn']==1], color='blue', fill = True, ax = tol)
tol.legend(['No Chrun','Chrun'], loc = 'upper right')
tol.set_title('Churn by Total Charges')
tol.set_ylabel('Density')
tol.set_xlabel('Total Charges')

plt.figure(figsize = (18,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind = 'bar')

plt.figure(figsize = (12,12))
sns.heatmap(telco_data_dummies.corr(), cmap = 'Paired')

df_target0 = telco_data.loc[telco_data["Churn"] == 0]
df_target1 = telco_data.loc[telco_data["Churn"] == 1]

def uniplot(df, col, title, hue = None):
  sns.set_style('whitegrid')
  sns.set_context('talk')
  plt.rcParams['axes.labelsize'] = 20 
  plt.rcParams['axes.titlesize'] = 22
  plt.rcParams['axes.titlepad'] = 30

  temp = pd.Series(data = hue)
  fig, ax = plt.subplots()
  width = len(df[col].unique()) + 7 + 4 * len(temp.unique())
  fig.set_size_inches(width, 8)
  plt.xticks(rotation = 45)
  plt.yscale('log')
  plt.title(title)
  ax = sns.countplot(data = df, x = col, order = df[col].value_counts().index, hue = hue, palette = 'bright')
  plt.show()


uniplot(df_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')

uniplot(df_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')

uniplot(df_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')

uniplot(df_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')

uniplot(df_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')

uniplot(df_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')

