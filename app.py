import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
n_samples = 1000

df = pd.DataFrame({
    'age': np.random.randint(18,65, size=n_samples),
    'sex': np.random.choice(['male','female'], size=n_samples),
    'bmi': np.random.uniform(15,40, size=n_samples),
    'children': np.random.randint(0,5, size=n_samples),
    'smoker': np.random.choice(['yes','no'], size=n_samples),
    'region': np.random.choice(['northeast','nothwest','southeast','southwest'], size=n_samples),
    'charges': np.random.uniform(2000,50000, size=n_samples),
})

print(df.shape)
print(df.head())
print(df.info())

categorical = [var for var in df.columns if df[var].dtype == 'O']
print('Non-numeric value are:',categorical)

numerical = [col for col in df.columns if df[col].dtype != 'O']
print('Numeric value are:',numerical)

print(df.isnull().sum())

corr = df[numerical].corr()
sns.heatmap(corr, cmap='Wistia', annot=True, cbar=False)
plt.show()

f2 = plt.figure(figsize=(14,5))
ax = f2.add_subplot(121)
sns.distplot(df['charges'], bins=30, color='r', ax=ax)
ax=f2.add_subplot(122)
sns.distplot(df['charges'], bins=30, color='b', ax=ax)
plt.show()

categorical_columns = ['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(data=df, columns=categorical_columns)
print(df_encoded.head())

df_encoded['charges'] = np.log(df_encoded['charges'])
x=df_encoded.drop('charges', axis=1)
y=df_encoded['charges']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=23)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

linreg = LinearRegression()

linreg.fit(x_train,y_train)
y_pred=linreg.predict(x_test)
msr = mean_absolute_error(y_test,y_pred=y_pred)
print("Mean square is:", msr)

sns.scatterplot(x=y_test,y=y_pred)
plt.show()

