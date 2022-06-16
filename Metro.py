# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:14:43 2019

@author: Shashank Awasthi
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


file = r'D:\Datasets\Metro_Interstate_Traffic_Volume.csv'
df_traffic_data=pd.read_csv(file,parse_dates=True)
df_traffic_data.columns
# =============================================================================
# data.drop(['weather_description'],axis=1)
# 
# data.drop(columns =['weather_description'], inplace = True)
# data['holiday'].value_counts()
# data['weather_main'].value_counts()
# data['snow_1h'].value_counts()
# 
# enc = LabelEncoder()
# data.holiday = enc.fit_transform(data.holiday)
# data.weather_main = enc.fit_transform(data.weather_main)
# 
# data['date_time'] = pd.to_datetime(data.date_time)
# data.dtypes
# 
# =============================================================================

df_traffic_data.head()
df_traffic_data.shape
df_traffic_data.info()
Dat_describe= df_traffic_data.describe()
Dat_describe2= df_traffic_data.describe(include='object')


df_traffic_data['date_time'].min()
df_traffic_data['date_time'].max()

df_traffic_data.isnull().sum()
data =pd.Series(df_traffic_data['traffic_volume'])
data.plot()


plt.figure(figsize = (8,6))
sns.countplot(y='holiday', data = df_traffic_data)
plt.show()

holidays = df_traffic_data.loc[df_traffic_data.holiday != 'None']
plt.figure(figsize=(8,6))
sns.countplot(y='holiday', data= holidays)
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot('temp', data = df_traffic_data)
plt.show()




df_traffic_data['temp'] = (df_traffic_data['temp']-273.15)
plt.figure(figsize=(6,4))
sns.boxplot('temp', data = df_traffic_data)
plt.show()


plt.figure(figsize=(6,4))
sns.distplot(df_traffic_data.rain_1h)
plt.show()

plt.hist(df_traffic_data.rain_1h.loc[df_traffic_data.rain_1h<1])
plt.show()

plt.hist(df_traffic_data.snow_1h)
plt.show()


sns.distplot(df_traffic_data.clouds_all)
plt.show()

sns.countplot(y='weather_main', data=df_traffic_data)

plt.figure(figsize=(10,8))
sns.countplot(y='weather_description', data=df_traffic_data)
plt.show()

plt.figure(figsize=(10,8))
sns.boxplot(y='holiday',x='traffic_volume', data = holidays)
plt.show()

num_vars = ['temp','rain_1h','snow_1h','clouds_all','traffic_volume']
from pandas.plotting import scatter_matrix
scatter_matrix(df_traffic_data[num_vars],figsize=(10,8))
plt.show()

plt.figure(figsize=(10,8))
sns.set_style('darkgrid')
sns.jointplot(y='traffic_volume', x='temp', data = df_traffic_data.loc[df_traffic_data.temp>-50])
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(y='traffic_volume', x='temp', data = df_traffic_data.loc[df_traffic_data.temp>-50])

plt.figure(figsize=(14,8))
sns.barplot(x='clouds_all', y = 'traffic_volume', data = df_traffic_data)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x='weather_main', y = 'traffic_volume', data = df_traffic_data)
plt.show()


plt.figure(figsize=(12,8))
sns.barplot(y='weather_description', x = 'traffic_volume', data = df_traffic_data)
plt.show()

sns.heatmap(df_traffic_data.corr(), annot=True)
plt.show()

#######

df_traffic_features = df_traffic_data.copy()

df_traffic_features['date_time'] = pd.to_datetime(df_traffic_features.date_time)
df_traffic_features['weekday'] = df_traffic_features.date_time.dt.weekday
df_traffic_features['date'] = df_traffic_features.date_time.dt.date
df_traffic_features['hour'] = df_traffic_features.date_time.dt.hour
df_traffic_features['month'] = df_traffic_features.date_time.dt.month
df_traffic_features['year'] = df_traffic_features.date_time.dt.year

df_traffic_features.head()

def hour_modify(x):
    Early_Morning = [4,5,6,7]
    Morning = [8,9,10,11]
    Afternoon = [12,13,14,15]
    Evening = [16,17,18,19]
    Night = [20,21,22,23]
    Late_Night = [24,1,2,3]
    if x in Early_Morning:
        return 'Early_Morning'
    elif x in Morning:
        return 'Morning'
    elif x in Afternoon:
        return 'Afternoon'
    elif x in Evening:
        return 'Evening'
    elif x in Night:
        return 'Night'
    else:
        return 'Late_Night'
    
df_traffic_features['hour'] = df_traffic_features.hour.map(hour_modify)


plt.figure(figsize=(8,6))
sns.boxplot(x='weekday', y='traffic_volume', data = df_traffic_features)
plt.show()

df_date_traffic = df_traffic_features.groupby('year').aggregate({'traffic_volume':'mean'})
plt.figure(figsize=(8,6))
sns.lineplot(x = df_date_traffic.index, y = df_date_traffic.traffic_volume, data = df_date_traffic)
plt.show()


def modify_holiday(x):
    if x == 'None':
        return False
    else:
        return True
df_traffic_features['holiday'] = df_traffic_features['holiday'].map(modify_holiday)



df_traffic_features = df_traffic_features.loc[df_traffic_features.temp>-250]

plt.figure(figsize=(8,6))
sns.barplot(x='holiday', y='traffic_volume', data = df_traffic_features)
plt.show()

df_traffic_features.groupby('weather_description').aggregate({'traffic_volume':[np.mean,np.size],
                                                              'clouds_all':'count','rain_1h':'mean','snow_1h':'mean'})
	
df_traffic_features['weather_description'] = df_traffic_features['weather_description'].map(lambda x:x.lower())

df_traffic_features.loc[df_traffic_features['weather_description'].str.contains('thunderstorm'),'weather_description'] = 'thunderstorm' 

weather = ['thunderstorm','mist','fog','haze']
df_traffic_features.loc[np.logical_not(df_traffic_features['weather_description'].isin(weather)),'weather_description'] = 'other'


df_traffic_features.weather_description.value_counts()

df_traffic_features = pd.get_dummies(columns=['weather_description'],data=df_traffic_features)

df_traffic_features.rename(columns={'weather_description_fog':'fog', 'weather_description_haze':'haze',
                                   'weather_description_mist':'mist', 'weather_description_thunderstorm':'thunderstorm'}, inplace = True)
df_traffic_features.drop(columns = ['weather_description_other', 'weather_main'], inplace = True)


df_traffic_features.columns

plt.figure(figsize=(8,6))
sns.boxplot('rain_1h',data = df_traffic_features)
plt.show()

sns.boxplot('rain_1h',data = df_traffic_features.loc[df_traffic_features.rain_1h<2000])


df_traffic_features = df_traffic_features.loc[df_traffic_features.rain_1h<2000]
df_traffic_features_temp = df_traffic_features.loc[df_traffic_features.rain_1h>0]
rain_q = pd.DataFrame(pd.qcut(df_traffic_features_temp['rain_1h'] ,q=3, labels=['light','moderate','heavy']))
df_traffic_cat = df_traffic_features.merge(rain_q,left_index=True, right_index=True, how='left')
df_traffic_cat['rain_1h_y'] = df_traffic_cat.rain_1h_y.cat.add_categories('no_rain')
df_traffic_cat['rain_1h_y'].fillna('no_rain', inplace = True) #no_rain is not in the category, adding it and filling

df_traffic_cat.drop(columns=['rain_1h_x'], inplace = True)
df_traffic_cat.rename(columns={'rain_1h_y':'rain_1h'}, inplace = True)
df_traffic_cat.head()

sns.boxplot('snow_1h',data = df_traffic_features)

df_traffic_features.snow_1h[df_traffic_features.snow_1h>0].count()

def modify_snow1h(x):
    if x==0:
        return 'no_snow'
    else:
        return 'snow'
    
df_date_traffic['snow_1h'] = df_traffic_cat.snow_1h.map(modify_snow1h)

df_traffic_features.head()


df_traffic_cat.set_index('date', inplace = True)

df_traffic_cat.columns

target = ['traffic_volume']
cat_vars = ['holiday', 'snow_1h','weekday', 'hour', 'month', 'year', 'fog', 'haze','mist', 'thunderstorm', 'rain_1h']
num_vars = ['temp','clouds_all']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('oneHot',OneHotEncoder())])

preprocessor = ColumnTransformer(transformers=[
    ('num',numeric_transformer,num_vars),
    ('cat',categorical_transformer,cat_vars)])

df_traffic_transformed = preprocessor.fit_transform(df_traffic_cat).toarray()



X_train = df_traffic_transformed[:32290]
X_test = df_traffic_transformed[32291:]
y_train = df_traffic_cat.traffic_volume[:32290]
y_test = df_traffic_cat.traffic_volume[32291:]

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

tscv = TimeSeriesSplit(n_splits=3)
model = xgb.XGBRegressor()

param_grid = {'nthread':[4,6,8], 
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07],
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

GridSearch = GridSearchCV(estimator = model,param_grid= param_grid,cv=tscv, n_jobs = -1 )
GridSearch.fit(X_train, y_train)

print('Best parameters found:\n', GridSearch.best_params_)
print('Best parameters found:\n', GridSearch.best_score_)

y_pred = GridSearch.predict(X_test)


from sklearn.externals import joblib

joblib.dump(GridSearch,'C:/Users/Shashank Awasthi/Desktop/Used_car_deploy/GridSearch.ml')

r2 = r2_score(y_test,y_pred)
print(r2)
RMSE = np.sqrt(mean_squared_error(y_pred=y_pred,y_true=y_test))
print(RMSE)

predicted_traffic = pd.DataFrame(y_pred,index=y_test.index,columns = ['traffic_volume'])  

predicted_traffic.plot(figsize=(10,5))  

y_test.plot()  

plt.legend(['predicted_traffic','actual_traffic'])  

plt.ylabel("Metro Traffic Predict")  
plt.show()

sns.lineplot(data=predicted_traffic)
sns.lineplot(data=y_test)
plt.show()

