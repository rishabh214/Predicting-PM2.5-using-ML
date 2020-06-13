#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pickle
from statistics import mean, median


df=pd.read_csv('data.csv')

df.drop([0,1,2],inplace=True)
df.drop('so2',axis=1,inplace=True)
df.drop(['start','end'],axis=1,inplace=True)
df.drop(['o3', 'co','nox'],axis=1,inplace=True)

# In[31]:

df.dropna(subset=['pm2.5', 'pm10'],inplace=True)
df.dropna(subset=['temp','bp','rh','ws'],inplace=True)

Y=df['pm2.5']
X=df[['pm10', 'temp','bp', 'rh', 'ws']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,Y_train,Y_test = train_test_split( X,Y, test_size=0.3, random_state=34)

lm=LinearRegression()

lm.fit(X_train,Y_train)

# Saving model to disk
pickle.dump(lm, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[300,30,2, 9, 6]]))
#predictions=model.predict(X_test)

#
#plt.figure(figsize=(16,9))
#plt.scatter(Y_test,predictions)
#sns.regplot(Y_test,predictions,scatter=False,color='red')
#plt.xlabel('Y_test(trueValues)')
#plt.ylabel('predicted values')
#
#from sklearn import metrics
#
#
## In[50]:
#
#
#print('MAE',metrics.mean_absolute_error(Y_test,predictions))
#print('MSE',metrics.mean_squared_error(Y_test,predictions))
#print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
#

# In[53]:


print(model.predict([[169.0,31.0,741.0,84.0,1.0]]))

# In[


# In[ ]:




