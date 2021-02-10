#!/usr/bin/env python
# coding: utf-8

# # Abhishek Kumar

# In[ ]:


# Question: Predict the percentage of an student based on the no. of study hours.


# In[14]:


#importing library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[15]:


#importing dataset 

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
dataset = pd.read_csv(url)

dataset.head(10)


# In[25]:


# Taking care of missing data 

x = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values


# In[ ]:


from sklearn.impute import SimpleImputer

Imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
Imputer.fit(x[:,:-1])
x[:,:-1] = Imputer.transform(x[:,:-1])

print(x)


# In[17]:


# Splitting the data for trainning and test set 

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[18]:


# Trainning simple linear resgression model 

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()
regressor.fit(x_train,y_train)

print("Trainning Done")


# In[19]:


# Predecting the test results 

y_pred = regressor.predict(x_test)

print(y_pred)


# In[38]:


# Comparing Actual vs Predicted Data 

avsp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
avsp


# In[20]:


# Visualising the results 

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,(regressor.predict(x_train)),color = 'blue')

plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  

plt.show()


# In[41]:


# Testing with data 

hours = [[7.6]]
result = regressor.predict(hours)

print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(result[0]))


# In[40]:


# Evaluating the model 

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




