#!/usr/bin/env python
# coding: utf-8

# scikit learn is very popular machine learning
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[2]:


data= pd.read_csv("pima.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


# the first 8 columns represent features and last 1 represent target


# In[6]:


# lets create numpy as arrays  for features  and target

X = data.drop('Outcome', axis = 1).values
y = data['Outcome'].values


# In[7]:


#let ssplit the data randonly into training and set test, we will fit / train a classifer on the training 
# and make predictions on test data


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


# it is best practise to perform our split in a such a way that our split reflects the labels in data,
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 42,stratify = y)


# In[10]:


# lets create a classfier using knnn classfier


# In[11]:


from sklearn.neighbors  import KNeighborsClassifier


# In[12]:


# setup arrays to store training and test accuracies

neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    


# In[13]:


train_accuracy[i] = knn.score(X_train,y_train)
test_accuracy[i] = knn.score(X_test,y_test)


# In[14]:


plt.title('knn carying number of neighbors')
plt.plot(neighbors,test_accuracy,label = 'testing accuarcy')
plt.plot(neighbors,train_accuracy, label = 'training accuracy')
plt.legend()
plt.xlabel('nuumber of neighbors')
plt.ylabel('accuarcy')
plt.show()


# In[15]:


#  we can observe above that we get maximum testing accuacy  for k = 7, so lets create a knn with k= 7


# In[16]:


knn = KNeighborsClassifier(n_neighbors = 7)


# In[17]:


knn.fit(X_train, y_train)


# In[18]:


knn.score(X_test, y_test)


# In[19]:


# confusion matrix :
#table which is used for performance check for classifiers


# In[20]:


from sklearn.metrics import confusion_matrix


# In[21]:


y_pred = knn.predict(X_test)


# In[22]:


confusion_matrix(y_test,y_pred)


# In[23]:


# TN = 165, FP = 36,TP = 60,FN = 47


# In[24]:


from sklearn.metrics import classification_report


# In[25]:


print(classification_report(y_test,y_pred))


# In[ ]:


# roc curve : 


# In[ ]:




