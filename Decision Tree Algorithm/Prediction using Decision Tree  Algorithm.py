#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv("C:/Users/Nihan Sayyed/Downloads/iris.csv")
df.head()


# In[6]:


df.info()


# In[8]:


df.describe()


# In[9]:


corr=df.corr()


# In[10]:


sns.heatmap(corr,annot=True)


# In[11]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
species=df.iloc[:,-1]
le.fit(species)
df.iloc[:,-1]=le.transform(species)
df


# In[12]:


x=df.iloc[:,0:-1]
x.head()


# In[14]:


y=df['species']
y.head()


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[16]:


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
y_predict=DT.predict(x_test)
y_predict


# In[17]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy:',accuracy_score(y_predict,y_test))


# In[18]:


confusion_matrix(y_predict,y_test)


# In[19]:


classification_report(y_predict,y_test)


# In[20]:


from sklearn import tree
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DT)


# In[21]:


fn=['sepal_length (cm)','sepal_width (cm)','petal_length (cm)','petal_width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
tree.plot_tree(DT,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[ ]:




