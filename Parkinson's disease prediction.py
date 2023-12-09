#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from sklearn.svm import SVC


# In[3]:


import os


# In[4]:


pwd


# In[5]:


cd\parkinson's


# In[6]:


parkinson_data=pd.read_csv("parkinsons_new.csv")
parkinson_data.head(
)


# In[7]:


parkinson_data.shape


# In[8]:


parkinson_data.isnull().sum()


# In[9]:


#parkinson_data=parkinson_data.dropna()


# In[10]:


parkinson_data.info()


# In[11]:


parkinson_data.columns


# In[12]:


#parkinson_data['sex'][parkinson_data['sex'] == 0] = 'F'
#parkinson_data['sex'][parkinson_data['sex'] == 1] = 'M'


# In[13]:


d1=pd.get_dummies(parkinson_data.sex)
parkinson_data.head()


# In[14]:


#parkinson_data.drop(['sex'],axis=1,inplace=True )
parkinson_data.drop(['sex','name'], axis=1,inplace=True)
parkinson_data= pd.concat([parkinson_data,d1],axis=1)
parkinson_data.head()


# In[15]:


prkn_dt=parkinson_data
plt.scatter(prkn_dt['age'],prkn_dt['status'])
plt.xlabel("age")

plt.ylabel("status")
plt.show()


# In[175]:


from sklearn.model_selection import train_test_split


# # Decision Tree

# In[176]:


x=parkinson_data.drop(["status"], axis=1)
y=parkinson_data.status


# In[177]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=5)


# In[178]:


from sklearn import tree
md=tree.DecisionTreeClassifier()
md.fit(X_train,y_train)
md.score(X_test,y_test)


# # Random Forest

# In[179]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier (n_estimators=100)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))
predictions=rf.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# # Logestic Regression

# In[180]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)


# In[181]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# # KNN

# In[182]:


from sklearn.neighbors import KNeighborsClassifier


# In[183]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[184]:


print("KNN with k=5 got {}% accuracy on the test set.".format(accuracy_score(y_test, knn.predict(X_test))*100))


# # SVM

# In[185]:


# import support vector classifier 
from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
  

clf.fit(X_train, y_train) 


# In[186]:


clf.predict(X_train)


# In[187]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[188]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # Perceptron

# In[189]:


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)


# In[190]:


class Perceptron(object):

        def __init__(self, eta, n_iter):
                self.eta = eta
                self.n_iter = n_iter

        def fit(self, X, y):

                self.w_ = np.zeros(1 + X.shape[1])
                self.errors_=[]
                for _ in range(self.n_iter):
                        errors=0
                        for xi, target in zip(X, y):
                                error = target - self.predict(xi)
                                if error != 0:
                                        update = self.eta * error
                                        self.w_[1:] += update * xi
                                        self.w_[0] += update
                                        errors+=int(update!=0.0)
                        self.errors_.append(errors)
                return self

        def net_input(self, X):
                return np.dot(X, self.w_[1:]) + self.w_[0]

        def predict(self, X):
                return np.where(self.net_input(X) >= 0.0, 0, 1)

import matplotlib.pyplot as plt
#from sklearn.linear_model import Perceptron

model = Perceptron(n_iter=1000,eta=0.01)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('misclassified samples: %d'%(y_test!=y_pred).sum())#compute
from sklearn.metrics import accuracy_score
print('Accuracy:%.2f'%accuracy_score(y_test,y_pred))
plt.show()


# # Linear regression

# In[191]:


from sklearn import datasets, linear_model, metrics
import numpy as np
from sklearn.linear_model import LinearRegression


# In[192]:


# create linear regression object
reg = linear_model.LinearRegression()
 
# train the model using the training sets
reg.fit(X_train, y_train)
 
# regression coefficients
print('Coefficients: ', reg.coef_)
 
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))
 
# plot for residual error
 
# setting plot style
plt.style.use('fivethirtyeight')
 
# plotting residual errors in training data
plt.scatter(reg.predict(X_train),
            reg.predict(X_train) - y_train,
            color="green", s=10,
            label='Train data')
 
# plotting residual errors in test data
plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10,
            label='Test data')
 
# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
 
# plotting legend
plt.legend(loc='upper right')
 
# plot title
plt.title("Residual errors")
 
# method call for showing the plot
plt.show()


# In[193]:


#model = LinearRegression()
lm = LinearRegression()
model = lm.fit(X_train, y_train)


# In[194]:


#model = LinearRegression().fit(X_train, y_train)
print(model.coef_, model.intercept_)


# In[195]:


#r_sq = model.score(X_train, y_train)
#print(f"coefficient of determination: {r_sq}")
print(model.score(X_test,y_test))


# In[196]:


#y_pred = model.predict(X_test)
#print(f"predicted response:\n{y_pred}")


# In[ ]:





# In[ ]:




