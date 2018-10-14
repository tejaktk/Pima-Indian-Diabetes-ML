# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:09:24 2018

@author: guna_
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("pima.csv")

x= df.iloc[:,0:8].values
y= df.iloc[:,8].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
classifier1= DecisionTreeClassifier(criterion="gini",random_state=123)
classifier1.fit(x_train,y_train)

y_pred1 = classifier2.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm1 = confusion_matrix(y_test,y_pred1)
acc1 = accuracy_score(y_test,y_pred1)

from sklearn.tree import DecisionTreeClassifier
classifier2= DecisionTreeClassifier(criterion="entropy",random_state=123)
classifier2.fit(x_train,y_train)

y_pred2 = classifier2.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm2 = confusion_matrix(y_test,y_pred2)
acc2 = accuracy_score(y_test,y_pred2)

from sklearn import tree
from sklearn.tree import export_graphviz
tree.export_graphviz(classifier1, out_file='tree.dot') 

from sklearn.svm import SVC
svclassifier1= SVC(kernel='rbf')
svclassifier1.fit(x_train,y_train)

y_pred3 = svclassifier1.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
SVCcm1 = confusion_matrix(y_test,y_pred3)
SVCacc1 = accuracy_score(y_test,y_pred3)

from sklearn.svm import SVC
svclassifier1= SVC(kernel='rbf')
svclassifier1.fit(x_train,y_train)

y_pred3 = svclassifier1.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
SVCcm1 = confusion_matrix(y_test,y_pred3)
SVCacc1 = accuracy_score(y_test,y_pred3)

from sklearn.svm import SVC
svclassifier2= SVC(kernel='linear')
svclassifier2.fit(x_train,y_train)

y_pred4 = svclassifier2.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
SVCcm2 = confusion_matrix(y_test,y_pred4)
SVCacc2 = accuracy_score(y_test,y_pred4)

from sklearn.svm import SVC
svclassifier3= SVC(kernel='poly')
svclassifier3.fit(x_train,y_train)

y_pred5 = svclassifier3.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
SVCcm3 = confusion_matrix(y_test,y_pred5)
SVCacc3 = accuracy_score(y_test,y_pred5)

from sklearn.svm import SVC
svclassifier4= SVC(kernel='sigmoid')
svclassifier4.fit(x_train,y_train)

y_pred6 = svclassifier4.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
SVCcm4 = confusion_matrix(y_test,y_pred6)
SVCacc4 = accuracy_score(y_test,y_pred6)

from sklearn import datasets
mydata=datasets.load_diabetes()


X=mydata.data
Y=mydata.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
reg1 = DecisionTreeRegressor(criterion = "mse")
reg1.fit(X_train,Y_train)

Y_pred1 = reg1.predict(X_test)

from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
mae1 = mean_absolute_error(Y_test,Y_pred1)
mse1 = mean_squared_error(Y_test,Y_pred1)
Rsquared1 = r2_score(Y_test,Y_pred1)

from sklearn.tree import DecisionTreeRegressor
reg2 = DecisionTreeRegressor(criterion = "friedman_mse")
reg2.fit(X_train,Y_train)

Y_pred2 = reg2.predict(X_test)

from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
mae2 = mean_absolute_error(Y_test,Y_pred2)
mse2 = mean_squared_error(Y_test,Y_pred2)
Rsquared2 = r2_score(Y_test,Y_pred2)


from sklearn.tree import DecisionTreeRegressor
reg3 = DecisionTreeRegressor(criterion = "mae")
reg3.fit(X_train,Y_train)

Y_pred3 = reg3.predict(X_test)

from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
mae3 = mean_absolute_error(Y_test,Y_pred3)
mse3 = mean_squared_error(Y_test,Y_pred3)
Rsquared3 = r2_score(Y_test,Y_pred3)




