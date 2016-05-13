import pandas as pd
import numpy as np
import csv as csv
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

df=pd.read_csv('/Users/prakashchandraprasad/Desktop/datasets/Titanic/train.csv',header=0)
df1=pd.read_csv('/Users/prakashchandraprasad/Desktop/datasets/Titanic/test.csv',header=0)
df['AgeN']=df['Age']
df1['AgeN']=df1['Age']


df['AgeN']=df['Age'].fillna(df['Age'].mean())
df1['AgeN']=df1['Age'].fillna(df1['Age'].mean())

df1['Fare']=df['Fare'].fillna(0)
df['SexN']=df['Sex']
df1['SexN']=df1['Sex']


enc=LabelEncoder()



df['SexN']=enc.fit_transform(df['Sex']) 
df1['SexN']=enc.fit_transform(df1['Sex'])



X_train=df[['Pclass','SibSp','Parch','Fare','AgeN','SexN']] 
y_train=df['Survived']
X_test=df1[['Pclass','SibSp','Parch','Fare','AgeN','SexN']]
X_test1=df1[['PassengerId','Pclass','SibSp','Parch','Fare','AgeN','SexN']]
svc=SVC(kernel='linear')
#svc=DecisionTreeClassifier(criterion='entropy')
rfecv=RFECV(estimator=svc, step=1, cv=StratifiedKFold(y_train, 5),scoring='accuracy')
rfecv.fit(X_train,y_train)
predictions=rfecv.predict(X_test)
print rfecv.score(X_train,y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

finlist=zip(X_test1['PassengerId'],predictions)
with open("/Users/prakashchandraprasad/Desktop/datasets/Titanic/Decision_tree_titanic7.csv","wb") as f:
    writer=csv.writer(f)
    writer.writerow(["PassengerId","Survived"])
    writer.writerows(finlist)
