import pandas as pd
import numpy as np
import csv as csv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

df=pd.read_csv('/Users/prakashchandraprasad/Desktop/datasets/Titanic/train.csv',header=0)
df['AgeN']=df['Age']

df['AgeN']=df['Age'].fillna(df['Age'].mean()) #fill missing values with the mean age value
df['SexN']=df['Sex']
enc=LabelEncoder()
df['SexN']=enc.fit_transform(df['Sex']) #transform the sex-value into 0 and 1
#df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
X=df[['PassengerId','Pclass','SibSp','Parch','Fare','AgeN','SexN']] # select the required features
y=df['Survived']
X_train=X[:500]
X_test=X[500:]
y_train=y[:500]
y_test=y[500:]
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.48)
pipeline=Pipeline([('clf',DecisionTreeClassifier(criterion='entropy'))]) #select classifier as decision tree
parameters={'clf__max_depth':(100,105,110,115,120),
           'clf__min_samples_split':(3,4,5,6,7),
           'clf__min_samples_leaf':(3,4,5,6,7)}

grid_search=GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1,scoring='f1')
grid_search.fit(X_train,y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
predictions = grid_search.predict(X_test)
print classification_report(y_test, predictions)
finlist=zip(X_test['PassengerId'],predictions)
with open("/Users/prakashchandraprasad/Desktop/datasets/Titanic/Decision_tree_titanic.csv","wb") as f:
    writer=csv.writer(f)
    writer.writerow(["PassengerId","Survived"])
    writer.writerows(finlist)
