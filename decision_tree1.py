# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:15:07 2019

@author: SURAJ BHADHORIYA
"""

#load libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.cross_validation import train_test_split
import pandas as pd
import pydotplus
from IPython.display import Image


#read data
df=pd.read_csv("C:/Users/SURAJ BHADHORIYA/Desktop/loan_dataset.csv")

#print the names of col.
col=df.columns
print(col)
#make label
df['safe_loans']=df['bad_loans'].apply(lambda s:+1 if s==0 else 0)
print(df['safe_loans'])

#find the +ive & -ive % of loan
pos_loan=len(df[df['safe_loans']==1])
neg_loan=len(df[df['safe_loans']==-0])
pos=(pos_loan*100)/122607
neg=(neg_loan*100)/122607
print("positive loan %",pos)
print("negative loan %",neg)

#put all feature together
feature=['grade','term','home_ownership','emp_length']
label=['safe_loans']

#make new dataframe where only feature and label append
loan=df[feature+label]


#make one hot encoding on dataframe
loan1=pd.get_dummies(loan)

#make feature one hot encoading
x=pd.get_dummies(loan[feature])
#make label
y=loan['safe_loans']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#apply DecisionTreeClassifier model with prunning or to overcome overfitting
clf=DecisionTreeClassifier(criterion="gini",max_leaf_nodes=100,
                           min_samples_leaf=10000, max_depth=10)
clf.fit(X_train,y_train)
#accuracy
accuracy=clf.score(X_test,y_test)
print("accuracy =",accuracy)

acc=clf.score(X_train,y_train)
print("accuracy =",acc)

feature1=x.columns
label1=['faulty','not_faulty']



# Create decision tree
dot_data = export_graphviz(clf, out_file=None, 
                                feature_names=feature1,
                                class_names=label1,
                                filled=True,
                                rounded=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())


# Create PDF
graph.write_pdf("Decisi.pdf")
# Create PNG
graph.write_png("Decisi.png")















