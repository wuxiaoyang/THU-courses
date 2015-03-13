import numpy as np
from sklearn import linear_model
from sklearn import datasets

iris=datasets.load_iris()
X,y=iris.data,iris.target

clf=linear_model.LogisticRegression()

ret=clf.fit(X,y)
yy=clf.predict(X)


