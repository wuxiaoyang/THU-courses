import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

class myLogisticRegression:

    def __init__(self):

        self.alpha=0.0001
        self.theta=np.zeros(1)

    @staticmethod
    def h(theta,x):
        return 1.0/( 1 + np.e**(- np.dot( theta, x) ) )

    def fit(self,X,y):

        N,D=X.shape
        self.theta=np.zeros(D)

        for it in range(1000):
            for i in range(N):
                self.theta += self.alpha *  ( y[i] - self.h( self.theta, X[i]) )  * X[i] 

    def evaluate(self,ret,y):

        tp=sum(np.logical_and(y==1,ret==1))
        tn=sum(np.logical_and(y==0,ret==0))
        fp=sum(np.logical_and(y==0,ret==1))
        fn=sum(np.logical_and(y==1,ret==0))
        
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1=2*precision*recall/(precision+recall)

        print 'Evaluation'.center(20,'=')
        print 'Precision:', precision
        print 'Recall:', recall
        print 'F1 Score:', F1
        print ''.center(20,'=')

    def predict(self,X):
        N,D=X.shape
        y=np.zeros(N)

        for i in range(N):
            y[i]=self.h(self.theta , X[i])

        y=np.around(y)
        return y

    def cross_validate(self, K):
        pass

def main():

    iris = load_iris()
    X,y = iris.data[:100],iris.target[:100]

    my_clf = myLogisticRegression()
    std_clf = LogisticRegression()

    std_clf.fit(X,y)
    my_clf.fit(X,y)
    ret = my_clf.predict(X)
    print my_clf.evaluate(ret,y)

if __name__ == '__main__':

    main()
