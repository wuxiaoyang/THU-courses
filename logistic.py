import numpy as np
from sklearn import cross_validation
import cPickle as pickle

class myLogisticRegression:

    def __init__(self, learning_rate=0.01):

        self.alpha=learning_rate
        self.theta=np.zeros(1)

    @staticmethod
    def h(theta,x):
        return 1.0/( 1 + np.e**(- np.dot( theta, x) ) )

    def fit(self,X,y):

        N,D=X.shape
        self.theta=np.zeros(D)

        for it in range(100):
            for i in range(N):
                self.theta += self.alpha *  ( y[i] - self.h( self.theta, X[i]) )  * X[i] 


    def evaluate(self,ret,y):

        tp = sum(np.logical_and(y==1,ret==1))
        tn = sum(np.logical_and(y==0,ret==0))
        fp = sum(np.logical_and(y==0,ret==1))
        fn = sum(np.logical_and(y==1,ret==0))
        tp,tn,fp,fn = map(float, [tp,tn,fp,fn])

        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1=2*precision*recall/(precision+recall)

        print 'Evaluation'.center(20,'=')
        print 'Precision:', precision
        print 'Recall:', recall
        print 'F1 Score:', F1
        print 'End'.center(20,'=')
        print

    def predict(self,X):

        N,D=X.shape
        y=np.zeros(N)

        for i in range(N):
            y[i]=self.h(self.theta , X[i])

        y=np.around(y)
        return y

    def cross_validate(self, X, y, K):
        N,D=X.shape
        rs = cross_validation.ShuffleSplit(N, n_iter=K, test_size=1.0-1.0/K, random_state=0)

        for train_index, test_index in rs:

            self.fit( X[train_index, :], y[ train_index ] )
            ret=self.predict(X[test_index , :])
            self.evaluate(ret,y[test_index])

def main():

    X,y = pickle.load(open('TF-IDF.dat'))
    N,D = X.shape

    my_clf = myLogisticRegression()
    rs = cross_validation.ShuffleSplit(N, n_iter=5, test_size=0.2, random_state=0)
    my_clf.cross_validate(X,y,5)

if __name__ == '__main__':

    main()
