import numpy as np
from sklearn.model_selection import KFold

class TrAdaboost():
    def __init__(self,base_regressor,S=10,F=10):
        self.base_regressor=base_regressor
        self.S=S
        self.F=F
        self.base_regressors=[]

    def fit(self,x_source,x_target,y_source,y_target):
        x_train=np.concatenate((x_source,x_target),axis=0)
        y_train=np.concatenate((y_source,y_target),axis=0)
        x_train=np.asarray(x_train,order='C')
        y_train=np.asarray(y_train,order='C')
        y_source=np.asarray(y_source,order='C')
        y_target=np.asarray(y_target,order='C')

        n=x_source.shape[0]
        m=x_target.shape[0]

        weight_source=np.ones([n,1])
        weight_target=np.ones([m,1])
        weights=np.concatenate((weight_source,weight_target),axis=0)
        result=np.ones([n+m,self.S])
        errors=np.zeros([n+m,self.S])
        for t in range(self.S):
            
            weights=self._calculate_weight(weights)
            error=self._Fold(weights,x_train,y_train,self.F)
            errors[:,t]=error
            beta_t=m/(n+m)+t/(self.S-1)*(1-m/(n+m))
            for i in range(n):
                weights[i]=weights[i]*np.power(beta_t,error[i])
                #print(weights[i])
            #print(weights[:,0])

    def _calculate_weight(self,weights):
        weights=weights/np.sum(weights)
        return weights

    def predict(self,x_test):
        result=np.ones([x_test.shape[0],self.S+1])

    def _Fold(self,weights,X,y,F):
        kf=KFold(n_splits=F)
        error_=np.zeros([X.shape[0]])
        error_sums=[]
        temps=[]
        
        for train_index,test_index in kf.split(X):
            base_regressor=GradientBoostingRegressor()
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=y[train_index],y[test_index]
            base_regressor.fit(X_train,y_train,sample_weight=weights[train_index,0])
            temps.append(base_regressor)
            error=np.abs(y-base_regressor.predict(X))
            
            error_sum=np.sum(error)
            error_sums.append(error_sum)
            #print(error_sum)
            error=error/np.max(error)
            #print(error.shape,error_.shape)
            error_+=error
        #print(np.min(np.asarray(error_sums)))
        self.base_regressors.append(temps[np.argmin(np.asarray(error_sums))])
        error=error_/F
        return error