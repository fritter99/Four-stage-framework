import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
from TrAdaboost import TrAdaboost
from sklearn.ensemble import GradientBoostingRegressor

data=pd.read_excel(r'Dataset.xlsx',sheet_name=0)
x_target,y_target=np.asarray(data.iloc[:91,1:-1]),np.asarray(data.iloc[:91,-1])
x_source,y_source=np.asarray(data.iloc[91:,1:-1]),np.asarray(data.iloc[91:,-1]) 

X=np.concatenate((x_source,x_target))
y=np.concatenate((y_source,y_target))
x_target_train,x_target_test, y_target_train, y_target_test =train_test_split(x_target,y_target,test_size=0.4, random_state=0)

reg=GradientBoostingRegressor()
t=TrAdaboost(reg,S=100,F=10)
t.fit(x_source,x_target_train,y_source,y_target_train)
X_train=np.concatenate((x_source,x_target_train))
y_train=np.concatenate((y_source,y_target_train))
print('TrAdaboost的结果：')
rMax,maxNum=r2_score(y_target,t.base_regressors[0].predict(x_target)),0
for i in range(len(t.base_regressors)):
    print(i)
    metrics(y_target_train,t.base_regressors[i].predict(x_target_train)),metrics(y_target_test,t.base_regressors[i].predict(x_target_test)),\
         metrics(y_target,t.base_regressors[i].predict(x_target))
    if r2_score(y_target,t.base_regressors[i].predict(x_target))>rMax:
        rMax=r2_score(y_target,t.base_regressors[i].predict(x_target))
        maxNum=i