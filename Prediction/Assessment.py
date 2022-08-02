from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
data=pd.read_excel(r'Experimental dataset.xlsx',sheet_name=0)
X,y=np.asarray(data.iloc[:91,1:-1]),np.asarray(data.iloc[:91,-1])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)

def get_MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def print_coefficient(labels,predicts):
    RMSE = np.sqrt(mean_squared_error(labels,predicts))
    print('RMSE of model on the  data set: {0:6.3f}'.format(RMSE))
    MAE = mean_absolute_error(labels,predicts)
    print('MAE of model on the  data set: {0:6.3f}'.format(MAE))
    MAPE = get_MAPE(labels,predicts)
    print('MAPE of model on the data set: {0:6.3f}'.format(MAPE))
    R2 = r2_score(labels,predicts)
    print('R2 of model on the  data set: {0:6.3f}'.format(R2))
    df2=pd.DataFrame({'Test':[RMSE, MAE, MAPE, R2]}
                         ,index=['RMSE', 'MAE', 'MAPE','R2'])

    df2.style.format("{:.4f}")  

#RF
from sklearn.ensemble import RandomForestRegressor
param_grid = [
{ 'n_estimators': range(3,12), 'max_features': [2, 3, 4, 5, 6,7,8]},
]
reg = RandomForestRegressor()
grid_search = GridSearchCV(reg, param_grid, cv=10,
                          scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train,grid_search.best_estimator_.predict(X_train))
print('----------------')
print("Testing set: ")
print_coefficient(y_test,grid_search.best_estimator_.predict(X_test))
print('----------------')
print("Dataset: ")
print_coefficient(y,grid_search.best_estimator_.predict(X))

#GBDT
from sklearn.ensemble import GradientBoostingRegressor
param_grid = { 'n_estimators': [400], 'random_state': range(1,3),'max_depth':range(4,10),
              "learning_rate":[0.01,0.03,0.05,0.07,0.08,0.09,0.1]}

reg = GradientBoostingRegressor()
grid_search = GridSearchCV(reg, param_grid, cv=10,
                          scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

#AdaBoost
from sklearn.ensemble import AdaBoostRegressor
param_grid = { 'n_estimators':[500], 'random_state': range(5),
              "learning_rate":np.logspace(-1,1,5)}

reg = AdaBoostRegressor()
grid_search = GridSearchCV(reg, param_grid, cv=10,
                          scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train,grid_search.best_estimator_.predict(X_train))
print('----------------')
print("Testing set: ")
print_coefficient(y_test,grid_search.best_estimator_.predict(X_test))
print('----------------')
print("Dataset: ")
print_coefficient(y,grid_search.best_estimator_.predict(X))


from sklearn import preprocessing
scaler= preprocessing.StandardScaler()
scaler.fit(X_train)
X_train_scale=scaler.transform(X_train)
X_test_scale=scaler.transform(X_test)

scaler_= preprocessing.StandardScaler()
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
scaler_.fit(y_train)
y_train_scale=scaler_.transform(y_train)
y_test_scale=scaler_.transform(y_test)

#SVM
from sklearn.svm import SVR
grid_search  = GridSearchCV(SVR(), param_grid={"kernel": ["linear","rbf"], "C": range(1000,5000,1000), 
                            'epsilon':[0.1,0.2,0.3,0.4,0.5,0.6]},scoring='neg_mean_squared_error',cv=10)
grid_search .fit(X_train_scale, y_train_scale.ravel())

print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train.ravel(),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_train_scale).reshape(-1,1)).ravel())
print('----------------')
print("Testing set: ")
print_coefficient(y_test.ravel(),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_test_scale).reshape(-1,1)).ravel())
print('----------------')
print("Dataset: ")
y_predict=np.vstack((scaler_.inverse_transform(grid_search.best_estimator_.predict(X_train_scale).reshape(-1,1)),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_test_scale).reshape(-1,1))))
y=np.vstack((y_train,y_test))
print_coefficient(y.ravel(),y_predict.ravel())

#ANN
from sklearn.neural_network import MLPRegressor
grid_search  = GridSearchCV(MLPRegressor(), param_grid={"activation": ["relu","tanh","identity"], "hidden_layer_sizes": range(5,20,10), 
                            'epsilon':np.logspace(-3,-1,5),'max_iter':[4000]},scoring='neg_mean_squared_error',cv=10)
grid_search.fit(X_train_scale,y_train_scale.ravel())
print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train.ravel(),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_train_scale).reshape(-1,1)).ravel())
print('----------------')
print("Testing set: ")
print_coefficient(y_test.ravel(),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_test_scale).reshape(-1,1)).ravel())
print('----------------')
print("Dataset: ")
y_predict=np.vstack((scaler_.inverse_transform(grid_search.best_estimator_.predict(X_train_scale).reshape(-1,1)),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_test_scale).reshape(-1,1))))
y=np.vstack((y_train,y_test))
print_coefficient(y.ravel(),y_predict.ravel())

#KNN
from sklearn.neighbors import KNeighborsRegressor
param_grid={'n_neighbors':range(3,10),'leaf_size':range(10,100,10),'p':range(2,10)}
reg=KNeighborsRegressor()
grid_search = GridSearchCV(reg, param_grid, cv=10,
                          scoring='neg_mean_squared_error')
grid_search.fit(X_train_scale,y_train_scale.ravel())
print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train.ravel(),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_train_scale).reshape(-1,1)).ravel())
print('----------------')
print("Testing set: ")
print_coefficient(y_test.ravel(),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_test_scale).reshape(-1,1)).ravel())
print('----------------')
print("Dataset: ")
y_predict=np.vstack((scaler_.inverse_transform(grid_search.best_estimator_.predict(X_train_scale).reshape(-1,1)),scaler_.inverse_transform(grid_search.best_estimator_.predict(X_test_scale).reshape(-1,1))))
y=np.vstack((y_train,y_test))
print_coefficient(y.ravel(),y_predict.ravel())
