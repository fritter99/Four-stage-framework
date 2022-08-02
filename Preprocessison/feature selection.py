import pandas as pd
from sklearn.model_selection import train_test_split
data_0=pd.read_excel(r'Raw dataset.xlsx',sheet_name=0)
X_0,y_0=data_0.iloc[:,1:-1],data_0.iloc[:,-1]
X_train_0,X_test_0,y_train_0,y_test_0=train_test_split(X_0,y_0,test_size=0.4)

data_1=pd.read_excel(r'Raw dataset.xlsx',sheet_name=1)
X_1,y_1=data_1.iloc[:,1:-1],data_1.iloc[:,-1]
X_train_1,X_test_1,y_train_0,y_test_1=train_test_split(X_1,y_1,test_size=0.4)

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

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
{ 'n_estimators': range(3,12), 'max_features': [2, 3, 4, 5, 6,7,8]},
]
reg = RandomForestRegressor()
grid_search = GridSearchCV(reg, param_grid, cv=10,
                          scoring='neg_mean_squared_error')
#11 parameters
grid_search.fit(X_train_0, y_train_0)
print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train_0,grid_search.best_estimator_.predict(X_train_0))
print('----------------')
print("Testing set: ")
print_coefficient(y_test_0,grid_search.best_estimator_.predict(X_test_0))
print('----------------')
print("Dataset: ")
print_coefficient(y_0,grid_search.best_estimator_.predict(X_0))

#9 parameters
from sklearn.ensemble import RandomForestRegressor
param_grid = [
{ 'n_estimators': range(3,12), 'max_features': [2, 3, 4, 5, 6,7,8]},
]
reg = RandomForestRegressor()
grid_search = GridSearchCV(reg, param_grid, cv=10,
                          scoring='neg_mean_squared_error')
grid_search.fit(X_train_1, y_train_1)
print(grid_search.best_params_)

print("Training set: ")
print_coefficient(y_train_1,grid_search.best_estimator_.predict(X_train_1))
print('----------------')
print("Testing set: ")
print_coefficient(y_test_1,grid_search.best_estimator_.predict(X_test_1))
print('----------------')
print("Dataset: ")
print_coefficient(y_1,grid_search.best_estimator_.predict(X_1))
