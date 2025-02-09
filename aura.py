import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import time
# from sklearn.xgboost import XGBRegressor

class RegressionModel():
    def __init__(self,  random_state = 42):
        # Initialize ShuffleSplit for cross-validation
        self.cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        self.best_model = None
        self.best_score = -1
        self.best_parameters=None
    
    
        self.algos = {
        'Linear Regression': {
        'model': LinearRegression(),
        'params': {
        'fit_intercept': [True, False],
        'positive': [True, False],
        'copy_X': [True, False]
        }
        },
        'Ridge': {
        'model': Ridge(),
        'params': {
        'alpha': [0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
        },
        'Lasso': {
        'model': Lasso(),
        'params': {
        'alpha': [0.1, 1, 10],
        'selection': ['random', 'cyclic'],
        'max_iter': [1000, 5000]
        }
        },
        'Elastic_net': {
        'model': ElasticNet(),
        'params': {
        'alpha': [0.1, 1, 10],
        'l1_ratio': [0.1, 0.5, 0.9],
        'max_iter': [1000, 5000]
        }
        },
        'Decision Tree': {
        'model': DecisionTreeRegressor(),
        'params': {
        'criterion': ['absolute_error', 'poisson', 'squared_error', 'friedman_mse'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
        }
        },
        'Random Forest': {
        'model': RandomForestRegressor(),
        'params': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
        }
        },
        'Gradient Boosting': {
        'model': GradientBoostingRegressor(),
        'params': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'subsample': [0.7, 0.8, 1.0],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
        }
        },
        'Ada Boost': {
        'model': AdaBoostRegressor(),
        'params': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'loss': ['linear', 'square', 'exponential']
        }
        },
        'SVR': {
        'model': SVR(),
        'params': {
        'C': [1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 0.2]
        }
        },
        'K_Neighbors': {
        'model': KNeighborsRegressor(),
        'params': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
        }
        },
        'Bayesian Ridge': {
        'model': BayesianRidge(),
        'params': {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4],
        }
        }
        }
    def Best_Algo_GSCV(self,X_train,y_train, X_test=None, y_test=None):
        # Dictionary to store the results
        self.results = {}
        
        # Iterate through each model and perform GridSearchCV
        for algo_name, config in self.algos.items():
            self.time = time.time()
            print(f"Running GridSearchCV for:   {algo_name}  ...")
            
            # Apply GridSearchCV to each model
            gs = GridSearchCV(config['model'], config['params'], cv=self.cv, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            y_pred = gs.best_estimator_.predict(X_test)
            test_score = r2_score(y_test, y_pred)
            
            
            # Store the best results
            self.results[algo_name] = {
            'best_score': gs.best_score_,
            'best_params': gs.best_params_,

            }
            
            if gs.best_score_>self.best_score:
                self.best_model=algo_name
                self.best_score= gs.best_score_
                self.best_parameters=gs.best_params_
            
            # Print out the results for each model
            print(f"{algo_name}: Best Score: {gs.best_score_}")
            print(f"Test Score: {test_score}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Time taken: {time.time()-self.time:.1f} sec.")
            print("-" * 100)
            # Return the results dictionary

        print(f" Best ML Model: {self.best_model}\n Best Score: {self.best_score} \n Best Parameters: {self.best_parameters}")
        return self.results
    
    def Best_Algo_RSCV(self,X_train,y_train, X_test=None, y_test=None ):
        self.results = {}
        
        for algo_name, config in self.algos.items():
            self.time = time.time()
            print(f"\nRunning RandomSearchCV for:  {algo_name}  ...")
            
            gs = RandomizedSearchCV(config['model'], config['params'], n_iter=25,cv=self.cv, scoring='r2', n_jobs=-1) #-1 use all available processures
            gs.fit(X_train, y_train)
            y_pred = gs.best_estimator_.predict(X_test)
            test_score = r2_score(y_test, y_pred)
            
            # Store the best results
            self.results[algo_name] = {
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
            }
            
            if gs.best_score_>self.best_score:
                self.best_model=algo_name
                self.best_score= gs.best_score_
                self.best_parameters=gs.best_params_
            
            # Print out the results for each model
            print(f"{algo_name}: Best Score: {gs.best_score_}")
            print(f"Test Score: {test_score}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Time taken: {time.time()-self.time:.1f} sec.")
            print("-" * 100,'\n')
            
        print(f" Best ML Model: {self.best_model}\n Best Score: {self.best_score} \n Best Parameters: {self.best_parameters}")
            
        return self.results
    
    # def optuna_search(self,X_train,y_train, X_test=None, y_test=None):
    #     def objective(trial):
    #         algo_name = trial.suggest_categorical('algo', list(self.algos.keys()))
    #         config = self.algos[algo_name]
    #         params ={}
    #         for key,value in config['params'].items():
    #             isinstances(value, list)
        
        
    
