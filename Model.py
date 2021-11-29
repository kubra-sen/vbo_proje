import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
import pickle

from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



def define_regressors():

    base_models = [('LR', LinearRegression()),
                   ("Ridge", Ridge()),
                   ("Lasso", Lasso()),
                   ("ElasticNet", ElasticNet()),
                   ('KNN', KNeighborsRegressor()),
                   ('CART', DecisionTreeRegressor()),
                   ('RF', RandomForestRegressor()),
                   ('SVR', SVR()),
                   ('GBM', GradientBoostingRegressor()),
                   ("XGBoost", XGBRegressor(objective='reg:squarederror')),
                   ("LightGBM", LGBMRegressor()),
                   # ("CatBoost", CatBoostRegressor(verbose=False))
                   ]

    rf_params = {"max_depth": [3,5,7,None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [8, 15, 20],
                 "n_estimators": [100,150,200]}

    xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                      "max_depth": [5, 8, 12, 20],
                      "n_estimators": [100, 200, 300, 500],
                      "colsample_bytree": [0.5, 0.8, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                       "n_estimators": [300, 500, 1500],
                       "colsample_bytree": [0.5, 0.7, 1]}

    gbm_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [3, 8],
                  "n_estimators": [500, 1000],
                  "subsample": [1, 0.5, 0.7]}

    regressors = [
        ("RF", RandomForestRegressor(), rf_params),
        ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
        ('LightGBM', LGBMRegressor(), lightgbm_params),
        ("GBM", GradientBoostingRegressor(), gbm_params)
    ]
    return base_models, regressors


def train_tune(df,cv=10, scoring="neg_mean_squared_error"):
    X = df.drop(["SALARY"], axis=1)
    y = df["SALARY"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20, random_state=42)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    base_models, regressors= define_regressors()

    print("Base Models....")
    for name, model in base_models:
        rmse = np.mean(np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

    print("Hyperparameter Optimization....")
    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=10, scoring="neg_mean_squared_error",
                               n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = regressor.set_params(**gs_best.best_params_).fit(X_train, y_train)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=cv, scoring=scoring)))
        print (np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=cv, scoring=scoring)))
        print(f"RMSE (After tuning): {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}")

        best_models[name] = final_model

        # save the model to disk
        filename = 'finalized_model' + str(name) + '.sav'
        pickle.dump(best_models[name], open(filename, 'wb'))

    print(f"########## Ensemble Model ##########")
    voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]), ('LightGBM', best_models["LightGBM"])])

    voting_reg.fit(X_train, y_train)
    rmse = np.mean(np.sqrt(-cross_val_score(voting_reg, X_train, y_train, cv=cv, scoring=scoring)))
    print(f"RMSE for CV: {round(rmse, 4)} ")

    y_pred = voting_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE for Test: {round(rmse, 4)} ({'Ensemble of RF and GBM'})")

    filename = 'finalized_model' + str('voting') + '.sav'
    pickle.dump(voting_reg, open(filename, 'wb'))

    return best_models, voting_reg


def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


if __name__ == "__main__":
    df = pd.read_pickle('pickles/X_data_train.pkl')
    regressors = define_regressors()
    best_models, ensemble_model = train_tune(df,
                                             cv=10, scoring="neg_mean_squared_error")





