import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from helpers.eda import *
from helpers.data_prep import *

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import  roc_curve, auc
from sklearn.metrics import  recall_score, precision_score, f1_score

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



def define_classifiers():

    base_models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier()),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    knn_params = {"n_neighbors": range(2, 50)}

    cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

    xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

    classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]
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

    base_models, classifiers= define_classifiers()

    print("Base Models....")
    for name, model in base_models:
        rmse = np.mean(np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

    print("Hyperparameter Optimization....")
    best_models = {}

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
        print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


        gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=3, scoring=["roc_auc"])
        print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model

        # save the model to disk
        filename = 'finalized_model' + str(name) + '.sav'
        pickle.dump(best_models[name], open(filename, 'wb'))

    print(f"########## Ensemble Model ##########")
    voting_clf = VotingClassifier(
        estimators=[('XGBoost', best_models["XGBoost"]),
                    ('RF', best_models["RF"]),
                    ('LightGBM', best_models["LightGBM"])],
        voting='soft')

    voting_clf.fit(X_train, y_train)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    acc = cv_results['test_accuracy'].mean()
    f1 = cv_results['test_f1'].mean()
    roc = cv_results['test_roc_auc'].mean()
    print(f"Accuracy for CV: {round(acc, 4)} ")

    y_pred = voting_clf.predict(X_test)
    test_acc = voting_clf.score(X_test, y_test)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred, average = 'macro') ,4)
    print(f"RMSE for Test: {round(test_acc, 4)} ({'Ensemble of RF and GBM'})")

    filename = 'finalized_model' + str('voting') + '.sav'
    pickle.dump(voting_clf, open(filename, 'wb'))

    return best_models, voting_clf


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
    regressors = define_classifiers()
    best_models, ensemble_model = train_tune(df,
                                             cv=10, scoring="neg_mean_squared_error")





