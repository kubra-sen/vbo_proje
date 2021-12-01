########################################################################
# Libraries and Packages
########################################################################
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import  recall_score, precision_score, f1_score
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.model_selection import GridSearchCV, cross_validate


#############################################################
# FEATURE SELECTION
#############################################################
train_data = pd.read_pickle('pickles/train_data.pkl')
test_data = pd.read_pickle('pickles/test_data')
X_train_final_hcdr_new = pd.read_pickle('pickles/X_train_final_hcdr_new')
Y_train_final_hcdr_new = pd.read_pickle('pickles/Y_train_final_hcdr_new')
X_cv_final_hcdr_new = pd.read_pickle('pickles/X_cv_final_hcdr_new')
Y_cv_final_hcdr_new = pd.read_pickle('pickles/Y_cv_final_hcdr_new')

## E.Obtaining the Dataframe from only the Top 500 Important Features

X_train_final_arr = np.nan_to_num(X_train_final_hcdr_new)
X_cv_final_arr = np.nan_to_num(X_cv_final_hcdr_new)

S = SelectKBest(f_classif, k=500)

X_train_k_best = S.fit_transform(X_train_final_arr, Y_train_final_hcdr_new)
X_cv_k_best = S.transform(X_cv_final_arr)

# Get columns to keep and create new dataframe with those only
cols = S.get_support(indices=True)

features_top_df_train = X_train_final_hcdr_new.iloc[:,cols]
features_top_df_cv = X_cv_final_hcdr_new.iloc[:,cols]

if not os.path.isfile('pickles/features_top_df_train.pkl'):
    features_top_df_train.to_pickle('pickles/features_top_df_train.pkl')
features_top_df_train = pd.read_pickle('pickles/features_top_df_train.pkl')


#################################################
#ML MODEL
####################################################

def define_classifiers():

    base_models = [
        ("LGBM", LGBMClassifier())
    ]

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

    lr_params= { "class_weight":["balanced"],
                 "tol" : [0.0001,0.0002,0.0003],
                 "max_iter": [100,200,300],
                 "C" :[0.001,0.01, 0.1, 1, 10, 100],
                 "intercept_scaling": [1, 2, 3, 4]}

    classifiers = [ ('LR', LogisticRegression(), lr_params)]
    return base_models, classifiers


def train_tune(X_train, y_train, X_cv, y_cv,cv=3, scoring=["roc_auc"]):

    base_models, classifiers= define_classifiers()

    print("Base Models....")
    for name, model in base_models:
        cv_results = cross_validate(model, X_train, y_train, cv=3, scoring=scoring)
        print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


    print("Hyperparameter Optimization....")
    best_models = {}

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")

        gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model

    print(f"########## Ensemble Model ##########")
    voting_clf = VotingClassifier(
        estimators=[('XGBoost', best_models["XGBoost"]),
                    ('RF', best_models["RF"]),
                    ('LightGBM', best_models["LightGBM"])],
        voting='soft')

    voting_clf.fit(X_train, y_train)

    cv_results = cross_validate(voting_clf, X_train, y_train, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"])

    print(f"########## CV Results ##########")

    acc = cv_results['test_accuracy'].mean()
    f1 = cv_results['test_f1'].mean()
    roc = cv_results['test_roc_auc'].mean()
    print(f"CV Results: {round(acc, 4), round(f1, 4),round(roc, 4)} ")

    y_pred = voting_clf.predict(X_cv)
    test_acc = voting_clf.score(y_cv, y_pred)
    precision = round(precision_score(y_cv, y_pred), 4)
    recall = round(recall_score(y_cv, y_pred), 4)
    f1 = round(f1_score(y_cv, y_pred, average = 'macro') ,4)
    print(f"Test Results: {round(test_acc, 4), round(precision, 4),round(recall, 4)} ({'Ensemble of RF and GBM'})")

    # TO-DO : Confusion Matrix
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

    ## F.Standardising the final dataset obtained

    scaler = StandardScaler()

    features_top_df_test = test_data[features_top_df_train.columns]
    features_top_df_test_final = np.nan_to_num(features_top_df_test)
    features_top_df_cv_final = np.nan_to_num(features_top_df_cv)

    scaler.fit(features_top_df_train)
    X_train = scaler.transform(features_top_df_train)
    X_cv = scaler.transform(features_top_df_cv_final)
    X_test = scaler.transform(features_top_df_test_final)

    Y_train = np.nan_to_num(Y_train_final_hcdr_new)
    Y_cv = np.nan_to_num(Y_cv_final_hcdr_new)

    scaler_X_train = np.nan_to_num(X_train)
    scaler_X_cv = np.nan_to_num(X_cv)
    scaler_X_test = np.nan_to_num(X_test)


    best_models, ensemble_model = train_tune(scaler_X_train,Y_train,scaler_X_cv,Y_cv,
                                             cv=3, scoring="neg_mean_squared_error")

#############################
#DENEME
###############################3

base_models, classifiers= define_classifiers()

print("Base Models....")
for name, model in base_models:
    cv_results = cross_validate(model, scaler_X_train, Y_train, cv=3, scoring=["roc_auc"])
    print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}({name})")


print("Hyperparameter Optimization....")
best_models = {}

for name, classifier, params in classifiers:
    print(f"########## {name} ##########")

    gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(scaler_X_train, Y_train)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, scaler_X_train, Y_train, cv=3, scoring=["roc_auc"])
    print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}({name})")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

print(f"########## Ensemble Model ##########")
voting_clf = VotingClassifier(
    estimators=[('XGBoost', best_models["XGBoost"]),
                ('RF', best_models["RF"]),
                ('LightGBM', best_models["LightGBM"])],
    voting='soft')

voting_clf.fit(X_train, Y_train)

cv_results = cross_validate(voting_clf, scaler_X_train, Y_train, cv=3,
                            scoring=["accuracy", "f1", "roc_auc"])

print(f"########## CV Results ##########")

acc = cv_results['test_accuracy'].mean()
f1 = cv_results['test_f1'].mean()
roc = cv_results['test_roc_auc'].mean()
print(f"CV Results: {round(acc, 4), round(f1, 4),round(roc, 4)} ")

y_pred = voting_clf.predict(scaler_X_cv)
test_acc = voting_clf.score(Y_cv, y_pred)
precision = round(precision_score(Y_cv, y_pred), 4)
recall = round(recall_score(Y_cv, y_pred), 4)
f1 = round(f1_score(Y_cv, y_pred, average = 'macro') ,4)
print(f"Test Results: {round(test_acc, 4), round(precision, 4),round(recall, 4)} ({'Ensemble of RF and GBM'})")


# TO-DO : Confusion Matrix




