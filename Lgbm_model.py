import pandas as pd
import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

def load_application_train_main():
    data = pd.read_csv("datasets/project_datasets/application_train.csv")
    return data

data = load_application_train_main()


def FE_application_data(data):

    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['CREDIT_ANNUITY_PERCENT'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']

    data['FAMILY_CNT_INCOME_PERCENT'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['BIRTH_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['CHILDREN_CNT_INCOME_PERCENT'] = data['AMT_INCOME_TOTAL' ] /data['CNT_CHILDREN']

    data['CREDIT_GOODS_DIFF'] = data['AMT_CREDIT'] - data['AMT_GOODS_PRICE']
    data['EMPLOYED_REGISTRATION_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_REGISTRATION']
    data['BIRTH_REGISTRATION_PERCENT'] = data['DAYS_BIRTH'] / data['DAYS_REGISTRATION']
    data['ID_REGISTRATION_DIFF'] = data['DAYS_ID_PUBLISH'] - data['DAYS_REGISTRATION']

    data['ANNUITY_LENGTH_EMPLOYED_PERCENT'] = data['CREDIT_TERM' ]/ data['DAYS_EMPLOYED']

    data['AGE_LOAN_FINISH'] = data['DAYS_BIRTH' ] *(-1.0 /365) + \
                              (data['AMT_CREDIT' ] /data['AMT_ANNUITY']) *(1.0 / 12)
    # (This basically refers to the client's age when he/she finishes loan repayment)

    data['CAR_AGE_EMP_PERCENT'] = data['OWN_CAR_AGE'] / data['DAYS_EMPLOYED']
    data['CAR_AGE_BIRTH_PERCENT'] = data['OWN_CAR_AGE'] / data['DAYS_BIRTH']
    data['PHONE_CHANGE_EMP_PERCENT'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_EMPLOYED']
    data['PHONE_CHANGE_BIRTH_PERCENT'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']

    income_by_contract = data[['AMT_INCOME_TOTAL', 'NAME_CONTRACT_TYPE']].groupby('NAME_CONTRACT_TYPE').median()[
        'AMT_INCOME_TOTAL']
    data['MEDIAN_INCOME_CONTRACT_TYPE'] = data['NAME_CONTRACT_TYPE'].map(income_by_contract)

    income_by_suite = data[['AMT_INCOME_TOTAL', 'NAME_TYPE_SUITE']].groupby('NAME_TYPE_SUITE').median()[
        'AMT_INCOME_TOTAL']
    data['MEDIAN_INCOME_SUITE_TYPE'] = data['NAME_TYPE_SUITE'].map(income_by_suite)

    income_by_housing = data[['AMT_INCOME_TOTAL', 'NAME_HOUSING_TYPE']].groupby('NAME_HOUSING_TYPE').median()[
        'AMT_INCOME_TOTAL']
    data['MEDIAN_INCOME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].map(income_by_housing)

    income_by_org = data[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()[
        'AMT_INCOME_TOTAL']
    data['MEDIAN_INCOME_ORG_TYPE'] = data['ORGANIZATION_TYPE'].map(income_by_org)

    income_by_occu = data[['AMT_INCOME_TOTAL', 'OCCUPATION_TYPE']].groupby('OCCUPATION_TYPE').median()[
        'AMT_INCOME_TOTAL']
    data['MEDIAN_INCOME_OCCU_TYPE'] = data['OCCUPATION_TYPE'].map(income_by_occu)

    income_by_education = data[['AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE']].groupby('NAME_EDUCATION_TYPE').median()[
        'AMT_INCOME_TOTAL']
    data['MEDIAN_INCOME_EDU_TYPE'] = data['NAME_EDUCATION_TYPE'].map(income_by_education)

    data['ORG_TYPE_INCOME_PERCENT'] = data['MEDIAN_INCOME_ORG_TYPE'] / data['AMT_INCOME_TOTAL']
    data['OCCU_TYPE_INCOME_PERCENT'] = data['MEDIAN_INCOME_OCCU_TYPE'] / data['AMT_INCOME_TOTAL']
    data['EDU_TYPE_INCOME_PERCENT'] = data['MEDIAN_INCOME_EDU_TYPE'] / data['AMT_INCOME_TOTAL']

    data = data.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                      'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                      'FLAG_DOCUMENT_13',
                      'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                      'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                      'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1)

    cat_col = [category for category in data.columns if data[category].dtype == 'object']
    data = pd.get_dummies(data, columns=cat_col)

    return data

########################################
# LGBM Model
########################################

def lgbm_modelling(final_df, target):
    import re
    final_data_1 = final_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    y = final_data_1[target]
    X = final_data_1.drop([target], axis=1)
    lgbm_model = LGBMClassifier(random_state=17)
    lgbm_params = {"learning_rate": [0.01, 0.02, 0.03, 0.1, 0.001],
                   "n_estimators": [100, 250, 300, 350, 500, 1000],
                   "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1]}
    lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params,
                                  cv=5, n_jobs=-1, verbose=True).fit(X, y)
    lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,
                                       random_state=17).fit(X, y)
    pickle.dump(lgbm_final, open('lgbm_final.pkl', 'wb'))
    
    return lgbm_final


########################################
# Validation
########################################

cv_results = cross_validate(lgbm_final, X, y, cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_one = X.sample(1, random_state=45)
