
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_classif,chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def filter_fs(X_train, y_train, k=100, p_threshold=0.05):
    """
    A feature selection method that selects features according to their f_score for numeric values

    and chi_square score for object type values

    k: number of features to select
    p_threshold: the threshold value to select features

    """
    fscore_model = SelectKBest(score_func=f_classif, k=k)
    fscore_model.fit_transform(X_train, y_train)
    indices = fscore_model.get_support(indices=True)
    names = X_train.columns[fscore_model.get_support(indices=True)]
    pvalues = fscore_model.pvalues_[indices]
    fscores = list(zip(indices,names,pvalues, fscore_model.scores_[fscore_model.get_support()]))
    fscores_df = pd.DataFrame(data = fscores, columns = ['Indices','feat_names','p_value', 'filter_score'])
    df_filter = X_train[fscores_df[fscores_df['p_value']  < p_threshold].feat_names]
    df_filter = pd.merge (df_filter, y_train, left_on='SUREC_NO', right_on='SUREC_NO' )
    return df_filter, fscores_df


def wrapper_selection(X_train, y_train, model):

    #X_train, y_train = outlier_detection(X_train, y_train)

    wrapper_model = model
    wrapper_model = RFE(wrapper_model)

    wrapper_model.fit(X_train, y_train)

    rfe_rank = wrapper_model.ranking_
    features = list(zip(rfe_rank, X_train.columns))

    rfe_scores = pd.DataFrame(data = features, columns = ['ranks','feat_names']).sort_values(by='ranks')
    rfe_selected = np.where(rfe_scores.ranks==1)
    df_rfe = X_train[X_train.columns[rfe_selected]]
    df_rfe = pd.merge (df_rfe, y_train, left_on='SUREC_NO', right_on='SUREC_NO' )

    return df_rfe, rfe_rank


def embedded_lasso_fs(columns, X_train, y_train, model):

    # using logistic regression with penalty l1.
    lr_embedded = SelectFromModel(model)
    lr_embedded.fit(X_train, y_train)


    # see the selected features.
    names = X_train.columns[lr_embedded.get_support(indices=True)]
    indices_ = lr_embedded.get_support(indices=True)

    theta = lr_embedded.estimator_.coef_
    theta = theta.reshape(len(columns)-1,1)
    theta = np.concatenate( theta, axis=0 )

    # the dataset with the selected features
    df_lasso = X_train[X_train.columns[indices_]]
    df_lasso = pd.merge(df_lasso, y_train, left_on='SUREC_NO', right_on='SUREC_NO')

    return df_lasso,theta


def embedded_imp_fs(X_train, y_train, model):

    embedded = SelectFromModel(model)
    embedded.fit(X_train, y_train)

    # see the selected features.
    indices_ = embedded.get_support(indices=True)
    sel_feat = X_train.columns[indices_]
    importance = embedded.estimator_.feature_importances_

    # Seçilmiş featurelar ile dataset
    df_sel = X_train[sel_feat]
    df_sel = pd.merge (df_sel, y_train, left_on='SUREC_NO', right_on='SUREC_NO')

    return df_sel, importance, sel_feat




