import numpy as np
import gc
import pandas as pd
from helpers.data_prep import *


def feature_early_shutdown(row):
    early_shutdown = 0
    if row.CREDIT_ACTIVE == "Active" and row.DAYS_CREDIT_ENDDATE < 0:
        early_shutdown = 1
    return early_shutdown


def one_hot_encode(df):
    #Refer :- https://www.kaggle.com/jsaguiar/lightgbm-7th-place-solution
    #one hot encode the categorical data
    original_columns = list(df.columns)
    categories = [cat for cat in df.columns if df[cat].dtype == 'object']
    df = pd.get_dummies(df, columns= categories, dummy_na= True) #one_hot_encode the categorical features
    categorical_columns = [cat for cat in df.columns if cat not in original_columns]
    return df, categorical_columns


def bureau_fe(bu_df):

    bu_df.fillna(0, inplace=True)
    ################ Feature Engineering from mvk notebook ###############
    bu_df_new = pd.DataFrame()
    bu_df_new["BURO_CREDIT_APPLICATION_COUNT"] = bu_df.groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]

    # aktif kredi sayısı
    bu_df_new["BURO_ACTIVE_CREDIT_APPLICATION_COUNT"] = \
        bu_df[bu_df["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").count()["CREDIT_ACTIVE"]
    bu_df_new["BURO_ACTIVE_CREDIT_APPLICATION_COUNT"].fillna(0, inplace=True)

    # pasif kredi sayısı
    bu_df_new["BURO_CLOSED_CREDIT_APPLICATION_COUNT"] = \
        bu_df[bu_df["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR").count()["CREDIT_ACTIVE"]
    bu_df_new["BURO_CLOSED_CREDIT_APPLICATION_COUNT"].fillna(0, inplace=True)

    # erken kredi kapama
    bu_df_new["BURO_EARLY_SHUTDOWN_NEW"] = bu_df.apply(lambda x: feature_early_shutdown(x), axis=1)

    # geciktirilmiş ödeme sayısı
    bu_df_new["BURO_NUMBER_OF_DELAYED_PAYMENTS"] = \
        bu_df[bu_df["AMT_CREDIT_MAX_OVERDUE"] != 0].groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].count()
    bu_df_new["BURO_NUMBER_OF_DELAYED_PAYMENTS"].fillna(0, inplace=True)

    # son kapanmış başvurusu üzerinden geçen max süre
    bu_df_new["BURO_MAX_TIME_PASSED_CREDIT_APPLICATION"] = \
        bu_df[bu_df["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR")["DAYS_ENDDATE_FACT"].max()
    bu_df_new["BURO_MAX_TIME_PASSED_CREDIT_APPLICATION"].fillna(0, inplace=True)

    # geciktirilmiş max ödeme tutari
    bu_df_new["BURO_MAX_DELAYED_PAYMENTS"] = bu_df.groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].max()
    bu_df_new["BURO_MAX_DELAYED_PAYMENTS"].fillna(0, inplace=True)

    # geciktirilmiş ödeyenlerden oluşan top liste - en yüksek 100
    # gecikme olan (80302, 12)
    bu_df_new["BURO_DELAYED_PAYMENTS_TOP_100_NEW"] = \
        bu_df_new.sort_values("BURO_MAX_DELAYED_PAYMENTS", ascending=False)["BURO_MAX_DELAYED_PAYMENTS"].rank()
    bu_df_new["BURO_DELAYED_PAYMENTS_TOP_100_NEW"].fillna(0, inplace=True)

    # kredi uzatma yapilmis mi
    bu_df_new["BURO_IS_CREDIT_EXTENSION_NEW"] = bu_df.groupby("SK_ID_CURR")["CNT_CREDIT_PROLONG"].count().apply(
        lambda x: 1 if x > 0 else 0)

    # max yapilan kredi uzatmasi
    bu_df_new["BURO_CREDIT_EXTENSION_MAX"] = bu_df.groupby("SK_ID_CURR")["CNT_CREDIT_PROLONG"].max()
    bu_df_new["BURO_CREDIT_EXTENSION_MAX"].fillna(0, inplace=True)

    # unsuccessful credit payment - borç takarak kapanmış kredi ödemeleri tespit et
    bu_df_new["BURO_IS_UNSUCCESSFUL_CREDIT_PAYMENT_NEW"] = \
        bu_df[(bu_df["CREDIT_ACTIVE"] == "Closed") & (bu_df["AMT_CREDIT_SUM_DEBT"] > 0)].groupby(
            "SK_ID_CURR").all()["AMT_CREDIT_SUM_DEBT"].apply(lambda x: 1 if x == True else 0)
    bu_df_new["BURO_IS_UNSUCCESSFUL_CREDIT_PAYMENT_NEW"].fillna(0, inplace=True)


    ####################### Feature Engineering ekstra############################33
    # Ref: https://www.kaggle.com/c/home-credit-default-risk/discussion/57750

    # Number of types of past loans per customer
    bu_df['CREDIT_TYPE'].isnull().sum()
    bu_df_new['PAST_LOANS_NO_CR_TYP']= bu_df[['SK_ID_CURR', 'CREDIT_TYPE']]. \
        groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique()

    # Average number of Past loans per type of loan per customer
    bu_df_new['AVG_NO_PAST_CR_TYP'] = bu_df_new['BURO_CREDIT_APPLICATION_COUNT']/ bu_df_new['PAST_LOANS_NO_CR_TYP']


    # The Ratio of Total Debt to Total Credit for each Customer
    bu_df['AMT_CREDIT_SUM_DEBT'].fillna(0,inplace=True)
    bu_df['AMT_CREDIT_SUM'].fillna(0,inplace=True)

    bu_df_new['TOTAL_CUSTOMER_DEBT'] = bu_df[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by= \
                                                                                               ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum()

    bu_df_new['TOTAL_CUSTOMER_CREDIT'] = bu_df[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(by= \
                                                                                            ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum()

    bu_df_new['DEBT_CREDIT_RATIO'] = bu_df_new['TOTAL_CUSTOMER_DEBT']/bu_df_new['TOTAL_CUSTOMER_CREDIT']

    # Fraction of Total Debt Overdue for each customer
    bu_df['AMT_CREDIT_SUM_OVERDUE'].fillna(0,inplace=True)
    bu_df_new['TOTAL_CUSTOMER_OVERDUE'] =bu_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']]. \
        groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum()

    bu_df_new['DEBT_CREDIT_RATIO'] = bu_df_new['TOTAL_CUSTOMER_OVERDUE']/bu_df_new['TOTAL_CUSTOMER_DEBT']

    bu_df_new = bu_df.merge(bu_df_new, on='SK_ID_CURR', how='left')
    bu_df_new = bu_df_new.reset_index()

    return bu_df_new


def bureau_bureau_balance_fe(bureau_data_fe,bb_df):

    bb_df.fillna(0, inplace=True)

    bureau_data, bureau_data_columns = one_hot_encode(bureau_data_fe)
    bureau_balance, bureau_balance_columns = one_hot_encode(bb_df)

    bureau_balance_agg = {'MONTHS_BALANCE': ['min','max','mean','size']}

    for column in bureau_balance_columns:
        bureau_balance_agg[column] = ['min','max','mean','size']
        bureau_balance_final_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bureau_balance_agg)

    col_list_1 =[]

    for col in bureau_balance_final_agg.columns.tolist():
        col_list_1.append(col[0] + "_" + col[1].upper())

    bureau_balance_final_agg.columns = pd.Index(col_list_1)
    bureau_data_balance = bureau_data.join(bureau_balance_final_agg, how='left', on='SK_ID_BUREAU')
    bureau_data_balance.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

    del bureau_balance_final_agg
    gc.collect()


    numerical_agg = {'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],'AMT_CREDIT_SUM_OVERDUE': ['mean','sum'],
                     'DAYS_CREDIT': ['mean', 'var'],'DAYS_CREDIT_UPDATE': ['mean','min'],'CREDIT_DAY_OVERDUE': ['mean','min'],
                     'DAYS_CREDIT_ENDDATE': ['mean'],'CNT_CREDIT_PROLONG': ['sum'],'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
                     'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],'AMT_CREDIT_MAX_OVERDUE': ['mean','max'],
                     'AMT_ANNUITY': ['max', 'mean','sum'],'AMT_CREDIT_SUM': ['mean', 'sum','max']
                     }
    categorical_agg = {}

    for col in bureau_data_columns:
        categorical_agg[col] = ['mean']
        categorical_agg[col] = ['max']

    for col in bureau_balance_columns:
        categorical_agg[col + "_MEAN"] = ['mean']
        categorical_agg[col + "_MIN"] = ['min']
        categorical_agg[col + "_MAX"] = ['max']

    bureau_data_balance_2 = bureau_data_balance.groupby('SK_ID_CURR').agg({**numerical_agg, \
                                                                           **categorical_agg})
    col_list_2=[]

    for col in bureau_data_balance_2.columns.tolist():
        col_list_2.append('BUREAU_'+col[0]+'_'+col[1])
    bureau_data_balance_2.columns = pd.Index(col_list_2)


    bureau_data_balance_3 = bureau_data_balance[bureau_data_balance['CREDIT_ACTIVE_Active'] == 1]
    bureau_data_balance_3_agg = bureau_data_balance_3.groupby('SK_ID_CURR').agg(numerical_agg)

    col_list_3=[]
    for col in bureau_data_balance_3_agg.columns.tolist():
        col_list_3.append('A_'+col[0]+'_'+col[1].upper())

    bureau_data_balance_3_agg.columns = pd.Index(col_list_3)
    b3_final = bureau_data_balance_2.join(bureau_data_balance_3_agg, how='left', \
                                          on='SK_ID_CURR')

    bureau_data_balance_4 = bureau_data_balance[bureau_data_balance['CREDIT_ACTIVE_Closed'] == 1]
    bureau_data_balance_4_agg = bureau_data_balance_4.groupby('SK_ID_CURR').agg(numerical_agg)
    col_list_4 =[]

    for col in bureau_data_balance_4_agg.columns.tolist():
        col_list_4.append('C_'+col[0]+'_'+col[1].upper())

    bureau_data_balance_4_agg.columns = pd.Index(col_list_4)
    bureau_data_balance_final = bureau_data_balance_2.join(bureau_data_balance_4_agg, \
                                                           how='left', on='SK_ID_CURR')

    del bureau_data_balance_3, bureau_data_balance_4_agg
    gc.collect()

    return bureau_data_balance_final


def application(app_df):
    app_df['CREDIT_INCOME_PERCENT'] = app_df['AMT_CREDIT'] / app_df['AMT_INCOME_TOTAL']
    app_df['ANNUITY_INCOME_PERCENT'] = app_df['AMT_ANNUITY'] / app_df['AMT_INCOME_TOTAL']
    app_df['CREDIT_ANNUITY_PERCENT'] = app_df['AMT_CREDIT'] / app_df['AMT_ANNUITY']

    app_df['FAMILY_CNT_INCOME_PERCENT'] = app_df['AMT_INCOME_TOTAL'] / app_df['CNT_FAM_MEMBERS']
    app_df['CREDIT_TERM'] = app_df['AMT_ANNUITY'] / app_df['AMT_CREDIT']
    app_df['BIRTH_EMPLOYED_PERCENT'] = app_df['DAYS_EMPLOYED'] / app_df['DAYS_BIRTH']
    app_df['CHILDREN_CNT_INCOME_PERCENT'] = app_df['AMT_INCOME_TOTAL']/app_df['CNT_CHILDREN']

    app_df['CREDIT_GOODS_DIFF'] = app_df['AMT_CREDIT'] - app_df['AMT_GOODS_PRICE']
    app_df['EMPLOYED_REGISTRATION_PERCENT'] = app_df['DAYS_EMPLOYED'] / app_df['DAYS_REGISTRATION']
    app_df['BIRTH_REGISTRATION_PERCENT'] = app_df['DAYS_BIRTH'] / app_df['DAYS_REGISTRATION']
    app_df['ID_REGISTRATION_DIFF'] = app_df['DAYS_ID_PUBLISH'] - app_df['DAYS_REGISTRATION']

    app_df['ANNUITY_LENGTH_EMPLOYED_PERCENT'] = app_df['CREDIT_TERM']/ app_df['DAYS_EMPLOYED']

    app_df['AGE_LOAN_FINISH'] = app_df['DAYS_BIRTH']*(-1.0/365) + \
                              (app_df['AMT_CREDIT']/app_df['AMT_ANNUITY']) *(1.0/12)
    # (This basically refers to the client's age when he/she finishes loan repayment)

    app_df['CAR_AGE_EMP_PERCENT'] = app_df['OWN_CAR_AGE']/app_df['DAYS_EMPLOYED']
    app_df['CAR_AGE_BIRTH_PERCENT'] = app_df['OWN_CAR_AGE']/app_df['DAYS_BIRTH']
    app_df['PHONE_CHANGE_EMP_PERCENT'] = app_df['DAYS_LAST_PHONE_CHANGE']/app_df['DAYS_EMPLOYED']
    app_df['PHONE_CHANGE_BIRTH_PERCENT'] = app_df['DAYS_LAST_PHONE_CHANGE']/app_df['DAYS_BIRTH']

    income_by_contract = app_df[['AMT_INCOME_TOTAL', 'NAME_CONTRACT_TYPE']].groupby('NAME_CONTRACT_TYPE').median()['AMT_INCOME_TOTAL']
    app_df['MEDIAN_INCOME_CONTRACT_TYPE'] = app_df['NAME_CONTRACT_TYPE'].map(income_by_contract)

    income_by_suite = app_df[['AMT_INCOME_TOTAL', 'NAME_TYPE_SUITE']].groupby('NAME_TYPE_SUITE').median()['AMT_INCOME_TOTAL']
    app_df['MEDIAN_INCOME_SUITE_TYPE'] = app_df['NAME_TYPE_SUITE'].map(income_by_suite)

    income_by_housing = app_df[['AMT_INCOME_TOTAL', 'NAME_HOUSING_TYPE']].groupby('NAME_HOUSING_TYPE').median()['AMT_INCOME_TOTAL']
    app_df['MEDIAN_INCOME_HOUSING_TYPE'] = app_df['NAME_HOUSING_TYPE'].map(income_by_housing)

    income_by_org = app_df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    app_df['MEDIAN_INCOME_ORG_TYPE'] = app_df['ORGANIZATION_TYPE'].map(income_by_org)

    income_by_occu = app_df[['AMT_INCOME_TOTAL', 'OCCUPATION_TYPE']].groupby('OCCUPATION_TYPE').median()['AMT_INCOME_TOTAL']
    app_df['MEDIAN_INCOME_OCCU_TYPE'] = app_df['OCCUPATION_TYPE'].map(income_by_occu)

    income_by_education = app_df[['AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE']].groupby('NAME_EDUCATION_TYPE').median()['AMT_INCOME_TOTAL']
    app_df['MEDIAN_INCOME_EDU_TYPE'] = app_df['NAME_EDUCATION_TYPE'].map(income_by_education)

    app_df['ORG_TYPE_INCOME_PERCENT'] = app_df['MEDIAN_INCOME_ORG_TYPE']/app_df['AMT_INCOME_TOTAL']
    app_df['OCCU_TYPE_INCOME_PERCENT'] = app_df['MEDIAN_INCOME_OCCU_TYPE']/app_df['AMT_INCOME_TOTAL']
    app_df['EDU_TYPE_INCOME_PERCENT'] = app_df['MEDIAN_INCOME_EDU_TYPE']/app_df['AMT_INCOME_TOTAL']

    app_df= app_df.drop(['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
                     'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
                     'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
                     'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21'],axis=1)

    cat_col = [category for category in app_df.columns if app_df[category].dtype == 'object']
    app_df = pd.get_dummies(app_df, columns= cat_col)
    return app_df


if __name__ == "__main__":
    # Bureau and Bureau Balance
    bb_df = pd.read_csv('datasets/bureau_balance.csv')
    bu_df = pd.read_csv('datasets/bureau.csv')
    app_df = pd.read_csv('datasets/application_train.csv')

    bureau_data_fe = bureau_fe(bu_df)
    bureau_and_bb_final = bureau_bureau_balance_fe(bureau_data_fe, bb_df)
    app_final = application(app_df)

