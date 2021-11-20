
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from helpers.eda import *
from helpers.data_prep import *

# Bureau and Bureau Balance
bu_df = pd.read_csv('datasets/bureau.csv')
bb_df = pd.read_csv('datasets/bureau_balance.csv')

########## BUREAU #############
# Analyzing feature dtypes
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(bu_df)

# Missing Values
missing_df = missing_values_table(bu_df)
# These features should be zero if missing
bu_df['AMT_CREDIT_SUM'] = bu_df['AMT_CREDIT_SUM'].fillna(0)
bu_df['AMT_CREDIT_SUM_DEBT'] = bu_df['AMT_CREDIT_SUM_DEBT'].fillna(0)
bu_df['AMT_CREDIT_MAX_OVERDUE'] = bu_df['AMT_CREDIT_MAX_OVERDUE'].fillna(0)
# Remaining features are filled with the mean because if there is a loan there should be a limit
bu_df = bu_df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

# Feature Engineering
# Ref: https://www.kaggle.com/c/home-credit-default-risk/discussion/57750

# Number of past loans per customer
bu_df['past_loan_count'] = bu_df.groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]
# Number of types of past loans per customer


# Average number of Past loans per type of loan per customer
# Number of Loans per Customer


# Number of types of Credit loans for each Customer
