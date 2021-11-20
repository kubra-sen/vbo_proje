"""readme:
https://becominghuman.ai/lightgbm-on-home-credit-default-risk-prediction-5b17e68a6e9
https://towardsdatascience.com/a-machine-learning-approach-to-credit-risk-assessment-ba8eda1cd11f
https://medium.com/thecyphy/home-credit-default-risk-part-1-3bfe3c7ddd7a
"""

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

# Application Train EDA
app_df = pd.read_csv('vbo_proje/datasets/application_train.csv')
app_df.head()

# Bureau and Bureau balance EDA
########### Analyzing feature dtypes ##################
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)










