import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'UniversalBank.csv')

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')
    df['Experience'] = df['Experience'].clip(lower=0)
    df['Education_Label'] = df['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Prof'})
    df['Income_Band'] = pd.cut(df['Income'], bins=[0,50,100,150,200,300,600],
                               labels=['<50K','50-100K','100-150K','150-200K','200-300K','300K+'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70],
                             labels=['20s','30s','40s','50s','60s'])
    df['CCAvg_Band'] = pd.cut(df['CCAvg'], bins=[-0.1,1,3,6,10,20],
                              labels=['<1K','1-3K','3-6K','6-10K','10K+'])
    return df

def get_features_target(df: pd.DataFrame):
    drop_cols = ['Personal Loan', 'Education_Label', 'Income_Band', 'Age_Group', 'CCAvg_Band']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['Personal Loan']
    return X, y
