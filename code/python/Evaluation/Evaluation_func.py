import pandas as pd 
import numpy as np
import random
from datetime import date
from sklearn import metrics
from datetime import date
from sklearn.metrics import classification_report

def label_encoding(row, reference, col):

    num = abs(row[col])
    label = 10
    for maximum in reference.keys():
        if num < maximum:
            label = reference[maximum]
            break

    if row[col] < 0:
        label = label * (-1)

    return label


def sampling(df, time_period, threshold, sample_size):

    sample_df = df[(df['TRADE'] == 1) & (df['frequency_annual'] >= threshold) & (df['MTH_DATE'] >= '2015-07-07') & (df[f'trading_days_total_{time_period}_days'].notnull()) & (df['LABEL_CHURN'] == 'ACTIVE')]
    sample_index = random.sample(range(0, len(sample_df)), sample_size)
    active_sample = sample_df.iloc[sample_index].index.tolist()
    df['sampled_eval'] = 'N'
    df.loc[active_sample, 'sampled_eval'] = 'Y'
    sample_id = df.loc[active_sample, 'S_IDNO'].unique().tolist()

    return df, sample_id


def pick_history_data(df, time_period):

    history_list = []
    df = df.sort_values(by='MTH_DATE').reset_index(drop=True)
    sample_index = df[df.sampled_eval == 'Y'].index.tolist()
    ID = df.at[0, 'S_IDNO']

    for i, s in enumerate(sample_index):
        sample_df = df.loc[(s-time_period+1):(s)]
        sample_df['sample_no'] = f'{ID}_{i}'
        history_list.append(sample_df)

    history_df = pd.concat(history_list, axis=0)

    return history_df


def label_asset(df, reference_dict):

    for col in ['ST_ASSET', 'MP_ASSET', 'SS_ASSET']:
        df[col] = df[col].fillna(0)
        reference = reference_dict[col]
        # df[f'{col}_label'] = df[col].apply(lambda x: next((str(v) for k, v in reference.items() if x in k), 10))
        df[f'{col}_label'] = df.apply(label_encoding, reference=reference, col=col, axis=1)

    return df

def transfer_data(df):

    df = df.sort_values(by='MTH_DATE').reset_index(drop=True)
    target = df.iloc[-1]

    trade_amount_B = df[[f'AMT_B_{i}' for i in range(11)]]

    trade_amount_S = df[[f'AMT_S_{i}' for i in range(11)]]

    ST_asset = df[[f'ST_ASSET_{i}' for i in range(11)] + [f'ST_ASSET_{-1 * i}' for i in range(1, 11)]]

    MP_asset = df[[f'MP_ASSET_{i}' for i in range(11)] + [f'MP_ASSET_{-1 * i}' for i in range(1, 11)]]

    SS_asset = df[[f'SS_ASSET_{i}' for i in range(11)] + [f'SS_ASSET_{-1 * i}' for i in range(1, 11)]]

    if target['GENDER'] == 'M':
        target['GENDER'] = int(1)
    else:
        target['GENDER'] = int(0)
        
    demographic = target[['GENDER', 'AGE']].values.tolist()
    label = target['LABEL_CHURN']
    information = target['sample_no']

    return [trade_amount_B, trade_amount_S, ST_asset, MP_asset, SS_asset, demographic, information, label]

    
def transfer_google(df, data_type='numerical'):

    time_len = len(df)
    df = df.sort_values(by='MTH_DATE').reset_index(drop=True)
    df[['ST_ASSET', 'MP_ASSET', 'SS_ASSET']] = df[['ST_ASSET', 'MP_ASSET', 'SS_ASSET']].fillna(0)
    d = pd.DataFrame(df.iloc[-1][['GENDER', 'AGE', 'LABEL_CHURN', 'sample_no', 'MTH_DATE']]).transpose().reset_index(drop=True)
    
    assert data_type in ['numerical', 'categorical']

    d_list = [d]
    if data_type == 'numerical':
        for col in ['AMT_B', 'AMT_S', 'ST_ASSET', 'MP_ASSET', 'SS_ASSET', 'TRADE']:
            column_df = pd.DataFrame(df[col].values.flatten(), index=[f'{col}_day{i}' for i in range(1, 121)]).transpose()
            d_list.append(column_df)
    else:
        for col in ['AMT_B', 'AMT_S', 'ST_ASSET', 'MP_ASSET', 'SS_ASSET', 'TRADE']:
            column_df = pd.DataFrame(df[col].values.flatten(), index=[f'{col}_label_day{i}' for i in range(1, 121)]).transpose()
            d_list.append(column_df)

    df_final = pd.concat(d_list, axis=1)

    return df_final


def predict(row, threshold):
    
    if row['LABEL_CHURN_1_score'] > threshold:
        return 1
    elif row['LABEL_CHURN_0_score'] > threshold:
        return 0
    else:
        return -1


def Evaluation(data, threshold=0.5):
    
    d = data.copy()
    d['prediction'] = d.apply(predict, threshold=threshold, axis=1)
    d = d[d.prediction != -1]
    accuracy = metrics.accuracy_score(d['LABEL_CHURN'], d['prediction'])

    target_names = ['ACTIVE', 'CHURN']
    report = classification_report(d['LABEL_CHURN'].tolist(), d['prediction'].tolist(), target_names=target_names)
    up_support = len(data[data.LABEL_CHURN_1_score > threshold])
    down_support = len(data[data.LABEL_CHURN_0_score > threshold])
    up_support_ratio = len(data[data.LABEL_CHURN_1_score > threshold]) / len(data)
    down_support_ratio = len(data[data.LABEL_CHURN_0_score > threshold]) / len(data)

    return [accuracy, report, up_support, down_support, up_support_ratio, down_support_ratio]


def transfer_google_feature(df, period=10, time_period=120):

    time_len = len(df)
    df = df.sort_values(by='MTH_DATE').reset_index(drop=True)
    df['AMT_diff'] = df['AMT_B'] - df['AMT_S']    
    d = df.iloc[-1][['GENDER', 'AGE', 'LABEL_CHURN', 'sample_no', 'MTH_DATE', 'AMT_B', 'AMT_S', 'AMT_diff', 'ST_ASSET', 'MP_ASSET', 'SS_ASSET', f'trading_days_total_{time_period}_days', f'trading_days_amount_{time_period}_days']]
    
    
    for i in range(10, time_len+1, period):
        if i == 10:
            for col in ['AMT_B', 'AMT_S', 'AMT_diff', 'ST_ASSET', 'MP_ASSET', 'SS_ASSET']:
                d[f'{col}_total_day{i}'] = df[col].iloc[:i].sum()
        else:
            for col in ['AMT_B', 'AMT_S', 'AMT_diff', 'ST_ASSET', 'MP_ASSET', 'SS_ASSET']:
                d[f'{col}_total_day{i}'] = df[col].iloc[i-10:i].sum()
                d[f'{col}_total_day{i}_diff'] = df[col].iloc[i-10:i].sum() - d[f'{col}_total_day{i-10}']

    df_final = pd.DataFrame(d).transpose()

    return df_final