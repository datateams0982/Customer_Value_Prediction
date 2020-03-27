import pandas as pd 
import numpy as np
import random
from datetime import date

##Data binning function
def label_encoding(row, reference, col):

    '''
    input: reference dictionary, the feature column for binning, data row
    '''

    num = abs(row[col])
    label = 10
    for maximum in reference.keys():
        if num < maximum:
            label = reference[maximum]
            break

    if row[col] < 0:
        label = label * (-1)
    
    return label


#Trading data preprocess function
def trading_days(df, time_period, timedf, reference_dict):

    '''
    input: trading data of a single customer, the period of time to trace back, a dataframe of all trading dates, data binning reference
    output: processed data
    '''

    start_date = df['MTH_DATE'].min()
    churn_date = df[df.LABEL_CHURN == 'CHURN']['MTH_DATE']
    timedf = timedf[timedf['MTH_DATE'] >= start_date]

    #data binning
    for col in ['AMT_B', 'AMT_S']:
        reference = reference_dict[col]
        df[f'{col}_label'] = df.apply(label_encoding, reference=reference, col=col, axis=1)

    #Cut off data if customer churns
    if len(churn_date) != 0:
        timedf_list = []
        for churn in churn_date:
            t_df = timedf[timedf['MTH_DATE'] <= churn]
            timedf_list.append(t_df)
            timedf = timedf[timedf['MTH_DATE'] > churn]

        timedf = pd.concat(timedf_list)

    #Filling missing dates
    full_df = pd.merge(df, timedf, on='MTH_DATE', how='right')
    full_df = full_df.sort_values(by='MTH_DATE').reset_index(drop=True)

    if len(full_df[full_df['S_IDNO'].isnull()]) != 0:
        interpolate_col = ['S_IDNO', 'GENDER', 'AGE', 'LABEL_CHURN', 'BIRTH_ADDRESS', 'LIVING_ADDRESS']
        zero_col = ['AMT_B', 'AMT_S', 'TRADE', 'AMT_B_label', 'AMT_S_label']

        for col in interpolate_col:
            full_df[col] = full_df[col].interpolate(method="pad")

        for col in zero_col:
            full_df[col] = full_df[col].fillna(0)

    #Features for EDA
    full_df[f'trading_days_total_{time_period}_days'] = full_df['TRADE'].rolling(window=time_period).sum()
    full_df[f'trading_days_amount_{time_period}_days'] = full_df['AMT_B'].rolling(window=time_period).sum() + full_df['AMT_S'].rolling(window=time_period).sum()
    full_df['frequency_annual'] = full_df['TRADE'].rolling(window=250).sum()
    full_df['amount_annual'] =  full_df['AMT_B'].rolling(window=250).sum() + full_df['AMT_S'].rolling(window=360).sum()
    full_df['YEAR'] = full_df['MTH_DATE'].apply(lambda x: str(x)[:4])

    return full_df


#Down sampling function for active customers
def sampling(df, time_period, threshold, multiple=1):

    '''
    input: processed trading data, the period of time to trace back, the threshold of frequency, the multiple of active data comparing to churn data
    output: data with a column indicating sampled or not, customer IDs that are sampled
    '''
    
    #Filter data
    sample_df = df[(df['TRADE'] == 1) & (df['frequency_annual'] >= threshold) & (df['MTH_DATE'] >= '2015-07-07') & (df[f'trading_days_total_{time_period}_days'].notnull())]
    active_df = sample_df[(sample_df['LABEL_CHURN'] == 'ACTIVE')]
    churn_df = sample_df[sample_df['LABEL_CHURN'] == 'CHURN']

    #Random sampling
    sample_index = random.sample(range(0, len(active_df)), multiple*len(churn_df))
    sample_index_list = active_df.iloc[sample_index].index.tolist()

    sample_list = sample_index_list + churn_df.index.tolist()
    df['sampled'] = 'N'
    df.loc[sample_list, 'sampled'] = 'Y'
    sample_id = df.loc[sample_list, 'S_IDNO'].unique().tolist()

    return df, sample_id


#Trace back based on sample data
def pick_history_data(df, time_period):

    '''
    input: sampled trading data of a single customer, the period of time to trace back
    output: traced back data
    '''

    history_list = []
    df = df.sort_values(by='MTH_DATE').reset_index(drop=True)
    sample_index = df[df.sampled == 'Y'].index.tolist()
    ID = df.at[0, 'S_IDNO']

    for i, s in enumerate(sample_index):
        sample_df = df.loc[(s-time_period+1):(s)]
        sample_df['sample_no'] = f'{ID}_{i}'
        history_list.append(sample_df)

    history_df = pd.concat(history_list, axis=0)

    return history_df


#Binning asset data
def label_asset(df, reference_dict):

    for col in ['ST_ASSET', 'MP_ASSET', 'SS_ASSET']:
        df[col] = df[col].fillna(0)
        reference = reference_dict[col]
        # df[f'{col}_label'] = df[col].apply(lambda x: next((str(v) for k, v in reference.items() if x in k), 10))
        df[f'{col}_label'] = df.apply(label_encoding, reference=reference, col=col, axis=1)

    return df

#Transfer data to list and separate training/validation/testing data for CNN
def transfer_data(df, train_date, val_date):

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

    if target['MTH_DATE'] < train_date:
        return [trade_amount_B, trade_amount_S, ST_asset, MP_asset, SS_asset, demographic, information, label, 'train']
    elif target['MTH_DATE'] < val_date:
        return [trade_amount_B, trade_amount_S, ST_asset, MP_asset, SS_asset, demographic, information, label, 'val']
    else:
        return [trade_amount_B, trade_amount_S, ST_asset, MP_asset, SS_asset, demographic, information, label, 'test']


#Transfer data for google autoML
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


