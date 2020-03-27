import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, Embedding, Activation, Flatten, LeakyReLU, Conv2D, concatenate, BatchNormalization, MaxPooling2D, AveragePooling1D, Reshape
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn import metrics
from sklearn.metrics import classification_report

# Prediction
def predict(original_df, threshold, model):
    
    prediction = model.predict(original_df, batch_size=128)
    pred = [1 if d[1] >= threshold else 0 for d in prediction]
    
    return pred

# Evaluation
def Evaluation(original_df, original_label, model, threshold=0.5):
    
    Y = [0 if element == 'ACTIVE' else 1 for element in original_label]
    prediction = predict(original_df, threshold=threshold, model=model)
    accuracy = metrics.accuracy_score(Y, prediction)

    target_names = ['ACTIVE', 'CHURN']
    report = classification_report(Y, prediction, target_names=target_names)

    return [accuracy, report]


# Concat prediction with original dataframe
def concat_original_df(prediction_df, information, original_path, model):

    prediction = model.predict(prediction_df, batch_size=128)
    d = pd.DataFrame(prediction, columns=['first_stage_0_score', 'first_stage_1_score'])
    d['sample_no'] = information

    original_df = pd.read_csv(original_path)
    df = pd.merge(d, original_df, on='sample_no', how='inner')

    return df


# Return churn/active/false positive/false negative data
def subset_df(df, threshold=0.5):

    df_list = [group[1] for group in df.groupby(df['sample_no'])]
    df_list = [df.sort_values(by='MTH_DATE').reset_index(drop=True) for df in df_list]
    false_positive = pd.concat([df for df in df_list if (df['LABEL_CHURN'].iloc[-1] == 'ACTIVE') and (df['first_stage_1_score'].iloc[-1] >= threshold)], axis=0)
    false_negative = pd.concat([df for df in df_list if (df['LABEL_CHURN'].iloc[-1] == 'CHURN') and (df['first_stage_0_score'].iloc[-1] >= threshold)], axis=0)
    churn = pd.concat([df for df in df_list if (df['LABEL_CHURN'].iloc[-1] == 'CHURN')], axis=0)
    active = pd.concat([df for df in df_list if (df['LABEL_CHURN'].iloc[-1] == 'ACTIVE')], axis=0)

    return false_positive, false_negative, churn, active


def false_positive(df, threshold=0.5):

    df_list = [group[1] for group in df.groupby(df['sample_no'])]
    false_positive = pd.concat([df for df in df_list if (df['LABEL_CHURN'].iloc[-1] == 'ACTIVE') and (df['first_stage_1_score'].iloc[-1] >= threshold)], axis=0)

    return false_positive