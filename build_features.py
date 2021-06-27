#%%
import pandas as pd
# %%
def select_columns(df, columns):
    return df[columns]
# %%
def remove_columns(df, columns):
    return df.drop(columns, axis=1)
# %%
def drop_nas(df):
    return df.dropna()
# %%
def engineer_features(df):
    df['review'] = df[['review_title', 'review_text']].agg(' '.join, axis=1)
    df = remove_columns(df, ['review_title', 'review_text'])
    return df
# %%
def lower_text(df):
    for column in df.select_dtypes('object').columns:
        df[column] = df[column].str.lower()
    return df

def preprocess_recommendation_column(df):
    df['recommend_to_a_friend'] = df['recommend_to_a_friend'].map({'yes': 1, 'no': 0})
    return df

def preprocess_rating(df, threshold=3):
    df['sentiment'] = df['overall_rating'].mask(df['overall_rating'] < threshold, -1)
    df['sentiment'] = df['sentiment'].mask(df['sentiment'] == threshold, 0)
    df['sentiment'] = df['sentiment'].mask(df['sentiment'] > threshold, 1)
    return df

def remove_punctuation(df):
    df['review'] = df['review'].str.replace('\.|,|!|;|\?|r$|$', ' ', regex=True)
    return df

def preprocess_features(df):
    df = lower_text(df)
    df = preprocess_recommendation_column(df)
    df = preprocess_rating(df)
    df = remove_punctuation(df)
    return df
    