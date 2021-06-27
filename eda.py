#%%
import pandas as pd
import seaborn as sns
#%%
PATH = 'https://github.com/b2wdigital/b2w-reviews01/raw/master/B2W-Reviews01.csv'
COLUMNS_TO_REMOVE = ['submission_date', 'reviewer_id', 'product_id',
    'product_name', 'product_brand', 'site_category_lv1', 'site_category_lv2']
TARGETS = {'overall': 'overall_rating', 'recommend': 'recommend_to_a_friend'}
CATEGORICAL_FEATURES = ['reviewer_gender', 'reviewer_state']
NUMERICAL_FEATURES = ['reviewer_birth_year']
#%%
def load_data(path):
    return pd.read_csv(path, delimiter=';')

reviews = load_data(PATH)
#%%
reviews.info()
#%%
reviews.describe()
#%%
def remove_columns(df, columns_to_remove):
    removed_columns = []
    columns_not_removed = []

    for column in columns_to_remove:
        try:
            df = df.drop(column, axis=1)
        except KeyError:
            columns_not_removed.append(column)
        else:
            removed_columns.append(column)
     
    if len(columns_not_removed) > 0:
        print('{} column(s) not removed'.format(', '.join(columns_not_removed)))
    if len(removed_columns) > 0:
        print('{} column(s) removed'.format(', '.join(removed_columns)))
    return df

reviews = remove_columns(reviews, COLUMNS_TO_REMOVE)
#%%
def engineer_features(df):
    df['review'] = df[['review_title', 'review_text']].agg(' '.join, axis=1)
# %%
def missing_values(df, normalize=False):
    normalize_factor = 1
    if normalize:
        normalize_factor = len(df)
    print(df.isna().sum() / normalize_factor)
# %%
def drop_nas(df):
    return df.dropna()
# %%
def plot_boxplot(x, y):
    return sns.boxplot(x=x, y=y);
# %%
