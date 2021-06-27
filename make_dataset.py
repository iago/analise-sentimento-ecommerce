#%%
import pandas as pd
# %%
PATH = 'https://github.com/b2wdigital/b2w-reviews01/raw/master/B2W-Reviews01.csv'
#%%
def load_data(path, delimiter=';'):
    return pd.read_csv(path, delimiter=delimiter)