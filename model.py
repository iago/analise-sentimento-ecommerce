#%%
import pickle
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, classification_report

from make_dataset import load_data
from build_features import select_columns, drop_nas, engineer_features, preprocess_features
#%%
PATH = 'https://github.com/b2wdigital/b2w-reviews01/raw/master/B2W-Reviews01.csv'
COLUMNS_TO_MAINTAIN = ['overall_rating', 'recommend_to_a_friend', 'review_title', 'review_text']
#%%
reviews = load_data(PATH)
reviews = select_columns(reviews, COLUMNS_TO_MAINTAIN)
reviews = drop_nas(reviews)
reviews = engineer_features(reviews)
reviews = preprocess_features(reviews)
#%%
def sample_data(df, target, sample_size):
    df = df[df[target] != 0]
    df_pos = df[df[target] == 1].sample(sample_size)
    df_neg = df[df[target] == -1].sample(sample_size)
    df_neg[target] = 0

    df = pd.concat([df_pos, df_neg])
    df = df.sample(len(df))
    df = df.reset_index()
    return df

reviews = sample_data(reviews, 'sentiment', 35758)
#%%
X = reviews['review'].values
y = reviews['sentiment'].values
#%%
def split_data(X, y, test_size=0.33):
    return train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = split_data(X, y)
#%%
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
#%%
def standardize_data(scaler, X_train, X_test):
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

X_train, X_test = standardize_data(MaxAbsScaler(), X_train, X_test)
#%%
def instatiate_log_reg(class_weight='balanced', scoring='f1', **kwargs):
    log_reg = LogisticRegressionCV(class_weight=class_weight,
    scoring=scoring, solver='liblinear', **kwargs)
    return log_reg

def instatiate_knn(**kwargs):
    knn = KNeighborsClassifier(**kwargs)
    return knn

model = instatiate_log_reg()
#%%
def train_model(X_train, y_train, model):
    return model.fit(X_train, y_train)

clf = train_model(X_train, y_train, model)
#%%
def make_prediction(model, X):
    return model.predict(X)

y_pred = make_prediction(clf, X_test)
#%%
def print_metrics(model, y_test, y_pred, X_test):
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('F1-Score: ', f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(model, X_test, y_test, labels=[0, 1]);

print_metrics(clf, y_test, y_pred, X_test)
# %%
def export_file(file, filename):
    pickle.dump(file, open(filename, 'wb'))
# %%
