#%%
import pickle
import uvicorn
from fastapi import FastAPI
#%%
app = FastAPI()

MODEL_FILEPATH = 'artifacts/20210402_041538_Logistic_Regression.sav'
VECTORIZER_FILEPATH = 'artifacts/20210402_041538_Vectorizer.sav'
#%%
def load_file(filename):
    file = pickle.load(open(filename, 'rb'))
    return file
#%%
def load_model_vectorizer(model_filepath, vectorizer_filepath):
    model = load_file(model_filepath)
    vectorizer = load_file(vectorizer_filepath)
    return model, vectorizer
#%%
def predict(X, model_filepath=MODEL_FILEPATH, vectorizer_filepath=VECTORIZER_FILEPATH):
    model, vectorizer = load_model_vectorizer(model_filepath, vectorizer_filepath)
    return model.predict(vectorizer.transform(X))
#%%
def get_responses(predictions):
    def get_response(prediction):
        return "Avaliação positiva" if prediction == 1 else "Avaliação negativa"
    responses = map(get_response, predictions)
    return responses
# %%
@app.get('/')
def root():
    return {'message': 'Boas-vindas à API de Análise de Sentimento de Reviews de Produtos'}
# %%
@app.get('/predict/{review}')
def get_sentiment(review: str) -> dict:
    predictions = predict(review.split('\n'))
    responses = get_responses(predictions)
    return {'sentiments': list(responses)}
# %%
if __name__ == '__main__':
    uvicorn.run('api:app', host='127.0.0.1', port=8000, reload=True)
    