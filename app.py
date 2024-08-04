from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from fastapi.middleware.cors import CORSMiddleware

nltk.download('stopwords')
nltk.download('punkt')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific domain of your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
with open("Email_Classifier_NB_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Preprocessing class
class EmailText(BaseModel):
    text: str

def preprocess_text(text):
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()

    # Tokenize
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords and stem the words
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stopwords_set]
    return " ".join(cleaned_tokens)

@app.post("/predict/")
async def predict_spam(email: EmailText):
    preprocessed_text = preprocess_text(email.text)
    prediction = model.predict([preprocessed_text])[0]
    return {"prediction": prediction}

# This is only for the local test, remove if deploying to production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
