from fastapi import FastAPI
from typing import Optional
import uvicorn
import numpy as np
import pickle
import re
import string
import pandas as pd
from sklearn.pipeline import Pipeline
from Model import Text
import nltk
import joblib 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
s = set(stopwords.words('english'))
app = FastAPI(title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the CoronaVirus Tweets ",
    version="0.1",)

from pydantic import BaseModel
class Text(BaseModel):
    text : str

model3 = pickle.load(open('model.pkl',"rb"))
bow_vec = pickle.load(open('bow.pkl',"rb"))

# removes pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word,"", input_txt)
    return input_txt 

# removing the URL
def remove_URL(headline_text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', headline_text)

# removing the punctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    return text

# removing ASCII characters
def encoded(data):
    encoded_string = data.encode("ascii", "ignore")
    return encoded_string.decode()

# removing irrelevant characters
def reg(data):
    regex = re.compile(r'[\r\n\r\n]')
    return re.sub(regex, '', data)

#removing multi spaces
def spaces(data):
    res = re.sub(' +', ' ',data)
    return res



df['clean_t']=np.vectorize(remove_pattern)(df['Text'],'@[\w]*') #takes nested sequence of objects or numpy ararys

df["clean_t"]=df["clean_t"].apply(remove_URL)
df['clean_t'] = df['clean_t'].apply(remove_punctuations)
df['clean_t'] = df['clean_t'].apply(encoded)
df['clean_t'] = df['clean_t'].str.replace("[^a-zA-Z]", " ")    # removing the numeric characters
df["clean_t"]=df["clean_t"].str.lower()                        # to convert into lower case
df['clean_t'] = df['clean_t'].apply(reg) 
df['clean_t']=df['clean_t'].apply(spaces)
@app.get("/greet/{text}")
def greeting(text:str):
    return {"Hi {} welcome to twitter Sentmental Analysis".format(text)}


@app.post("/predict")
def Predict_Sentiment(item:Text):
    input_text = item.text
    data_frame=pd.DataFrame([input_text],columns=['text'])
    data_frame['text'] = data_frame['text'].apply(str)
    data_frame['text'] = np.vectorize(remove_pattern)(data_frame['text'],'@[\w]*')
    data_frame['text'] = data_frame["text"].apply(remove_URL)
    data_frame['text'] = data_frame['text'].apply(remove_punctuations)
    data_frame['text'] = data_frame['text'].str.replace("[^a-zA-Z]", " ")    # removing the numeric characters
    data_frame['text'] = data_frame['text'].str.lower()                        # to convert into lower case
    data_frame['text'] = data_frame['text'].apply(reg) 
    data_frame['text'] = data_frame['text'].apply(spaces)

    data_frame['text'] = data_frame['text'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
    data_frame['text'] = data_frame['text'].apply(clean_sent)
    data_frame['text'] = data_frame['text'].apply(lambda x: nltk.word_tokenize(x)) 
    data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    data_frame['text'] = data_frame['text'].apply(str)
    bow1 = bow_vec.transform(data_frame['text'])
    final = pd.DataFrame(bow1.toarray())
    my_prediction = model3.predict(final)
    output = int(my_prediction[0])
    # output dictionary
    sentiments = {-1: "Negative \U0001F61E", 1: "Positive \U0001F603",0: "Neutral \U0001F610"}

    # show results
    result = {sentiments[output]}
    return result
