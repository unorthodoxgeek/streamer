import json
import requests
import pickle
from keras_preprocessing.sequence import pad_sequences

CONTRACTION_MAPPING = {
    "don't": "do not",
    "isn't": "is not",
    "aren't": "are not",
}

MAX_SEQ_LEN = 25


with open('models/sentiment_analyzer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def clean_text(text, mapping):
    replace_white_space = ["\n"]
    for s in replace_white_space:
        text = text.replace(s, " ")
    replace_punctuation = ["’", "‘", "´", "`", "\'", r"\'"]
    for s in replace_punctuation:
        text = text.replace(s, "'")
    
    # Random note: removing the URLs slightly degraded performance, it's possible the model learned that certain URLs were positive/negative
    # And was able to extrapolate that to retweets. Could also explain why re-training the Embeddings improves performance.
    # remove twitter url's
    #     text = re.sub(r"http[s]?://t.co/[A-Za-z0-9]*","TWITTERURL",text)
    mapped_string = []
    for t in text.split(" "):
        if t in mapping:
            mapped_string.append(mapping[t])
        elif t.lower() in mapping:
            mapped_string.append(mapping[t.lower()])
        else:
            mapped_string.append(t)
    return ' '.join(mapped_string)

def predict_sentiment(text):
    text_vector = [clean_text(text, CONTRACTION_MAPPING)]
    text_vector = tokenizer.texts_to_sequences(text_vector)
    text_vector = pad_sequences(text_vector, maxlen=MAX_SEQ_LEN)

    url = "http://tensorflow_serving:8501/v1/models/twitter_sentiment/versions/1:predict"
    prediction = requests.post(url, json={"instances": text_vector.tolist()})
    arr_prediction = prediction.json()["predictions"][0]
    max_prediction = max(arr_prediction)
    index = arr_prediction.index(max_prediction)
    if max_prediction > 0.75:
        if index == 0:
            return ("Negative", max_prediction)
        return ("Positive", max_prediction)
    return ("Unknown", max_prediction)