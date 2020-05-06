from flask import Flask, request
from models.sentiment_analyzer import predict_sentiment

app = Flask(__name__)

@app.route('/predict_sentiment', methods=['POST'])
def sentiment_prediction():
    text = request.json["text"]
    sentiment, confidence = predict_sentiment(text)
    return dict(sentiment=sentiment, confidence=confidence)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')