from flask import Flask, request
from models.sentiment_analyzer import predict_sentiment
from models.time_series import get_time_series

app = Flask(__name__)

@app.route('/predict_sentiment', methods=['POST'])
def sentiment_prediction():
    text = request.json["text"]
    sentiment, confidence = predict_sentiment(text)
    return dict(sentiment=sentiment, confidence=confidence)

@app.route('/get_sentiment')
def get_sentiment():
	return get_time_series()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')