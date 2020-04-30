import json
from pulsar import Function
from models.sentiment_analyzer import predict_sentiment

class AddSentiment(Function):
    def __init__(self):
        self.next_topic = "bar"

    def process(self, item, context):
    	json_item = json.loads(item)

        sentiment, score = predict_sentiment(json_item['text'])
        json_item['sentiment'] = sentiment
        json_item['score'] = score
    	context.publish(self.next_topic, json.dumps(json_item))
