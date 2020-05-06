import json
from pulsar import Function
import requests

class AddSentiment(Function):
    def __init__(self):
        self.next_topic = "tweets_with_sentiment"

    def process(self, item, context):
    	json_item = json.loads(item)

    	url = "http://model_serving:5000/predict_sentiment"
        prediction = requests.post(url, json={'text': json_item['text']})
        json_item.update(prediction.json())
    	context.publish(self.next_topic, json.dumps(json_item))
