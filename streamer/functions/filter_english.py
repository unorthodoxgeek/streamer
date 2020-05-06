import json
from pulsar import Function

class FilterEnglish(Function):
    def __init__(self):
        self.twitter_topic = "tweets"
        self.english_topic = "english_tweets"

    def process(self, item, context):
    	json_item = json.loads(item)
    	if json_item['lang'] is not None and json_item['lang'] == 'en':
        	context.publish(self.english_topic, json.dumps(json_item))
