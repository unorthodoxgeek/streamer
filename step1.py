import json
from pulsar import Function

class FilterEnglish(Function):
    def __init__(self):
        self.twitter_topic = "tweets"
        self.foo_topic = "foo"
        self.bar_topic = "bar"

    def process(self, item, context):
    	json_item = json.loads(item)
    	if json_item['lang'] is not None and json_item['lang'] == 'en':
        	context.publish(self.foo_topic, json.dumps(json_item))
        else:
        	context.publish(self.bar_topic, json.dumps(json_item))

