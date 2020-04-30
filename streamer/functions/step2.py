import json
from pulsar import Function

class TweetToRecord(Function):
    def __init__(self):
        self.twitter_topic = "tweets"
        self.tweet_records_topic = "tweet_records"

    def process(self, item, context):
        tweet = json.loads(item)
        record = dict(
            user_id=tweet["user"]["id"],
            lang=tweet["lang"],
            text=tweet["text"],
        )
        context.publish(self.tweet_records_topic, json.dumps(record))

