import json
import requests
import sys

from pulsar import Function

class TweetToInfluxdb(Function):
    def __init__(self):
        self.url = "http://influx_db:8086/write?db=tweets"
        self.error_topic = "python_exceptions"

    def process(self, item, context):
        tweet = json.loads(item)
        sentiment = tweet['sentiment']
        if sentiment == "Positive":
            sent = 1
        elif sentiment == "Neutral":
            sent = 0.5
        else:
            sent = 0
        requests.post(
            self.url,
            data="tweets value={}".format(sent),
        )
