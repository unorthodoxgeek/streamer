import json
import requests
import sys

from pulsar import Function

class TweetToInfluxdb(Function):
    def __init__(self):
        self.url = "http://influx_db:8086/write?db=tweets"
        self.error_topic = "python_exceptions"

    def process(self, item, context):
        try:
            print("!!!")
            #  tweet = json.loads(item)
            #  requests.post(
                #  self.url,
                #  data=f"tweets value={tweet['sentiment']}",
            #  )
        except:
            context.publish(self.error_topic, json.dumps(
                dict(function="write_to_influxdb", exception=f"{sys.exc_info()[0]}")
            ))
